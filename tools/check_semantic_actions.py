"""Inspect candidate input sequences; keyboard injection requires --execute.

This is a manual control check, not a detector, policy runner, or move validator.
The declared ground/air state must match the game during an executed check.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.input import NullInputController
from control.semantic_actions import ACTIONS, BY_NAME, ControlContext, contract
from control.semantic_executor import SemanticExecutor


class _LoggedInput:
    def __init__(self, sink) -> None:
        self.sink = sink
        self.now = 0.
        self.events: list[dict] = []

    def set_pressed(self, keys: set[str]) -> None:
        self.sink.set_pressed(keys)
        self.events.append({"t": self.now, "event": "held", "keys": sorted(keys)})

    def key_down(self, key: str) -> None:
        self.sink.key_down(key)
        self.events.append({"t": self.now, "event": "down", "key": key})

    def key_up(self, key: str) -> None:
        self.sink.key_up(key)
        self.events.append({"t": self.now, "event": "up", "key": key})

    def reset(self) -> None:
        self.sink.reset()
        self.events.append({"t": self.now, "event": "reset"})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List candidate actions; no input injection.")
    parser.add_argument("--action", choices=list(BY_NAME))
    parser.add_argument("--context", choices=["unknown", "ground", "air"], default="unknown",
                        help="Manually asserted context, NOT a sensed game state.")
    parser.add_argument("--weapon-state", choices=["unknown", "armed", "unarmed"], default="unknown")
    parser.add_argument("--mirrored", action="store_true", help="Canonical TOWARD is physical left.")
    parser.add_argument("--execute", action="store_true", help="Inject this one sequence into the foreground game.")
    parser.add_argument("--delay", type=float, default=5., help="Seconds to switch to the game before --execute.")
    parser.add_argument("--output", type=Path, help="Optional new JSON trace; existing files are never overwritten.")
    args = parser.parse_args(argv)
    if args.list:
        for action in ACTIONS:
            print(f"{action.id:2d} {action.name:28s} {action.context}")
        return
    if not args.action:
        parser.error("Choose --action or --list")
    if not math.isfinite(args.delay) or not 0 <= args.delay <= 60:
        parser.error("--delay must be finite and between 0 and 60 seconds")
    if BY_NAME[args.action].context != "any" and args.context == "unknown":
        parser.error("Contextual actions require an explicit --context ground/air")
    if args.execute and BY_NAME[args.action].possession is not None and args.weapon_state == "unknown":
        parser.error("Live pickup/throw checks require an explicit --weapon-state armed/unarmed")
    context = ControlContext(
        grounded={"unknown": None, "ground": True, "air": False}[args.context],
        armed={"unknown": None, "armed": True, "unarmed": False}[args.weapon_state],
        mirrored=args.mirrored,
    )
    # Open exclusively before creating a live controller. Bad paths must fail
    # before any keys are sent, not at the end of an unrecorded experiment.
    destination = args.output.open("x", encoding="utf-8") if args.output else nullcontext(None)
    with destination as stream:
        sink = NullInputController()
        if args.execute:
            from control.input import PyDirectInputController
            print(f"ONE input sequence: {args.action}. Context is MANUALLY asserted: {args.context}.")
            print(f"Switch to Brawlhalla within {args.delay:g}s; Ctrl+C releases keys.")
            time.sleep(args.delay)
            sink = PyDirectInputController()
        logged = _LoggedInput(sink)
        outcomes = []
        start = time.perf_counter()
        with SemanticExecutor(logged) as executor:
            result = executor.submit(args.action, context, now=0.)
            outcomes.append({"t": 0., **asdict(result)})
            tick = 0
            while executor.busy:
                if args.execute:
                    time.sleep(.005)
                    elapsed = time.perf_counter() - start
                else:
                    tick += 1
                    elapsed = tick * .005
                logged.now = elapsed
                result = executor.advance(context, now=elapsed)
                if result.status != "pending":
                    outcomes.append({"t": elapsed, **asdict(result)})
        report = {
            "simulated": not args.execute,
            "game_move_verified": False,
            "context_source": "manual_assertion",
            "context": asdict(context),
            "action_contract": contract(),
            "executor_version": executor.VERSION,
            "timing": asdict(executor.timing),
            "events": logged.events,
            "outcomes": outcomes,
        }
        if stream is not None:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(json.dumps({key: report[key] for key in ("simulated", "game_move_verified", "outcomes")}, indent=2))
        print("Input trace only. Verify the actual animation in-game before enabling semantic training.")


if __name__ == "__main__":
    main()
