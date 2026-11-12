"""Let a policy play the live game and report how it did (and whether it decided in time).

Start Brawlhalla in training mode (Wu Shang vs a Mordex bot, Brawlhaven small, camera Fixed, no
overlays), not paused, with the bot at the level you name. Run this and click into the game during
the countdown. Press **F10** to stop early.

    python scripts/play.py runs/bc/medium_v4/policy.zip --difficulty medium --minutes 5
    python scripts/play.py runs/bc/medium_v4/policy.zip --difficulty medium --minutes 5 --blocks 6   # gate N1
    python scripts/play.py random --difficulty easy --minutes 1       # no policy: random actions

``--blocks n`` plays n blocks of ``--minutes`` one after the other without stopping and reports each
block and their spread: how much the same policy's result varies from block to block is the noise
floor (gate N1) that any comparison of two policies has to beat. Virtual matches are races to 3
stocks, counted over the whole run.

The policy samples its actions as it was trained to (--greedy takes the most likely one instead; a
cloned policy then presses few buttons). Writes results/play/<name>.md and runs/play/<name>.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import brawl.game.capture as capture  # noqa: E402  first: the screen duplicator must exist before torch loads

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402

from brawl.paths import RESULTS, RUNS  # noqa: E402

DETECTOR = RUNS / "detector" / "armed_yolo11s-pose_960" / "weights" / "best.engine"
MATCH = 3  # stocks to win a virtual match


def play(env, choose, *, block_s: float, blocks: int, stop=lambda: False, now=time.perf_counter, log=print) -> dict:
    """Step ``env`` with ``choose(obs) -> action`` for ``blocks`` blocks of ``block_s`` seconds.
    Returns the per-block totals, the virtual matches, and the timing of every decision."""
    obs, _ = env.reset()
    new = lambda: {"dealt": 0.0, "taken": 0.0, "scored": 0, "lost": 0, "reward": 0.0, "frames": 0, "decisions": 0}  # noqa: E731
    rows, think, sense, present = [new()], [], [], []
    race, matches, late, buttons, t0 = [0, 0], [0, 0], 0, 0, now()
    while not stop():
        elapsed = now() - t0
        if elapsed >= block_s * blocks:
            break
        if elapsed >= block_s * len(rows):
            b = rows[-1]
            log(f"block {len(rows)}: stocks {b['scored']} - {b['lost']}, damage {b['dealt']:.0f} / {b['taken']:.0f}")
            rows.append(new())
        t = now()
        action = choose(obs)
        think.append(now() - t)
        obs, reward, _, _, info = env.step(action)
        e, timing, b = info["events"], info["timing"], rows[-1]
        b["dealt"] += e.damage_dealt
        b["taken"] += e.damage_taken
        b["scored"] += e.ko_scored
        b["lost"] += e.ko_lost
        b["reward"] += reward
        b["frames"] += timing.frames
        b["decisions"] += 1
        late += timing.late_frames
        buttons += int(action[1] != 0)
        sense.append(info["sense_s"])
        present.append(("wu_shang" in info["sense"].fighters, "mordex" in info["sense"].fighters))
        race[0] += e.ko_scored
        race[1] += e.ko_lost
        if max(race) >= MATCH:  # a virtual match ends; the next one starts from nothing
            matches[race[1] > race[0]] += 1
            race = [0, 0]
    return {"blocks": rows, "matches_won": matches[0], "matches_lost": matches[1], "late_frames": late, "buttons": buttons,
            "think_s": think, "sense_s": sense, "present": present, "seconds": now() - t0}


def report(r: dict, name: str, title: str, k: int, calls: dict, steps: int) -> list[str]:
    rows = r["blocks"]
    total = {key: sum(b[key] for b in rows) for key in rows[0]}
    n = max(total["decisions"], 1)
    cost = (np.array(r["think_s"]) + np.array(r["sense_s"])) * 1000
    budget = k / 60 * 1000
    per_min = lambda b, key: b[key] / max(b["frames"] / 3600, 1e-9)  # noqa: E731
    played = r["matches_won"] + r["matches_lost"]
    lines = [f"# Play: {name}", "",
             f"{datetime.now():%Y-%m-%d %H:%M}. {title} {n} decisions in {r['seconds']:.0f} s "
             f"({n / max(r['seconds'], 1e-9):.1f} per second; {total['frames'] / 60:.0f} s of game time).", "",
             "| Measure | Value |", "|---|---|",
             f"| Stocks taken - lost | {total['scored']} - {total['lost']} |",
             f"| Virtual matches (first to {MATCH} stocks) won - lost | {r['matches_won']} - {r['matches_lost']}"
             + (f" ({r['matches_won'] / played:.0%})" if played else "") + " |",
             f"| Damage dealt / taken | {total['dealt']:.0f} / {total['taken']:.0f} |",
             f"| Reward per minute of game time | {per_min(total, 'reward'):.2f} |",
             f"| Decisions with a button | {r['buttons'] / n:.1%} |",
             f"| Wu Shang / Mordex present | {np.mean([a for a, _ in r['present']]):.1%} / "
             f"{np.mean([b for _, b in r['present']]):.1%} |",
             f"| Perception + policy per decision (ms): median / p95 / max | {np.median(cost):.1f} / "
             f"{np.percentile(cost, 95):.1f} / {cost.max():.1f} (budget {budget:.0f}) |",
             f"| of which policy (ms): median / p95 | {np.median(r['think_s']) * 1000:.2f} / "
             f"{np.percentile(r['think_s'], 95) * 1000:.2f} |",
             f"| Decisions over the budget | {np.mean(cost > budget):.2%} |",
             f"| Frames whose keys were set late (realtime) | {r['late_frames']} of {total['frames']} |",
             f"| Each part once per step (advance, grab, perceive, build) | {calls} for {steps} steps |"]
    if len(rows) > 1:
        full = [b for b in rows if b["frames"] >= 0.9 * rows[0]["frames"]]  # a block cut short by F10 is left out
        v = {key: np.array([per_min(b, key) for b in full]) for key in ("reward", "scored", "lost", "dealt", "taken")}
        net = v["scored"] - v["lost"]
        se = lambda a: a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")  # noqa: E731
        lines += ["", "| Block | Stocks taken - lost | Damage dealt / taken | Reward per minute |", "|---|---|---|---|",
                  *(f"| {i + 1} | {b['scored']} - {b['lost']} | {b['dealt']:.0f} / {b['taken']:.0f} | {per_min(b, 'reward'):.2f} |"
                    for i, b in enumerate(rows)), "",
                  f"Over the {len(full)} full blocks, per minute of game time (mean ± spread between blocks; standard "
                  f"error of the mean):", "",
                  f"- Reward: {v['reward'].mean():.2f} ± {v['reward'].std(ddof=1):.2f} (standard error {se(v['reward']):.2f})",
                  f"- Net stocks: {net.mean():.2f} ± {net.std(ddof=1):.2f} (standard error {se(net):.2f})",
                  f"- Damage dealt: {v['dealt'].mean():.0f} ± {v['dealt'].std(ddof=1):.0f}; taken: "
                  f"{v['taken'].mean():.0f} ± {v['taken'].std(ddof=1):.0f}",
                  f"- Two policies each played for {len(full)} such blocks differ beyond chance (about 2 standard errors "
                  f"of the difference) when their reward per minute differs by more than "
                  f"{2 * np.sqrt(2) * se(v['reward']):.2f}.",
                  "- Blocks follow each other without a pause, so damage carries from one into the next."]
    return lines


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("policy", help="policy.zip from scripts/train_bc.py, or 'random'")
    p.add_argument("--difficulty", required=True, help="the bot's level, as set in the game (for the report)")
    p.add_argument("--minutes", type=float, default=5.0, help="length of one block")
    p.add_argument("--blocks", type=int, default=1, help="blocks played one after the other (gate N1: 6 or more)")
    p.add_argument("--regime", choices=("realtime", "stepped"), default="realtime")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--greedy", action="store_true", help="take the most likely action instead of sampling")
    p.add_argument("--detector", type=Path, default=DETECTOR)
    p.add_argument("--name", default=None, help="report name (default: policy folder, level and regime)")
    p.add_argument("--countdown", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    print(capture.status())
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")

    import torch

    from brawl import learn
    from brawl.brawlenv import BrawlEnv
    from brawl.game.clock import RealtimeClock, SteppedClock
    from brawl.game.counter import FrameCounter
    from brawl.game.keys import DirectInput, KeyReader, Keys
    from brawl.sense.detectors import Yolo
    from brawl.sense.perception import Perception

    torch.set_num_threads(1)  # a small MLP on one core; more threads only add waiting
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    policy = None if args.policy == "random" else learn.load_policy(Path(args.policy), "cpu")
    perception = Perception(Yolo(args.detector, conf=0.05), step=args.k)
    screen = capture.Screen(timeout_s=0.05)
    keys = Keys(DirectInput(), tap_s=0.01)
    clock = (RealtimeClock(keys) if args.regime == "realtime"
             else SteppedClock(keys, counter=FrameCounter(screen), key_gap_s=0.01))
    reader = KeyReader()

    def choose(obs):
        if policy is None:
            return int(rng.integers(9)), int(rng.integers(6))
        with torch.no_grad():
            return policy._predict(torch.as_tensor(obs[None]), deterministic=args.greedy)[0].numpy()

    for i in range(args.countdown, 0, -1):
        print(f"click into the game window... {i}", flush=True)
        time.sleep(1)
    perception(screen.grab(fresh=True).image, 0)  # warm the detector up before the clock starts
    clock.start()
    env = BrawlEnv(clock, screen, perception, k=args.k)
    print(f"playing {args.blocks} x {args.minutes:g} min: press F10 to stop", flush=True)
    try:
        r = play(env, choose, block_s=args.minutes * 60, blocks=args.blocks, stop=reader.stop_requested)
    finally:
        env.close()
        keys.release_all()
    if not r["think_s"]:
        print("stopped before the first decision: nothing to report")
        return 1
    name = args.name or (f"{Path(args.policy).parent.name if policy else 'random'}_{args.difficulty}_{args.regime}"
                         + (f"_x{args.blocks}" if args.blocks > 1 else ""))
    title = (f"Policy {args.policy} ({'most likely action' if args.greedy else 'sampled'}), bot {args.difficulty}, "
             f"{args.regime}, k = {args.k}, {args.blocks} x {args.minutes:g} min.")
    lines = report(r, name, title, args.k, env.calls, env.steps)
    path = RESULTS / "play" / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    record = RUNS / "play" / f"{name}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"policy": args.policy, "difficulty": args.difficulty, "regime": args.regime, "k": args.k,
                                  "minutes": args.minutes, "greedy": args.greedy, "seed": args.seed,
                                  "blocks": r["blocks"], "matches_won": r["matches_won"],
                                  "matches_lost": r["matches_lost"]}, indent=1), encoding="utf-8")
    print("\n".join(lines[4:]), f"\n-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
