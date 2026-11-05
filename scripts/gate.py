"""Run one Stage 0 gate against the live game and write results/gates/<gate>.md.

Start Brawlhalla in training mode (Wu Shang vs a Mordex bot, Brawlhaven), unpaused, then run a
gate and click into the game window during the countdown. The script presses keys in whatever
window has focus.

    python scripts/gate.py t1            # keys: F5/F6/F7 and every fight key reach the game
    python scripts/gate.py t2            # stepping: one F6 = exactly one new frame; held vs re-armed keys
    python scripts/gate.py t3            # determinism: rewind and replay give identical frames
    python scripts/gate.py t4            # throughput: steps and rewind frames per second
    python scripts/gate.py t5            # difficulty macro (saves screenshots for you to confirm)

Options: --rearm (press keys together with every F6), --tap-ms (key hold time), --countdown.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # sys and pathlib do not load torch

import brawl.game.capture as capture  # noqa: E402  first: the screen duplicator must exist before torch loads

import argparse
import random
import sys
import time
from datetime import datetime

import numpy as np

from brawl.game.clock import SteppedClock, change_difficulty
from brawl.game.keys import FIGHT_KEYS, LEFT, RIGHT, DirectInput, Keys
from brawl.paths import LEGACY_RUNS, RESULTS

GATES_DIR = RESULTS / "gates"
NO_KEYS: frozenset[str] = frozenset()


def small(image) -> np.ndarray:
    """Grey, downscaled copy for comparing frames (cheap and insensitive to subpixel noise)."""
    grey = np.asarray(image, dtype=np.float32).mean(axis=2)
    return grey[::8, ::8]


def diff(a, b) -> float:
    return float(np.abs(small(a) - small(b)).mean())


class Session:
    def __init__(self, args):
        self.args = args
        self.keys = Keys(DirectInput(), tap_s=args.tap_ms / 1000)
        self.clock = SteppedClock(self.keys, rearm_each_frame=args.rearm)
        self.screen = capture.Screen(timeout_s=0.3)

    def step(self, frame_keys=NO_KEYS):
        self.clock.advance([frame_keys])
        return self.screen.grab()

    def play(self, plan):
        return [self.step(k).image for k in plan]


def t1(s: Session) -> tuple[bool, list[str]]:
    lines, ok = [], True
    s.screen.grab(fresh=False)
    s.clock.start()
    time.sleep(0.3)
    s.screen.grab(fresh=False)
    paused = not s.screen.grab().fresh
    lines.append(f"- F7 pause: {'PASS' if paused else 'FAIL'} (no new frame for 0.3 s after F7)")
    ok &= paused
    stepped = s.step().fresh
    lines.append(f"- F6 step: {'PASS' if stepped else 'FAIL'} (a new frame after F6)")
    ok &= stepped
    before = s.screen.grab(fresh=False).image
    s.step()
    s.clock.rewind(1)
    back = s.screen.grab()
    rewound = back.fresh and diff(back.image, before) < 1.0
    lines.append(f"- F5 rewind: {'PASS' if rewound else 'FAIL'} (back to the previous frame, diff {diff(back.image, before):.2f})")
    ok &= rewound
    lines.append("")
    lines.append("| Key | Mean pixel difference after 12 frames vs no key | Reached the game |")
    lines.append("|---|---|---|")
    n = 12
    baseline = s.play([NO_KEYS] * n)[-1]
    s.clock.rewind(n)
    for key in sorted(FIGHT_KEYS):
        image = s.play([frozenset({key})] * n)[-1]
        s.clock.rewind(n)
        d = diff(image, baseline)
        reached = d > 0.5
        ok &= reached
        lines.append(f"| `{key}` | {d:.2f} | {'yes' if reached else 'NO'} |")
    return ok, lines


def t2(s: Session) -> tuple[bool, list[str]]:
    s.clock.start()
    n = s.args.frames
    one_frame = 0
    for _ in range(n):
        got = s.step()
        extra = s.screen.grab()  # a second new frame here would mean the step produced more than one
        one_frame += int(got.fresh and not extra.fresh)
    exact = one_frame == n
    lines = [f"- One F6 = exactly one new frame: {one_frame}/{n} {'PASS' if exact else 'FAIL'}"]
    # Held keys vs keys re-pressed with every F6: if they differ, only re-arming works.
    m = 30
    held = SteppedClock(s.keys, rearm_each_frame=False, frame=s.clock.frame)
    held.paused = True
    rearm = SteppedClock(s.keys, rearm_each_frame=True, frame=s.clock.frame)
    rearm.paused = True
    held.advance([frozenset({RIGHT})] * m)
    a = s.screen.grab().image
    held.rewind(m)
    rearm.advance([frozenset({RIGHT})] * m)
    b = s.screen.grab().image
    rearm.rewind(m)
    d = diff(a, b)
    lines.append(f"- Holding RIGHT for {m} frames: held vs re-armed differ by {d:.2f} "
                 f"({'held keys apply every frame: rearm not needed' if d < 1.0 else 'held keys do NOT carry over: use --rearm'})")
    return exact, lines


def t3(s: Session) -> tuple[bool, list[str]]:
    s.clock.start()
    rng = random.Random(0)
    n, reps = s.args.frames, 5
    keys = sorted(FIGHT_KEYS)
    worst, mismatches = 0.0, 0
    for _ in range(reps):
        plan = [frozenset(rng.sample(keys, rng.randint(0, 2))) for _ in range(n)]
        first = s.play(plan)
        s.clock.rewind(n)
        second = s.play(plan)
        s.clock.rewind(n)
        for a, b in zip(first, second):
            d = diff(a, b)
            worst = max(worst, d)
            mismatches += int(d > 0.5)
    ok = mismatches == 0
    return ok, [f"- {reps} replays of {n} random frames: {mismatches} mismatching frames, worst diff {worst:.2f} "
                f"{'PASS' if ok else 'FAIL'}"]


def t4(s: Session) -> tuple[bool, list[str]]:
    s.clock.start()
    n = s.args.frames
    t0 = time.perf_counter()
    s.play([frozenset({LEFT if i % 20 < 10 else RIGHT}) for i in range(n)])
    step_s = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    s.clock.rewind(n)
    rewind_s = (time.perf_counter() - t0) / n
    return True, [f"- Step + capture: {step_s * 1000:.1f} ms per frame ({1 / step_s:.1f} frames/s; "
                  f"{1 / (3 * step_s):.1f} decisions/s at k = 3), tap {s.args.tap_ms} ms",
                  f"- Rewind: {rewind_s * 1000:.1f} ms per frame"]


def t5(s: Session) -> tuple[bool, list[str]]:
    import cv2

    out = LEGACY_RUNS.parent / "gates" / "t5"
    out.mkdir(parents=True, exist_ok=True)
    reps = 10
    for i in range(reps):
        for direction, steps in (("up", 1), ("down", -1)):
            change_difficulty(s.clock, steps)
            cv2.imwrite(str(out / f"{i:02d}_{direction}.png"), s.screen.grab(fresh=False).image)
    return True, [f"- {2 * reps} switches done; screenshots in `{out}`. "
                  "Open them and confirm each shows the expected level, then record the count here."]


GATES = {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gate", choices=sorted(GATES))
    p.add_argument("--frames", type=int, default=None, help="frames per test (default per gate)")
    p.add_argument("--rearm", action="store_true", help="re-press keys together with every F6")
    p.add_argument("--tap-ms", type=float, default=20.0)
    p.add_argument("--countdown", type=int, default=5)
    args = p.parse_args(argv)
    args.frames = args.frames or {"t2": 1000, "t3": 120, "t4": 300}.get(args.gate, 0)
    print(capture.status())
    for i in range(args.countdown, 0, -1):
        print(f"click into the game window... {i}", flush=True)
        time.sleep(1)
    s = Session(args)
    try:
        ok, lines = GATES[args.gate](s)
    finally:
        s.keys.release_all()
    GATES_DIR.mkdir(parents=True, exist_ok=True)
    report = GATES_DIR / f"{args.gate}.md"
    header = [f"# Gate {args.gate.upper()}: {'PASS' if ok else 'FAIL'}", "",
              f"Run {datetime.now():%Y-%m-%d %H:%M}; rearm={args.rearm}, tap={args.tap_ms} ms, frames={args.frames}.",
              "The game was left paused (press F7 to resume).", ""]
    report.write_text("\n".join(header + lines) + "\n", encoding="utf-8")
    print("\n".join(header + lines))
    print(f"written: {report}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
