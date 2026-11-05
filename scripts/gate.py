"""Run one Stage 0 gate against the live game and write results/gates/<gate>.md.

Start Brawlhalla in training mode (Wu Shang vs a Mordex bot, Brawlhaven), unpaused, with no menu
open. Run a gate, then click into the game window during the countdown: the script presses keys
in whatever window has focus.

    python scripts/gate.py t1     # pause, and every fight key reaches the game
    python scripts/gate.py t2     # N steps then N rewinds return to the start; held vs re-pressed keys
    python scripts/gate.py t3     # rewind and replay give identical frames
    python scripts/gate.py t4     # steps and rewind frames per second
    python scripts/gate.py t5     # difficulty macro (saves screenshots for you to confirm)

How frames are compared. Parts of the screen keep animating while the game is paused (smoke, the
respawn ring, menus). Each gate first watches the paused screen for one second and keeps only the
pixels that stay still (the *static mask*). Two frames "differ" by the fraction of static pixels
whose brightness changed by more than PIXEL_THRESHOLD. The *noise floor* is that fraction between
two captures of the same paused frame. Evidence images go to runs/gates/<gate>/.

Options: --rearm, --tap-ms (key hold time), --settle-ms (wait after a step before capturing).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # sys and pathlib do not load torch

import brawl.game.capture as capture  # noqa: E402  first: the screen duplicator must exist before torch loads

import argparse  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402

from brawl.game.clock import SteppedClock, change_difficulty  # noqa: E402
from brawl.game.keys import FIGHT_KEYS, RIGHT, DirectInput, Keys  # noqa: E402
from brawl.paths import RESULTS, RUNS  # noqa: E402

GATES_DIR = RESULTS / "gates"
NO_KEYS: frozenset[str] = frozenset()
SCALE = 4               # compare at 480 x 270
PIXEL_THRESHOLD = 24.0  # grey levels (0-255) for a pixel to count as changed
ANIMATED = 12.0         # a pixel whose brightness ranges more than this while paused is masked out


def grey(image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32).mean(axis=2)[::SCALE, ::SCALE]


class Session:
    def __init__(self, args):
        self.args = args
        self.out = RUNS / "gates" / args.gate
        self.out.mkdir(parents=True, exist_ok=True)
        self.keys = Keys(DirectInput(), tap_s=args.tap_ms / 1000)
        self.clock = SteppedClock(self.keys, rearm_each_frame=args.rearm, settle_s=args.settle_ms / 1000)
        self.screen = capture.Screen(timeout_s=0.3)
        self.mask: np.ndarray | None = None
        self.floor = 0.0

    def shot(self) -> np.ndarray:
        return grey(self.screen.grab(fresh=False).image)

    def step(self, frame_keys=NO_KEYS) -> np.ndarray:
        self.clock.advance([frame_keys])
        return self.shot()

    def play(self, plan) -> list[np.ndarray]:
        return [self.step(k) for k in plan]

    def calibrate(self, seconds: float = 1.0, n: int = 20) -> list[str]:
        """Watch the paused screen: build the static mask and the noise floor."""
        shots = []
        for _ in range(n):
            shots.append(self.shot())
            time.sleep(seconds / n)
        stack = np.stack(shots)
        self.mask = (stack.max(0) - stack.min(0)) <= ANIMATED
        self.floor = max(self.differ(a, b) for a, b in zip(shots, shots[1:]))
        self.save_mask(shots[0])
        return [f"- Static mask: {self.mask.mean():.1%} of the screen stays still while paused "
                f"(the rest animates and is ignored); noise floor {self.floor:.4%}"]

    def differ(self, a: np.ndarray, b: np.ndarray) -> float:
        changed = np.abs(a - b) > PIXEL_THRESHOLD
        return float(changed[self.mask].mean()) if self.mask is not None else float(changed.mean())

    def different(self, a, b) -> bool:
        """Clearly different: above three times the noise floor and above 0.02% of static pixels."""
        return self.differ(a, b) > max(3 * self.floor, 2e-4)

    def save_diff(self, name: str, a: np.ndarray, b: np.ndarray) -> None:
        import cv2

        changed = (np.abs(a - b) > PIXEL_THRESHOLD) & self.mask
        view = np.dstack([a, a, a]).astype(np.uint8)
        view[changed] = (0, 0, 255)
        view[~self.mask] = (view[~self.mask] * 0.3).astype(np.uint8)
        cv2.imwrite(str(self.out / f"{name}.png"), view)

    def save_mask(self, frame: np.ndarray) -> None:
        import cv2

        view = np.dstack([frame, frame, frame]).astype(np.uint8)
        view[~self.mask] = (view[~self.mask] * 0.3).astype(np.uint8)
        cv2.imwrite(str(self.out / "static_mask.png"), view)


def t1(s: Session) -> tuple[bool, list[str]]:
    lines, ok = [], True
    s.screen.grab(fresh=False)
    running_a = s.shot()
    time.sleep(1.0)
    running_b = s.shot()
    s.clock.start()
    time.sleep(0.5)
    lines += s.calibrate()
    paused_a = s.shot()
    time.sleep(1.0)
    paused_b = s.shot()
    moving, still = s.differ(running_a, running_b), s.differ(paused_a, paused_b)
    paused = still <= max(3 * s.floor, 2e-4) and moving > still
    lines.append(f"- F7 pause: {'PASS' if paused else 'FAIL'} (static pixels changing over 1 s: "
                 f"{moving:.3%} running, {still:.4%} paused)")
    ok &= paused
    n = 12
    baseline = s.play([NO_KEYS] * n)[-1]
    s.clock.rewind(n)
    again = s.play([NO_KEYS] * n)[-1]
    s.clock.rewind(n)
    replay_noise = s.differ(baseline, again)
    lines.append(f"- Replaying the same {n} frames twice differs by {replay_noise:.4%} (keys must beat this)")
    lines += ["", f"| Key | Static pixels changed after {n} frames vs no key | Reached the game |", "|---|---|---|"]
    for key in sorted(FIGHT_KEYS):
        image = s.play([frozenset({key})] * n)[-1]
        s.clock.rewind(n)
        d = s.differ(image, baseline)
        reached = d > max(3 * replay_noise, 3 * s.floor, 2e-4)
        ok &= reached
        s.save_diff(f"key_{key}", image, baseline)
        lines.append(f"| `{key}` | {d:.3%} | {'yes' if reached else 'NO'} |")
    return ok, lines


def t2(s: Session) -> tuple[bool, list[str]]:
    s.clock.start()
    time.sleep(0.5)
    lines = s.calibrate()
    n = s.args.frames
    start = s.shot()
    s.play([frozenset({RIGHT}) if i % 30 < 15 else NO_KEYS for i in range(n)])
    moved = s.shot()
    s.clock.rewind(n)
    back = s.shot()
    returned = not s.different(start, back)
    s.save_diff("after_steps_vs_start", moved, start)
    s.save_diff("after_rewind_vs_start", back, start)
    lines.append(f"- {n} steps then {n} rewinds return to the start: {'PASS' if returned else 'FAIL'} "
                 f"(difference {s.differ(start, back):.4%}; the steps themselves changed {s.differ(start, moved):.3%})")
    # Held keys vs keys re-pressed with every F6: if they differ, only re-pressing works.
    m = 30
    held = SteppedClock(s.keys, settle_s=s.clock.settle_s, frame=s.clock.frame)
    rearm = SteppedClock(s.keys, rearm_each_frame=True, settle_s=s.clock.settle_s, frame=s.clock.frame)
    held.paused = rearm.paused = True
    held.advance([frozenset({RIGHT})] * m)
    a = s.shot()
    held.rewind(m)
    rearm.advance([frozenset({RIGHT})] * m)
    b = s.shot()
    rearm.rewind(m)
    s.save_diff("held_vs_rearmed", a, b)
    same = not s.different(a, b)
    lines.append(f"- Holding RIGHT for {m} frames, held vs re-pressed differ by {s.differ(a, b):.4%}: "
                 + ("held keys carry over, re-pressing not needed" if same else "they differ: inspect held_vs_rearmed.png"))
    return returned, lines


def t3(s: Session) -> tuple[bool, list[str]]:
    s.clock.start()
    time.sleep(0.5)
    lines = s.calibrate()
    rng = random.Random(0)
    n, reps = s.args.frames, 5
    keys = sorted(FIGHT_KEYS)
    worst, worst_pair, mismatches = 0.0, None, 0
    for _ in range(reps):
        plan = [frozenset(rng.sample(keys, rng.randint(0, 2))) for _ in range(n)]
        first = s.play(plan)
        s.clock.rewind(n)
        second = s.play(plan)
        s.clock.rewind(n)
        for a, b in zip(first, second):
            d = s.differ(a, b)
            mismatches += int(s.different(a, b))
            if d > worst:
                worst, worst_pair = d, (a, b)
    if worst_pair is not None:
        s.save_diff("worst_replay_pair", *worst_pair)
    ok = mismatches == 0
    lines.append(f"- {reps} replays of {n} random frames: {mismatches}/{reps * n} frames differ beyond the noise "
                 f"floor; worst {worst:.4%} {'PASS' if ok else 'FAIL'} (see worst_replay_pair.png)")
    return ok, lines


def t4(s: Session) -> tuple[bool, list[str]]:
    s.clock.start()
    n = s.args.frames
    t0 = time.perf_counter()
    s.play([frozenset({RIGHT}) if i % 20 < 10 else NO_KEYS for i in range(n)])
    step_s = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    s.clock.rewind(n)
    rewind_s = (time.perf_counter() - t0) / n
    return True, [f"- Step + capture: {step_s * 1000:.1f} ms per frame ({1 / step_s:.1f} frames/s; "
                  f"{1 / (3 * step_s):.1f} decisions/s at k = 3, i.e. {20 * 3 * step_s:.2f} s of wall time per game second)",
                  f"- Rewind: {rewind_s * 1000:.1f} ms per frame",
                  f"- Settings: tap {s.args.tap_ms} ms, settle {s.args.settle_ms} ms"]


def t5(s: Session) -> tuple[bool, list[str]]:
    import cv2

    reps = 10
    for i in range(reps):
        for direction, steps in (("up", 1), ("down", -1)):
            change_difficulty(s.clock, steps, settle_s=0.0)
            for delay in (0.5, 1.5):
                time.sleep(delay if delay == 0.5 else 1.0)
                cv2.imwrite(str(s.out / f"{i:02d}_{direction}_{delay:.1f}s.png"), s.screen.grab(fresh=False).image)
    return True, [f"- {2 * reps} switches done; screenshots at 0.5 s and 1.5 s after each in `{s.out}`.",
                  "- Check that each shows the expected level and that the menu has closed by 1.5 s."]


GATES = {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gate", choices=sorted(GATES))
    p.add_argument("--frames", type=int, default=None, help="frames per test (default per gate)")
    p.add_argument("--rearm", action="store_true", help="re-press keys together with every F6")
    p.add_argument("--tap-ms", type=float, default=20.0)
    p.add_argument("--settle-ms", type=float, default=34.0, help="wait after a step before capturing")
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
              f"Run {datetime.now():%Y-%m-%d %H:%M}; rearm={args.rearm}, tap={args.tap_ms} ms, "
              f"settle={args.settle_ms} ms, frames={args.frames}. Evidence: `{s.out}`.",
              "The game was left paused (press F7 to resume).", ""]
    report.write_text("\n".join(header + lines) + "\n", encoding="utf-8")
    print("\n".join(header + lines))
    print(f"written: {report}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
