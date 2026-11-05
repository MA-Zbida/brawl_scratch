"""Run one Stage 0 gate against the live game and write results/gates/<gate>.md.

Start Brawlhalla in training mode (Wu Shang vs a Mordex bot, Brawlhaven) with no menu open, paused
or not. Run a gate, then click into the game window during the countdown: the script presses keys
in whatever window has focus.

    python scripts/gate.py t0     # witness: you read the frame counter, the script compares
    python scripts/gate.py t1     # every fight key reaches the game
    python scripts/gate.py t2     # every F6 advances and every F5 rewinds exactly one frame
    python scripts/gate.py t3     # rewind and replay give identical frames
    python scripts/gate.py t4     # steps and rewind frames per second

Ground truth is the game's own **frame counter**, shown in the training toolbar while paused
(between "F5 Prev" and "F6 Next"). A step or rewind counts only if the counter changed. Pause is a
toggle, so each gate first finds out whether the game is paused (the toolbar box is only there
when it is) instead of assuming.

Fighter comparisons use a *static mask*: parts of the screen animate while paused (smoke, the
respawn ring), so each gate watches the paused screen for a second and compares only the pixels
that stay still. Evidence images go to runs/gates/<gate>/.

The difficulty macro (former t5) needs menu navigation and is not automated yet: switch the bot's
level by hand for now.
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

from brawl.game.clock import SteppedClock  # noqa: E402
from brawl.game.keys import FIGHT_KEYS, PAUSE, RIGHT, DirectInput, Keys  # noqa: E402
from brawl.paths import RESULTS, RUNS  # noqa: E402

GATES_DIR = RESULTS / "gates"
NO_KEYS: frozenset[str] = frozenset()
COUNTER_BOX = (700, 40, 760, 62)   # x0, y0, x1, y1 of the frame counter at 1920 x 1080
TOOLBAR_BOX = (600, 40, 900, 62)   # dark toolbar box (paused) or sky (running)
SCALE = 4                          # fighter comparisons at 480 x 270
PIXEL_THRESHOLD = 24.0             # grey levels for a pixel to count as changed
ANIMATED = 12.0                    # brightness range while paused above which a pixel is masked out


def grey(image, box=None, scale=1) -> np.ndarray:
    a = np.asarray(image)
    if box is not None:
        x0, y0, x1, y1 = box
        a = a[y0:y1, x0:x1]
    return a.astype(np.float32).mean(axis=2)[::scale, ::scale]


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

    def image(self):
        return self.screen.grab(fresh=False).image

    # -- pause and the frame counter -------------------------------------------------------
    def toolbar_brightness(self) -> float:
        return float(grey(self.image(), TOOLBAR_BOX).mean())

    def ensure_paused(self) -> list[str]:
        """Pause is a toggle: press it, and press again if that made things worse."""
        before = self.toolbar_brightness()
        self.keys.tap(PAUSE)
        time.sleep(0.4)
        after = self.toolbar_brightness()
        if after > before:  # the toolbar box disappeared: it was paused and we just resumed
            self.keys.tap(PAUSE)
            time.sleep(0.4)
            after = self.toolbar_brightness()
        self.clock.paused = True
        return [f"- Pause found by the toolbar box (brightness {max(before, after):.0f} running/other, "
                f"{min(before, after):.0f} paused)"]

    def counter(self) -> np.ndarray:
        return grey(self.image(), COUNTER_BOX)

    @staticmethod
    def counter_changed(a: np.ndarray, b: np.ndarray) -> bool:
        """The counter is static UI text: any digit change moves dozens of pixels by ~50 grey
        levels, while an unchanged counter moves none. Six pixels is a safe margin."""
        return int((np.abs(a - b) > 20).sum()) >= 6

    def step(self, frame_keys=NO_KEYS) -> tuple[np.ndarray, bool]:
        """One frame step; returns the frame and whether the counter moved."""
        before = self.counter()
        self.clock.advance([frame_keys])
        image = self.image()
        return image, self.counter_changed(before, grey(image, COUNTER_BOX))

    def rewind_one(self) -> bool:
        before = self.counter()
        self.clock.rewind(1)
        return self.counter_changed(before, self.counter())

    def play(self, plan) -> tuple[list[np.ndarray], int]:
        frames, missed = [], 0
        for frame_keys in plan:
            image, moved = self.step(frame_keys)
            frames.append(grey(image, scale=SCALE))
            missed += int(not moved)
        return frames, missed

    def rewind(self, n: int) -> int:
        return sum(int(not self.rewind_one()) for _ in range(n))

    # -- fighter comparisons ----------------------------------------------------------------
    def calibrate(self, seconds: float = 1.0, n: int = 20) -> list[str]:
        shots = []
        for _ in range(n):
            shots.append(grey(self.image(), scale=SCALE))
            time.sleep(seconds / n)
        stack = np.stack(shots)
        self.mask = (stack.max(0) - stack.min(0)) <= ANIMATED
        self.floor = max(self.differ(a, b) for a, b in zip(shots, shots[1:]))
        self.save(shots[0], None, "static_mask")
        return [f"- Static mask: {self.mask.mean():.1%} of the screen stays still while paused; "
                f"noise floor {self.floor:.4%}"]

    def differ(self, a: np.ndarray, b: np.ndarray) -> float:
        changed = np.abs(a - b) > PIXEL_THRESHOLD
        return float(changed[self.mask].mean()) if self.mask is not None else float(changed.mean())

    def different(self, a, b, noise: float = 0.0) -> bool:
        return self.differ(a, b) > max(3 * self.floor, 3 * noise, 2e-4)

    def save(self, a: np.ndarray, b: np.ndarray | None, name: str) -> None:
        import cv2

        view = np.dstack([a, a, a]).astype(np.uint8)
        if b is not None:
            view[(np.abs(a - b) > PIXEL_THRESHOLD) & self.mask] = (0, 0, 255)
        view[~self.mask] = (view[~self.mask] * 0.3).astype(np.uint8)
        cv2.imwrite(str(self.out / f"{name}.png"), view)


def t0(s: Session) -> tuple[bool, list[str]]:
    """Witness test: you read the frame counter; the script compares it with what it detected.

    Two phases (no keys held; random keys held). In each: 200 F6 presses, then 200 F5 presses.
    Before, between and after, you read the counter in the toolbar and type it in this terminal.
    """
    lines = s.ensure_paused()
    rng = random.Random(1)
    keys = sorted(FIGHT_KEYS)
    n = s.args.frames or 200
    ok = True

    def read(prompt: str) -> int:
        while True:
            text = input(f"\n{prompt}\n  Click this terminal, type the frame counter shown in the game's toolbar, press Enter: ")
            try:
                return int(text.strip())
            except ValueError:
                print("  numbers only, please")

    def back_to_game() -> None:
        for i in range(3, 0, -1):
            print(f"  click into the game window... {i}", flush=True)
            time.sleep(1)

    lines += ["", "| Phase | Presses | Counter moved (you read) | Counter moved (script detected) | Misses (by you) | Keys held at detected misses |",
              "|---|---|---|---|---|---|"]
    for phase in ("no keys", "random keys"):
        plan = [NO_KEYS if phase == "no keys" else frozenset(rng.sample(keys, rng.randint(1, 2))) for _ in range(n)]
        v0 = read(f"[{phase}] before {n} steps")
        back_to_game()
        missed_keys: dict[str, int] = {}
        missed_f6 = 0
        for frame_keys in plan:
            if not s.step(frame_keys)[1]:
                missed_f6 += 1
                for key in frame_keys:
                    missed_keys[key] = missed_keys.get(key, 0) + 1
        v1 = read(f"[{phase}] after {n} steps")
        back_to_game()
        missed_f5 = s.rewind(n)
        v2 = read(f"[{phase}] after {n} rewinds")
        back_to_game()
        human_f6, human_f5 = v1 - v0, v1 - v2
        held = ", ".join(f"`{k}` {c}" for k, c in sorted(missed_keys.items(), key=lambda kv: -kv[1])) or "-"
        lines.append(f"| {phase}: F6 | {n} | {human_f6} | {n - missed_f6} | {n - human_f6} | {held} |")
        lines.append(f"| {phase}: F5 | {n} | {human_f5} | {n - missed_f5} | {n - human_f5} | - |")
        ok &= human_f6 == n and human_f5 == n
    lines.append("\nIf 'you read' equals the presses, the game applied every press and any detected misses are "
                 "the script's detector. If 'you read' is lower, the game really skipped presses.")
    return ok, lines


def t1(s: Session) -> tuple[bool, list[str]]:
    lines = s.ensure_paused() + s.calibrate()
    n = 12
    baseline, m1 = s.play([NO_KEYS] * n)
    r1 = s.rewind(n)
    again, m2 = s.play([NO_KEYS] * n)
    r2 = s.rewind(n)
    noise = s.differ(baseline[-1], again[-1])
    lines.append(f"- Replaying the same {n} frames twice differs by {noise:.4%}; missed steps {m1 + m2}, "
                 f"missed rewinds {r1 + r2}")
    lines += ["", f"| Key | Static pixels changed after {n} frames vs no key | Missed steps | Reached the game |",
              "|---|---|---|---|"]
    ok = m1 + m2 + r1 + r2 == 0
    for key in sorted(FIGHT_KEYS):
        frames, missed = s.play([frozenset({key})] * n)
        missed += s.rewind(n)
        d = s.differ(frames[-1], baseline[-1])
        reached = d > max(3 * noise, 3 * s.floor, 2e-4)
        ok &= reached and missed == 0
        s.save(frames[-1], baseline[-1], f"key_{key}")
        lines.append(f"| `{key}` | {d:.3%} | {missed} | {'yes' if reached else 'NO'} |")
    return ok, lines


def t2(s: Session) -> tuple[bool, list[str]]:
    lines = s.ensure_paused()
    n = s.args.frames
    taps = [float(t) for t in s.args.sweep_tap.split(",")] if s.args.sweep_tap else [s.args.tap_ms]
    lines += ["", f"| Key hold (ms) | Missed F6 of {n} | Missed F5 of {n} | Back at the start frame | ms per step |",
              "|---|---|---|---|---|"]
    best_ok = False
    for tap_ms in taps:
        s.keys.tap_s = tap_ms / 1000
        start = s.counter()
        t0 = time.perf_counter()
        missed_f6 = sum(int(not s.step(frozenset({RIGHT}) if i % 30 < 15 else NO_KEYS)[1]) for i in range(n))
        per_step = (time.perf_counter() - t0) / n
        missed_f5 = s.rewind(n)
        back = not s.counter_changed(start, s.counter())
        ok = missed_f6 == 0 and missed_f5 == 0 and back
        best_ok |= ok
        lines.append(f"| {tap_ms:g} | {missed_f6} | {missed_f5} | {'yes' if back else 'NO'} | {per_step * 1000:.0f} |")
    if best_ok:
        lines.append("\nUse the shortest key hold with no misses (`--tap-ms`).")
    # Held keys vs keys re-pressed with every F6, compared on exact, counter-checked frames.
    s.keys.tap_s = s.args.tap_ms / 1000
    lines += [""] + s.calibrate()
    m = 30
    s.clock.rearm_each_frame = False
    held, mh = s.play([frozenset({RIGHT})] * m)
    s.rewind(m)
    s.clock.rearm_each_frame = True
    rearmed, mr = s.play([frozenset({RIGHT})] * m)
    s.rewind(m)
    s.clock.rearm_each_frame = s.args.rearm
    s.save(held[-1], rearmed[-1], "held_vs_rearmed")
    same = not s.different(held[-1], rearmed[-1])
    lines.append(f"- Holding RIGHT for {m} frames, held vs re-pressed differ by {s.differ(held[-1], rearmed[-1]):.4%} "
                 f"(missed steps {mh + mr}): " + ("held keys carry over" if same else "they differ: see held_vs_rearmed.png"))
    return best_ok, lines


def t3(s: Session) -> tuple[bool, list[str]]:
    lines = s.ensure_paused() + s.calibrate()
    rng = random.Random(0)
    n, reps = s.args.frames, 5
    keys = sorted(FIGHT_KEYS)
    worst, worst_pair, mismatches, missed = 0.0, None, 0, 0
    for _ in range(reps):
        plan = [frozenset(rng.sample(keys, rng.randint(0, 2))) for _ in range(n)]
        first, m1 = s.play(plan)
        missed += m1 + s.rewind(n)
        second, m2 = s.play(plan)
        missed += m2 + s.rewind(n)
        for a, b in zip(first, second):
            d = s.differ(a, b)
            mismatches += int(s.different(a, b))
            if d > worst:
                worst, worst_pair = d, (a, b)
    if worst_pair is not None:
        s.save(*worst_pair, "worst_replay_pair")
    ok = mismatches == 0 and missed == 0
    lines.append(f"- {reps} replays of {n} random frames: {mismatches}/{reps * n} frames differ beyond the noise "
                 f"floor (worst {worst:.4%}); missed steps or rewinds: {missed} {'PASS' if ok else 'FAIL'}")
    return ok, lines


def t4(s: Session) -> tuple[bool, list[str]]:
    lines = s.ensure_paused()
    n = s.args.frames
    t0 = time.perf_counter()
    for i in range(n):  # what the environment will do: step, then grab one frame
        s.clock.advance([frozenset({RIGHT}) if i % 20 < 10 else NO_KEYS])
        s.screen.grab(fresh=False)
    step_s = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    s.clock.rewind(n)
    rewind_s = (time.perf_counter() - t0) / n
    return True, lines + [
        f"- Step + capture: {step_s * 1000:.1f} ms per frame ({1 / step_s:.1f} frames/s; "
        f"{1 / (3 * step_s):.1f} decisions/s at k = 3)",
        f"- Rewind: {rewind_s * 1000:.1f} ms per frame",
        f"- Settings: tap {s.args.tap_ms} ms, settle {s.args.settle_ms} ms"]


GATES = {"t0": t0, "t1": t1, "t2": t2, "t3": t3, "t4": t4}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gate", choices=sorted(GATES))
    p.add_argument("--frames", type=int, default=None, help="frames per test (default per gate)")
    p.add_argument("--rearm", action="store_true", help="re-press keys together with every F6")
    p.add_argument("--tap-ms", type=float, default=20.0)
    p.add_argument("--settle-ms", type=float, default=34.0, help="wait after a step before capturing")
    p.add_argument("--sweep-tap", default="10,20,35,50", help="t2: key hold times to compare (ms)")
    p.add_argument("--countdown", type=int, default=5)
    args = p.parse_args(argv)
    args.frames = args.frames or {"t2": 300, "t3": 120, "t4": 300}.get(args.gate, 0)
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
              "The game is left paused.", ""]
    report.write_text("\n".join(header + lines) + "\n", encoding="utf-8")
    print("\n".join(header + lines))
    print(f"written: {report}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
