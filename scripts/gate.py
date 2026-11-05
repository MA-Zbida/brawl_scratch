"""Run one Stage 0 gate against the live game and write results/gates/<gate>.md.

Start Brawlhalla in training mode (Wu Shang vs a Mordex bot, Brawlhaven) with no menu open, paused
or not. Run a gate, then click into the game window during the countdown: the script presses keys
in whatever window has focus.

    python scripts/gate.py counter  # once, first: teach the script the counter's digits
    python scripts/gate.py t0     # witness: you read the frame counter, the script compares
    python scripts/gate.py t1     # every fight key reaches the game
    python scripts/gate.py t2     # every F6 advances and every F5 rewinds exactly one frame
    python scripts/gate.py t3     # rewind and replay give identical frames
    python scripts/gate.py t4     # steps and rewind frames per second

Ground truth is the game's own **frame counter**, shown in the training toolbar while paused
(between "F5 Prev" and "F6 Next"). It restarts at 0 on every pause. With confirmation (the
default) the clock reads its number before and after every press. Pause is a
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

from brawl.game.clock import StepError, SteppedClock  # noqa: E402
from brawl.game.counter import GLYPHS_PATH, Digits, FrameCounter, counter_changed, held_out_accuracy  # noqa: E402
from brawl.game.keys import FIGHT_KEYS, PAUSE, REWIND, RIGHT, STEP, DirectInput, Keys  # noqa: E402
from brawl.paths import RESULTS, RUNS  # noqa: E402

GATES_DIR = RESULTS / "gates"
NO_KEYS: frozenset[str] = frozenset()
TOOLBAR_BOX = (600, 40, 900, 62)   # dark toolbar box (paused) or sky (running)
SCALE = 4                          # fighter comparisons at 480 x 270
PIXEL_THRESHOLD = 24.0             # grey levels for a pixel to count as changed
ANIMATED = 12.0                    # brightness range while paused above which a pixel is masked out


def foreground_window() -> str:
    """Title of the window that receives the keys (Windows)."""
    try:
        import ctypes

        user32 = ctypes.windll.user32
        handle = user32.GetForegroundWindow()
        buffer = ctypes.create_unicode_buffer(user32.GetWindowTextLengthW(handle) + 1)
        user32.GetWindowTextW(handle, buffer, len(buffer))
        return buffer.value or "(untitled)"
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        return f"(unknown: {exc})"


def probe_stall(s: "Session", key: str, keys_held_s: float = 20.0, released_s: float = 10.0) -> list[str]:
    """After the game stopped responding: keep pressing ``key`` to learn whether the stall ends
    by itself, and whether releasing the fight keys ends it."""
    lines = []
    for phase, seconds in (("keys still held", keys_held_s), ("all keys released", released_s)):
        if phase == "all keys released":
            s.keys.release_all()
        before, t0 = s.frame_counter.value(), time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            s.keys.tap(key, hold_s=0.03)
            moved = s.frame_counter.wait_value(before, timeout_s=1.0)
            if moved is not None:
                return lines + [f"- Probe: the game answered {key} again after a further "
                                f"{time.perf_counter() - t0:.1f} s ({phase}); counter {before} -> {moved}"]
        lines.append(f"- Probe: no answer to {key} for {seconds:.0f} s more ({phase}); window with focus: "
                     f"\"{foreground_window()}\"")
    return lines


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
        self.screen = capture.Screen(timeout_s=0.05)
        self.frame_counter = FrameCounter(self.screen)
        # --confirm: the clock checks every press against the counter and retries dropped ones.
        self.clock = SteppedClock(self.keys, counter=self.frame_counter if args.confirm else None,
                                  key_gap_s=args.key_gap_ms / 1000, rearm_each_frame=args.rearm,
                                  settle_s=0.0 if args.confirm else args.settle_ms / 1000)
        self.retries = 0
        self.repairs = 0
        self.events: list[tuple[int, int]] = []  # (retries, repairs) of every step, in order
        self.lines: list[str] = []  # report lines so far: kept if the gate stops early
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
        lines = [f"- Pause found by the toolbar box (brightness {max(before, after):.0f} running/other, "
                 f"{min(before, after):.0f} paused)"]
        if self.clock.counter is not None:
            self.clock.sync()
            lines.append(f"- The frame counter reads {self.clock.frame}")
        return lines

    def counter(self) -> np.ndarray:
        return self.frame_counter.read()

    counter_changed = staticmethod(counter_changed)

    def step(self, frame_keys=NO_KEYS) -> tuple[np.ndarray, bool]:
        """One frame step; returns the frame and whether the counter moved (waiting for it)."""
        before = self.counter()
        timing = self.clock.advance([frame_keys])
        self.retries += timing.retries
        self.repairs += timing.repairs
        self.events.append((timing.retries, timing.repairs))
        moved = True if self.args.confirm else self.frame_counter.wait_change(before)
        return self.image(), moved

    def rewind_one(self) -> bool:
        before = self.counter()
        timing = self.clock.rewind(1)
        self.retries += timing.retries
        self.repairs += timing.repairs
        return True if self.args.confirm else self.frame_counter.wait_change(before)

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


def read_counter_by_eye(prompt: str) -> int:
    while True:
        text = input(f"\n{prompt}\n  Click this terminal, type the frame counter shown in the game's toolbar, press Enter: ")
        try:
            return int(text.strip())
        except ValueError:
            print("  numbers only, please")


def counter(s: Session) -> tuple[bool, list[str]]:
    """Teach the script the counter's digits (run once; writes assets/counter/glyphs.npz).

    Pausing restarts the counter at 0. The script re-pauses, steps forward ``up`` frames and then
    rewinds below 0, slowly, waiting for every change, so each crop is labelled with the number it
    shows; you confirm the final number by eye. ``--refit`` fits again from the saved crops.
    """
    s.clock.counter = None
    lines = s.ensure_paused()
    up, below = s.args.frames or 110, 15
    s.keys.tap(PAUSE)  # resume and pause again: the counter restarts at 0
    time.sleep(0.4)
    s.keys.tap(PAUSE)
    time.sleep(0.6)
    samples, value, retries = [(s.counter(), 0)], 0, 0

    def press(key: str, delta: int) -> None:
        nonlocal value, retries
        before = s.counter()
        for attempt in range(4):
            s.keys.tap(key, hold_s=0.03)
            if s.frame_counter.wait_change(before, timeout_s=1.0):
                break
            retries += 1
        else:
            raise StepError(f"the game ignored {key} 4 times while calibrating (counter {value})")
        time.sleep(0.15)
        after = s.counter()
        time.sleep(0.15)
        if s.counter_changed(after, s.counter()):
            raise StepError(f"a press landed twice while calibrating (counter {value}): run again")
        value += delta
        samples.append((after, value))

    for _ in range(up):
        press(STEP, +1)
    for _ in range(up + below):
        press(REWIND, -1)
    seen = read_counter_by_eye(f"The script expects the counter to read {value}.")
    if seen != value:
        return False, lines + [f"- **The counter reads {seen}, the script counted {value}**: labels are off "
                               f"(did the counter start at 0?). Nothing saved."]
    np.savez_compressed(s.out / "samples.npz", crops=np.stack([c for c, _ in samples]),
                        labels=np.array([v for _, v in samples]))
    ok, fitted = fit_counter(samples)
    return ok, lines + [f"- {len(samples)} labelled crops ({up} steps from 0, then down to {value}); "
                        f"you confirmed {seen}; {retries} presses retried"] + fitted


COUNTER_SAMPLES = RUNS / "gates" / "counter" / "samples.npz"


def fit_counter(samples: list[tuple[np.ndarray, int]]) -> tuple[bool, list[str]]:
    """Fit the reader on labelled crops, check it, and save it if every check passes."""
    held, n_held = held_out_accuracy(samples)
    digits, skipped = Digits.fit(samples)
    correct = sum(digits.read(c) == v for c, v in samples)
    ok = skipped == 0 and held == n_held and correct == len(samples)
    if ok:
        digits.save(GLYPHS_PATH)
    chars = "".join(sorted(set(digits.chars.tolist())))
    return ok, [
        f"- Characters seen: `{chars}`; {len(digits.exemplars)} distinct renderings; crops skipped "
        f"(glyph count off): {skipped}",
        f"- Numbers never seen in fitting (every third, fitted on the rest): {held}/{n_held} read correctly",
        f"- All crops read back: {correct}/{len(samples)} " + (f"PASS: saved `{GLYPHS_PATH}`" if ok else "FAIL: not saved")]


def refit_counter() -> tuple[bool, list[str]]:
    """``counter --refit``: fit again from the crops saved by the last calibration (no game)."""
    z = np.load(COUNTER_SAMPLES)
    samples = list(zip(z["crops"], z["labels"].tolist()))
    ok, lines = fit_counter(samples)
    return ok, [f"- Refitted from `{COUNTER_SAMPLES}` ({len(samples)} labelled crops)"] + lines


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

    read = read_counter_by_eye

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
    lines.append(f"\nWith --confirm the clock re-pressed {s.retries} dropped presses (key gap {s.args.key_gap_ms} ms).")
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
    s.save(baseline[-1], again[-1], "replay_noise")
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
    confirm, s.args.confirm, s.clock.counter = s.args.confirm, False, None  # measure raw drops
    gaps = [float(g) for g in s.args.sweep_gap.split(",")]
    rng = random.Random(2)
    keys = sorted(FIGHT_KEYS)
    lines += ["", "Keys change on every step (random fight keys), the case where the game drops F6.", "",
              f"| Key gap (ms) | Dropped F6 of {n} | Dropped F5 of {n} | Back at the start frame | ms per step |",
              "|---|---|---|---|---|"]
    best_ok = False
    for gap_ms in gaps:
        s.clock.key_gap_s = gap_ms / 1000
        plan = [frozenset(rng.sample(keys, rng.randint(1, 2))) for _ in range(n)]
        start = s.counter()
        t0 = time.perf_counter()
        missed_f6 = sum(int(not s.step(k)[1]) for k in plan)
        per_step = (time.perf_counter() - t0) / n
        missed_f5 = s.rewind(n)
        back = not s.counter_changed(start, s.counter())
        ok = missed_f6 == 0 and missed_f5 == 0 and back
        best_ok |= ok
        lines.append(f"| {gap_ms:g} | {missed_f6} | {missed_f5} | {'yes' if back else 'NO'} | {per_step * 1000:.0f} |")
    lines.append("\nWithout --confirm a dropped press is lost; the shortest gap with no drops is the one to use.")
    s.clock.key_gap_s = s.args.key_gap_ms / 1000
    s.args.confirm = confirm
    s.clock.counter = s.frame_counter if confirm else None
    # Held keys vs keys re-pressed with every F6, compared on exact, counter-checked frames.
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
    """Replay determinism: play random keys, rewind, play them again, compare frame by frame.

    Cosmetic animations (phantoms, weapon spawns, the KO explosion and stock icons) run on their
    own clock and are not rewound, so replays differ from the first frame, most right after a KO,
    and the difference fades as those animations end. A fighter that diverges does the opposite:
    the replays stay apart, so the difference persists to the end. The gate therefore compares
    the last 20 frames with the cosmetic level measured by T1 (0.18%), shows the difference per
    20-frame block, and saves each replay's first differing pair for a human to judge. Pixels
    cannot tell a fighter from an effect: this is provisional until perception gives positions.
    """
    lines = s.lines
    lines += s.ensure_paused() + s.calibrate()
    rng = random.Random(0)
    n, reps, tail, cosmetic = s.args.frames, 5, 20, 0.0018
    keys = sorted(FIGHT_KEYS)
    csv = ["rep,frame,keys,diff,retries_1,repairs_1,retries_2,repairs_2"]
    lines += ["", "| Replay | Frames differing | Mean difference per 20-frame block | Last 20 frames | "
              "First differs at | Retry/repair in the 3 frames before it (pass 1 / pass 2) |",
              "|---|---|---|---|---|---|"]
    ok = True
    for rep in range(reps):
        plan = [frozenset(rng.sample(keys, rng.randint(0, 2))) for _ in range(n)]
        s.events = []
        first, m1 = s.play(plan)
        e1 = s.events
        m1 += s.rewind(n)
        s.events = []
        second, m2 = s.play(plan)
        e2 = s.events
        m2 += s.rewind(n)
        diffs = [s.differ(a, b) for a, b in zip(first, second)]
        bad = [s.different(a, b) for a, b in zip(first, second)]
        blocks = " ".join(f"{np.mean(diffs[i:i + 20]):.3%}" for i in range(0, n, 20))
        last = float(np.mean(diffs[-tail:]))
        start = bad.index(True) if any(bad) else None
        near = "-"
        if start is not None:
            window = range(max(0, start - 3), start + 1)
            near = (f"{sum(sum(e1[i]) for i in window)} / {sum(sum(e2[i]) for i in window)}")
            s.save(first[start], second[start], f"replay{rep}_first_difference_frame{start}")
        for i, d in enumerate(diffs):
            csv.append(f"{rep},{i},{'+'.join(sorted(plan[i]))},{d:.6f},{e1[i][0]},{e1[i][1]},{e2[i][0]},{e2[i][1]}")
        ok &= last < cosmetic and m1 + m2 == 0
        lines.append(f"| {rep} | {sum(bad)}/{n} | {blocks} | {last:.3%} | {start if start is not None else '-'} | {near} |")
        (s.out / "frames.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
    lines += ["", f"- Pass: in every replay the last {tail} frames differ by less than the cosmetic level "
              f"({cosmetic:.2%}), and no press was missed: {'PASS' if ok else 'FAIL'}",
              f"- Presses retried: {s.retries}; presses that undid a double landing: {s.repairs}",
              f"- Per frame: `{s.out / 'frames.csv'}`; first differing pairs: `replay*_first_difference_*.png`"]
    return ok, lines


def t4(s: Session) -> tuple[bool, list[str]]:
    lines = s.ensure_paused()
    n = s.args.frames
    retries = repairs = 0
    t0 = time.perf_counter()
    for i in range(n):  # what the environment will do: step, then grab one frame
        timing = s.clock.advance([frozenset({RIGHT}) if i % 20 < 10 else NO_KEYS])
        retries, repairs = retries + timing.retries, repairs + timing.repairs
        s.screen.grab(fresh=False)
    step_s = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    timing = s.clock.rewind(n)
    rewind_s = (time.perf_counter() - t0) / n
    retries, repairs = retries + timing.retries, repairs + timing.repairs
    return True, lines + [
        f"- Step + capture: {step_s * 1000:.1f} ms per frame ({1 / step_s:.1f} frames/s; "
        f"{1 / (3 * step_s):.1f} decisions/s at k = 3)",
        f"- Rewind: {rewind_s * 1000:.1f} ms per frame",
        f"- Presses retried: {retries}; presses that undid a double landing: {repairs}",
        f"- Settings: tap {s.args.tap_ms} ms, settle {s.args.settle_ms} ms"]


GATES = {"counter": counter, "t0": t0, "t1": t1, "t2": t2, "t3": t3, "t4": t4}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gate", choices=sorted(GATES))
    p.add_argument("--frames", type=int, default=None, help="frames per test (default per gate)")
    p.add_argument("--rearm", action="store_true", help="re-press keys together with every F6")
    p.add_argument("--tap-ms", type=float, default=10.0)
    p.add_argument("--settle-ms", type=float, default=34.0, help="wait after a step before capturing")
    p.add_argument("--key-gap-ms", type=float, default=10.0, help="wait between a key change and the F6 that follows")
    p.add_argument("--no-confirm", dest="confirm", action="store_false",
                   help="do not confirm presses on the frame counter (t2 always measures without)")
    p.add_argument("--sweep-gap", default="0,5,10,20,35", help="t2: key gaps to compare (ms)")
    p.add_argument("--countdown", type=int, default=5)
    p.add_argument("--refit", action="store_true", help="counter: fit again from the saved crops, without the game")
    args = p.parse_args(argv)
    args.frames = args.frames or {"t2": 300, "t3": 120, "t4": 300}.get(args.gate, 0)
    if args.gate == "counter" and args.refit:
        ok, lines = refit_counter()
        return write_report(args, ok, lines, COUNTER_SAMPLES.parent, "Fitted offline; the game was not touched.")
    print(capture.status())
    for i in range(args.countdown, 0, -1):
        print(f"click into the game window... {i}", flush=True)
        time.sleep(1)
    s = Session(args)
    s.started, s.focus = time.perf_counter(), foreground_window()
    try:
        ok, lines = GATES[args.gate](s)
    except StepError as exc:
        import cv2

        stalled_at = time.perf_counter() - s.started
        focus = foreground_window()
        image = s.image()
        cv2.imwrite(str(s.out / "step_error_screen.png"), image)
        cv2.imwrite(str(s.out / "step_error_counter.png"), np.asarray(image)[30:72, 680:780])
        ok, lines = False, s.lines + ["", f"- **Stopped {stalled_at:.0f} s into the gate: {exc}**",
                                      f"- Window with focus: \"{focus}\" (at the start: \"{s.focus}\")",
                                      f"- The screen at that moment: `{s.out / 'step_error_screen.png'}` "
                                      f"(counter crop: `step_error_counter.png`)"]
        if exc.key is not None:
            lines += probe_stall(s, exc.key)
    finally:
        s.keys.release_all()
    return write_report(args, ok, lines, s.out, "The game is left paused.")


def write_report(args, ok: bool, lines: list[str], out: Path, note: str) -> int:
    GATES_DIR.mkdir(parents=True, exist_ok=True)
    report = GATES_DIR / f"{args.gate}.md"
    header = [f"# Gate {args.gate.upper()}: {'PASS' if ok else 'FAIL'}", "",
              f"Run {datetime.now():%Y-%m-%d %H:%M}; confirm={args.confirm}, key gap={args.key_gap_ms} ms, rearm={args.rearm}, tap={args.tap_ms} ms, "
              f"settle={args.settle_ms} ms, frames={args.frames}. Evidence: `{out}`.",
              note, ""]
    report.write_text("\n".join(header + lines) + "\n", encoding="utf-8")
    print("\n".join(header + lines))
    print(f"written: {report}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
