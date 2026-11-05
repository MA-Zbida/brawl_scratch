"""Game time: one interface, two regimes (research/DESIGN.md section 3).

A *plan* is the list of key sets to hold, one per game frame, for one decision. ``advance(plan)``
plays it: in ``stepped`` mode by frame-stepping the paused game once per entry, in ``realtime``
mode by holding each set for one frame period (1/60 s) of wall-clock time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence

from brawl.game.keys import DIFFICULTY_DOWN, DIFFICULTY_UP, MENU, PAUSE, REWIND, STEP, Keys

FPS = 60
Plan = Sequence[frozenset[str]]


@dataclass
class Timing:
    """What one ``advance`` cost. Recorded with every step."""

    frames: int = 0
    wall_s: float = 0.0
    late_frames: int = 0  # realtime only: frames whose keys were set after the frame began
    retries: int = 0      # stepped only: presses the game dropped and that were pressed again


class StepError(RuntimeError):
    """The game did not respond to a frame step or rewind, even after retries."""


class Clock(Protocol):
    frame: int

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def advance(self, plan: Plan) -> Timing: ...

    def rewind(self, frames: int) -> Timing: ...


@dataclass
class SteppedClock:
    """The game stays paused; it advances one frame per F6 press, with the plan's keys held.

    ``counter``: the toolbar's frame counter (``brawl.game.counter.FrameCounter``). With it, every
    F6 and F5 is confirmed: the clock waits for the counter to change and presses again if the
    game dropped the press. The game drops F6 when fight keys change at the same moment (gate
    T0), so ``key_gap_s`` separates a key change from the F6 that follows it.
    ``rearm_each_frame``: release and re-press the frame's keys with every F6.
    ``settle_s``: a fixed wait after a plan, for use without a counter.
    """

    keys: Keys
    counter: Optional[Any] = None
    key_gap_s: float = 0.0
    max_retries: int = 5
    rearm_each_frame: bool = False
    settle_s: float = 0.0
    now: Callable[[], float] = time.perf_counter
    sleep: Callable[[float], None] = time.sleep
    frame: int = 0
    paused: bool = field(default=False, init=False)

    def start(self) -> None:
        """Pause the game (assumed running when the session starts)."""
        if not self.paused:
            self.keys.release_all()
            self.keys.tap(PAUSE)
            self.paused = True

    def stop(self) -> None:
        """Release everything and let the game run again."""
        if self.paused:
            self.keys.release_all()
            self.keys.tap(PAUSE)
            self.paused = False

    def advance(self, plan: Plan) -> Timing:
        if not self.paused:
            raise RuntimeError("SteppedClock.advance needs a paused game: call start() first")
        timing = Timing(frames=len(plan))
        t0 = self.now()
        for frame_keys in plan:
            if self.rearm_each_frame:
                self.keys.release_all()
            keys_changed = frozenset(frame_keys) != self.keys.held
            self.keys.hold(frame_keys)
            if keys_changed and self.key_gap_s:
                self.sleep(self.key_gap_s)
            timing.retries += self._press(STEP)
            self.frame += 1
        self.sleep(self.settle_s)
        timing.wall_s = self.now() - t0
        return timing

    def _press(self, key: str) -> int:
        """Tap ``key``; with a counter, confirm it moved and retry if the game dropped the press.
        Returns the number of retries."""
        if self.counter is None:
            self.keys.tap(key)
            return 0
        before = self.counter.read()
        held = self.keys.held
        for attempt in range(self.max_retries + 1):
            if attempt >= 2 and held:
                # Repeated drops with keys held: release them, then press them again before the step.
                self.keys.release_all()
                self.sleep(self.key_gap_s)
                self.keys.hold(held)
                self.sleep(self.key_gap_s)
            self.keys.tap(key)
            if self.counter.wait_change(before):
                return attempt
        raise StepError(f"the game ignored {key} {self.max_retries + 1} times "
                        f"(frame {self.frame}, keys held: {sorted(held) or 'none'})")

    def rewind(self, frames: int) -> Timing:
        """Step back ``frames`` frames. The game is deterministic, so replaying the same keys from
        there reproduces the same frames; different keys change the past."""
        if not self.paused:
            raise RuntimeError("rewind needs a paused game")
        if frames > self.frame:
            raise ValueError(f"cannot rewind {frames} frames from frame {self.frame}")
        timing = Timing(frames=-frames)
        t0 = self.now()
        self.keys.release_all()
        for _ in range(frames):
            timing.retries += self._press(REWIND)
            self.frame -= 1
        self.sleep(self.settle_s)
        timing.wall_s = self.now() - t0
        return timing


@dataclass
class RealtimeClock:
    """The game runs; each plan entry is held for one frame period of wall-clock time.

    Frames are counted from the schedule, not observed, so ``frame`` is the scheduled frame index.
    A frame whose keys are set after its scheduled start is counted as late.
    """

    keys: Keys
    now: Callable[[], float] = time.perf_counter
    sleep: Callable[[float], None] = time.sleep
    frame: int = 0
    _origin: float | None = field(default=None, init=False)

    def start(self) -> None:
        self._origin = self.now() - self.frame / FPS

    def stop(self) -> None:
        self.keys.release_all()
        self._origin = None

    def advance(self, plan: Plan) -> Timing:
        if self._origin is None:
            raise RuntimeError("call start() first")
        t0 = self.now()
        late = 0
        for frame_keys in plan:
            begin = self._origin + self.frame / FPS
            if self.now() > begin + 1 / FPS:
                late += 1
            self.keys.hold(frame_keys)
            self.frame += 1
            remaining = self._origin + self.frame / FPS - self.now()
            if remaining > 0:
                self.sleep(remaining)
        return Timing(frames=len(plan), wall_s=self.now() - t0, late_frames=late)

    def rewind(self, frames: int) -> Timing:
        raise RuntimeError("rewind is only available in the stepped regime")


def change_difficulty(clock: Clock, steps: int, *, settle_s: float = 0.5,
                      sleep: Callable[[float], None] = time.sleep) -> None:
    """Operator action: move the bot ``steps`` levels up (positive) or down (negative).

    Training mode: Tab+D raises the level (Easy, Medium, Hard, Extreme, Chosen, Evasive), Tab+A
    lowers it. All fighting keys are released first, so the policy cannot act during the switch.
    """
    keys: Keys = clock.keys  # type: ignore[attr-defined]
    keys.release_all()
    step_key = DIFFICULTY_UP if steps > 0 else DIFFICULTY_DOWN
    for _ in range(abs(steps)):
        keys.chord([MENU, step_key])
        sleep(settle_s)
