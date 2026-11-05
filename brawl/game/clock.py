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
    repairs: int = 0      # stepped only: presses made to undo a press that landed twice


class StepError(RuntimeError):
    """The game did not respond to a frame step or rewind, even after retries."""

    def __init__(self, message: str, key: Optional[str] = None) -> None:
        super().__init__(message)
        self.key = key  # the key the game ignored, if that is what happened


class Clock(Protocol):
    frame: int

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def advance(self, plan: Plan) -> Timing: ...

    def rewind(self, frames: int) -> Timing: ...


@dataclass
class SteppedClock:
    """The game stays paused; it advances one frame per F6 press, with the plan's keys held.

    ``counter``: the toolbar's frame counter (``brawl.game.counter.FrameCounter``). With it, the
    clock's ``frame`` is the game's own frame number, and every F6 and F5 is confirmed by reading
    that number. A dropped press is pressed again, waiting longer each time; a press that landed
    late, together with its retry, is undone: rewinding is exact, and so is replaying a frame with
    the keys that first produced it (``keys_at``), because the game is deterministic.
    The game drops F6 when fight keys change at the same moment (gate T0), so ``key_gap_s``
    separates a key change from the F6 that follows it.
    ``rearm_each_frame``: release and re-press the frame's keys with every F6.
    ``settle_s``: a fixed wait after a plan, for use without a counter.
    """

    keys: Keys
    counter: Optional[Any] = None
    key_gap_s: float = 0.0
    max_retries: int = 5
    max_wait_s: float = 2.0
    rearm_each_frame: bool = False
    settle_s: float = 0.0
    now: Callable[[], float] = time.perf_counter
    sleep: Callable[[float], None] = time.sleep
    frame: int = 0
    paused: bool = field(default=False, init=False)
    keys_at: dict[int, frozenset[str]] = field(default_factory=dict, init=False)
    repairs: int = field(default=0, init=False)

    def start(self) -> None:
        """Pause the game (assumed running when the session starts)."""
        if not self.paused:
            self.keys.release_all()
            self.keys.tap(PAUSE)
            self.paused = True
        self.sync()

    def stop(self) -> None:
        """Release everything and let the game run again."""
        if self.paused:
            self.keys.release_all()
            self.keys.tap(PAUSE)
            self.paused = False

    def sync(self) -> None:
        """Adopt the game's frame number (with a counter). Frames before it have unknown keys."""
        if self.counter is None:
            return
        value = self.counter.value()
        if value is None:
            raise StepError("cannot read the frame counter (is the game paused, with no menu open?)")
        self.frame = value
        self.keys_at.clear()

    def advance(self, plan: Plan) -> Timing:
        if not self.paused:
            raise RuntimeError("SteppedClock.advance needs a paused game: call start() first")
        timing = Timing(frames=len(plan))
        t0, repairs = self.now(), self.repairs
        for frame_keys in plan:
            self._check()
            if self.rearm_each_frame:
                self.keys.release_all()
            keys_changed = frozenset(frame_keys) != self.keys.held
            self.keys.hold(frame_keys)
            if keys_changed and self.key_gap_s:
                self.sleep(self.key_gap_s)
            timing.retries += self._press(STEP)
        self.sleep(self.settle_s)
        timing.wall_s = self.now() - t0
        timing.repairs = self.repairs - repairs
        return timing

    def rewind(self, frames: int) -> Timing:
        """Step back ``frames`` frames. The game is deterministic, so replaying the same keys from
        there reproduces the same frames; different keys change the past."""
        if not self.paused:
            raise RuntimeError("rewind needs a paused game")
        if self.counter is None and frames > self.frame:
            raise ValueError(f"cannot rewind {frames} frames from frame {self.frame}")
        timing = Timing(frames=-frames)
        t0, repairs = self.now(), self.repairs
        if self.keys.held:
            self.keys.release_all()
            self.sleep(self.key_gap_s)  # a release right before F5 is dropped like a press before F6
        for _ in range(frames):
            self._check()
            timing.retries += self._press(REWIND)
        self.sleep(self.settle_s)
        timing.wall_s = self.now() - t0
        timing.repairs = self.repairs - repairs
        return timing

    def _press(self, key: str) -> int:
        """Move one frame with ``key`` (F6 forward, F5 back). With a counter, confirm it on the
        counter's number and retry dropped presses. Returns the number of retries."""
        step = 1 if key == STEP else -1
        if self.counter is None:
            self.keys.tap(key)
            self.frame += step
            return 0
        goal, held = self.frame + step, self.keys.held
        for attempt in range(self.max_retries + 1):
            # Only the step key is pressed again: releasing and re-pressing a fight key would be a
            # new press (a second jump, a new attack) that the frame did not have.
            # Each retry holds the key longer and waits longer: the game can be slow to respond,
            # and pressing again too early is what makes a press land twice.
            self.keys.tap(key, hold_s=self.keys.tap_s * (1 + attempt))
            wait = min(self.max_wait_s, self.counter.timeout_s * 2 ** attempt)
            value = self.counter.wait_value(self.frame, timeout_s=wait)
            if value is not None:
                break
        else:
            raise StepError(f"the game ignored {key} {self.max_retries + 1} times "
                            f"(frame {self.frame}, keys held: {sorted(held) or 'none'})", key=key)
        if key == STEP:
            self.keys_at[goal] = held
        self.frame = goal
        if value != goal:
            self._repair(value)
        return attempt

    def _check(self) -> None:
        """Before a press: a late press may have landed since the last one. Undo it."""
        if self.counter is not None:
            value = self.counter.value()
            if value != self.frame:
                self._repair(value)

    def _repair(self, value: Optional[int], max_presses: int = 30) -> None:
        """Bring the game from frame ``value`` back to ``self.frame``: back with F5, forward with
        F6 holding the keys that first produced each frame. Restores the held keys afterwards."""
        held = self.keys.held
        for _ in range(max_presses):
            if value is None:
                value = self.counter.value()
                if value is None:
                    raise StepError(f"cannot read the frame counter (clock at frame {self.frame})")
            if value == self.frame:
                break
            self.repairs += 1
            if value > self.frame:
                if self.keys.held:
                    self.keys.release_all()
                    self.sleep(self.key_gap_s)
                self.keys.tap(REWIND)
            else:
                keys = self.keys_at.get(value + 1)
                if keys is None:
                    raise StepError(f"cannot replay frame {value + 1}: its keys are unknown")
                if keys != self.keys.held:
                    self.keys.hold(keys)
                    self.sleep(self.key_gap_s)
                self.keys.tap(STEP)
            moved = self.counter.wait_value(value, timeout_s=self.max_wait_s)
            value = moved if moved is not None else self.counter.value()
        else:
            raise StepError(f"could not bring the game back to frame {self.frame} (it shows {value})")
        if self.keys.held != held:
            self.keys.hold(held)
            self.sleep(self.key_gap_s)


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
