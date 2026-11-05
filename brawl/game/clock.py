"""Game time: one interface, two regimes (research/DESIGN.md section 3).

A *plan* is the list of key sets to hold, one per game frame, for one decision. ``advance(plan)``
plays it: in ``stepped`` mode by frame-stepping the paused game once per entry, in ``realtime``
mode by holding each set for one frame period (1/60 s) of wall-clock time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from brawl.game.keys import DIFFICULTY_DOWN, DIFFICULTY_UP, MENU, PAUSE, REWIND, STEP, Keys

FPS = 60
Plan = Sequence[frozenset[str]]


@dataclass
class Timing:
    """What one ``advance`` cost. Recorded with every step."""

    frames: int = 0
    wall_s: float = 0.0
    late_frames: int = 0  # realtime only: frames whose keys were set after the frame began


class Clock(Protocol):
    frame: int

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def advance(self, plan: Plan) -> Timing: ...

    def rewind(self, frames: int) -> Timing: ...


@dataclass
class SteppedClock:
    """The game stays paused; it advances one frame per F6 press, with the plan's keys held.

    ``rearm_each_frame``: release and re-press the frame's keys together with every F6, for the
    case where the game only applies keys pressed at the same moment as the step (Gate T2
    decides which mode is needed).
    """

    keys: Keys
    rearm_each_frame: bool = False
    now: Callable[[], float] = time.perf_counter
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
        t0 = self.now()
        for frame_keys in plan:
            if self.rearm_each_frame:
                self.keys.release_all()
            self.keys.hold(frame_keys)
            self.keys.tap(STEP)
            self.frame += 1
        return Timing(frames=len(plan), wall_s=self.now() - t0)

    def rewind(self, frames: int) -> Timing:
        """Step back ``frames`` frames. The game is deterministic, so replaying the same keys from
        there reproduces the same frames; different keys change the past."""
        if not self.paused:
            raise RuntimeError("rewind needs a paused game")
        if frames > self.frame:
            raise ValueError(f"cannot rewind {frames} frames from frame {self.frame}")
        t0 = self.now()
        self.keys.release_all()
        for _ in range(frames):
            self.keys.tap(REWIND)
        self.frame -= frames
        return Timing(frames=-frames, wall_s=self.now() - t0)


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
