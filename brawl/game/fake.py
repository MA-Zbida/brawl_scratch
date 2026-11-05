"""A deterministic stand-in for Brawlhalla's training mode, for tests and smoke runs.

It implements the keyboard backend and the camera interface. It models what the design relies
on: F7 toggles pause; F6 advances one frame while paused, with the held keys; F5 steps back one
frame and restores that frame's state exactly; Tab+D / Tab+A change the bot's difficulty. One
fighter moves with left/right; the bot's state evolves as a function of the state alone, so
rewinding and replaying the same keys reproduces the same frames, as in the real game.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brawl.game.keys import DIFFICULTY_DOWN, DIFFICULTY_UP, LEFT, MENU, PAUSE, REWIND, RIGHT, STEP

LEVELS = ("easy", "medium", "hard", "extreme", "chosen", "evasive")


@dataclass
class FakeGame:
    needs_rearm: bool = False  # True: the game only applies keys pressed since the last step
    paused: bool = False
    frame: int = 0
    x: int = 0
    bot: int = 7
    difficulty: int = 0
    held: set[str] = field(default_factory=set)
    log: list[tuple[str, str]] = field(default_factory=list)
    _since_step: set[str] = field(default_factory=set)
    _history: list[tuple[int, int]] = field(default_factory=list)
    _new_frame: bool = True

    def __post_init__(self) -> None:
        self._history = [(self.x, self.bot)]

    # -- keyboard backend ----------------------------------------------------------------
    def key_down(self, key: str) -> None:
        self.log.append(("down", key))
        if key == PAUSE:
            self.paused = not self.paused
        elif key == STEP and self.paused:
            keys = self.held & self._since_step if self.needs_rearm else set(self.held)
            self._advance(keys)
        elif key == REWIND and self.paused and self.frame > 0:
            self.frame -= 1
            self.x, self.bot = self._history[self.frame]
            self._new_frame = True
        elif key in (DIFFICULTY_UP, DIFFICULTY_DOWN) and MENU in self.held:
            self.difficulty = min(len(LEVELS) - 1, max(0, self.difficulty + (1 if key == DIFFICULTY_UP else -1)))
        else:
            self.held.add(key)
            self._since_step.add(key)

    def key_up(self, key: str) -> None:
        self.log.append(("up", key))
        self.held.discard(key)

    # -- game ------------------------------------------------------------------------------
    def _advance(self, keys: set[str]) -> None:
        self.x += (RIGHT in keys) - (LEFT in keys)
        self.bot = (self.bot * 31 + self.x + 17) % 1009
        self.frame += 1
        del self._history[self.frame:]
        self._history.append((self.x, self.bot))
        self._since_step.clear()
        self._new_frame = True

    def tick(self, frames: int = 1) -> None:
        """Let the running (unpaused) game advance in real time."""
        for _ in range(frames):
            if not self.paused:
                self._advance(set(self.held))

    # -- camera ----------------------------------------------------------------------------
    def grab(self):
        """Like dxcam: a new image if the screen changed since the last grab, else None."""
        if not self._new_frame:
            return None
        self._new_frame = False
        return np.array([[[self.frame, self.x, self.bot]]], dtype=np.int64)
