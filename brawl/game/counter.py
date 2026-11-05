"""The training toolbar's frame counter: ground truth for every frame step and rewind.

While training mode is paused, the toolbar shows the game's frame number between "F5 Prev" and
"F6 Next". A press of F6 or F5 happened if, and only if, that number changed. The game does drop
presses (about 17% of F6 presses when fight keys change on the same step, measured with a human
reading the counter, gate T0), so the clock confirms each press against this counter instead of
trusting it.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

import numpy as np

COUNTER_BOX = (700, 40, 760, 62)  # x0, y0, x1, y1 at 1920 x 1080


def crop_counter(image: Any, box: tuple[int, int, int, int] = COUNTER_BOX) -> np.ndarray:
    x0, y0, x1, y1 = box
    return np.asarray(image)[y0:y1, x0:x1].astype(np.float32).mean(axis=2)


def counter_changed(a: np.ndarray, b: np.ndarray) -> bool:
    """The counter is static UI text: a digit change moves dozens of pixels by about 50 grey
    levels; an unchanged counter moves none. Six pixels is a safe margin."""
    return int((np.abs(a - b) > 20).sum()) >= 6


class FrameCounter:
    """Reads the counter from the screen and waits for it to change."""

    def __init__(self, screen: Any, *, extract: Callable[[Any], np.ndarray] = crop_counter,
                 changed: Callable[[np.ndarray, np.ndarray], bool] = counter_changed,
                 timeout_s: float = 0.15, now: Callable[[], float] = time.perf_counter) -> None:
        self.screen = screen
        self.extract = extract
        self.changed = changed
        self.timeout_s = timeout_s
        self.now = now

    def read(self) -> np.ndarray:
        return self.extract(self.screen.grab(fresh=False).image)

    def wait_change(self, before: np.ndarray, timeout_s: Optional[float] = None) -> bool:
        """Poll fresh frames until the counter differs from ``before``; False on timeout."""
        deadline = self.now() + (self.timeout_s if timeout_s is None else timeout_s)
        while True:
            frame = self.screen.grab(fresh=True)
            if self.changed(before, self.extract(frame.image)):
                return True
            if self.now() >= deadline:
                return False
