from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CursorBounds:
    left: int
    top: int
    right: int
    bottom: int


class TargetCursor:
    """Move system cursor to normalized training target coordinates on Windows."""

    def __init__(self, bounds: Optional[CursorBounds] = None, update_every_steps: int = 1):
        self.bounds = bounds
        self.update_every_steps = max(1, int(update_every_steps))
        self._step = 0

        try:
            import ctypes

            self._user32 = ctypes.windll.user32
            self._enabled = True
        except Exception:
            self._user32 = None
            self._enabled = False

    def update_target_norm(self, x_norm: float, y_norm: float, visible: bool = True) -> None:
        if not self._enabled or not visible:
            return

        user32 = self._user32
        if user32 is None:
            return

        self._step += 1
        if (self._step % self.update_every_steps) != 0:
            return

        x_norm = max(0.0, min(1.0, float(x_norm)))
        y_norm = max(0.0, min(1.0, float(y_norm)))

        if self.bounds is None:
            left = 0
            top = 0
            right = int(user32.GetSystemMetrics(0))
            bottom = int(user32.GetSystemMetrics(1))
        else:
            left = int(self.bounds.left)
            top = int(self.bounds.top)
            right = int(self.bounds.right)
            bottom = int(self.bounds.bottom)

        width = max(1, right - left)
        height = max(1, bottom - top)

        x_px = left + int(x_norm * (width - 1))
        y_px = top + int(y_norm * (height - 1))

        try:
            user32.SetCursorPos(int(x_px), int(y_px))
        except Exception:
            pass

    def stop(self) -> None:
        return
