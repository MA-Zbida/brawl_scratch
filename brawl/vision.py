"""Fighter crops: what an image branch sees (research/DESIGN.md section 5.3).

A crop is a square window of ``WINDOW`` screen pixels (on the 1920 x 1080 screen) centred on a
fighter's torso point, in grey, resized to ``size`` x ``size``. The window holds the fighter and its
weapon swing (fighter plus weapon never exceeded 151 x 174 px in the labels). The part of the
window beyond the screen's edge is black. The same function serves recorded half-size frames and
live full-size frames, so the probes and the agent see the same picture.
"""
from __future__ import annotations

import numpy as np

WINDOW = 256
SCREEN = 1920


def crop(frame: np.ndarray, point: tuple[float, float], size: int = 96, window: int = WINDOW) -> np.ndarray:
    """``frame``: BGR, any scale of the 1920 x 1080 screen; ``point``: on the 1920 x 1080 screen."""
    import cv2

    scale = frame.shape[1] / SCREEN
    half = window * scale / 2
    cx, cy = point[0] * scale, point[1] * scale
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    side = int(round(2 * half))
    window_bgr = np.zeros((side, side, 3), np.uint8)
    h, w = frame.shape[:2]
    a0, b0, a1, b1 = max(0, x0), max(0, y0), min(w, x0 + side), min(h, y0 + side)
    if a1 > a0 and b1 > b0:
        window_bgr[b0 - y0:b1 - y0, a0 - x0:a1 - x0] = frame[b0:b1, a0:a1]
    out = cv2.cvtColor(window_bgr, cv2.COLOR_BGR2GRAY)  # only the window: the whole frame costs about 1 ms
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA) if side != size else out
