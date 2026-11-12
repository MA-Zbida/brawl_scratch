"""The HUD at the top right: each fighter's damage, and the badge that marks a lost or scored stock.

Two slots, each a portrait over a ring whose colour is the fighter's damage. The slots are ordered
by score, so the fighters **swap sides** when the lead changes: who is in a slot is read from the
portrait every frame, on a patch under the face that the badges never cover. On a KO the loser's
ring turns white at once, and a quarter of a second later a red "-1" covers the loser's portrait
and a cyan "+2" the scorer's, for about two seconds.

All positions are in 1920 x 1080 pixels and scaled to the frame given, so the same reader works on
the live capture and on the half-size frames of a recording. Measured by gate R1
(scripts/gate_r1.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SLOT_DX = 91  # the right slot is the left one shifted by this
RING = ((1755, 103), (1761, 103), (1767, 103), (1773, 101), (1779, 99), (1785, 93), (1791, 87))
IDENTITY = (1750, 80, 1776, 90)  # x0, y0, x1, y1: Mordex's blue disc or Wu Shang's beard
BADGE = (1740, 48, 1780, 76)     # where the "-1" and "+2" badges are drawn

# Ring colour (R, G, B) -> damage: white, yellow, orange, red, then darker reds.
CURVE = ((255, 255, 255, 0.0), (255, 255, 0, 50.0), (255, 153, 0, 100.0), (255, 0, 0, 150.0),
         (191, 0, 0, 200.0), (140, 0, 0, 250.0), (74, 0, 0, 300.0), (0, 0, 0, 351.0))
# Furthest a ring colour may be from the curve (RGB distance). A clean ring is within 10, JPEG
# included; while a badge drops in, it tints the whole ring for a few frames, 12 to 30 away.
RING_TOLERANCE = 12.0


def damage(rgb) -> float | None:
    """The damage a ring colour stands for: the nearest point of the colour curve, or None when
    the colour is not on it (something else is on screen)."""
    c = np.asarray(rgb, dtype=np.float64)
    best, best_d = None, RING_TOLERANCE
    for (*a, da), (*b, db) in zip(CURVE[:-1], CURVE[1:]):
        a, b = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
        t = float(np.clip(np.dot(c - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0))
        d = float(np.linalg.norm(c - (a + t * (b - a))))
        if d < best_d:
            best, best_d = da + t * (db - da), d
    return best


@dataclass(frozen=True)
class Slot:
    who: str | None       # "wu_shang", "mordex", or None when the portrait cannot be read
    damage: float | None  # None when the ring cannot be read
    badge: str | None     # "lost" (red -1), "scored" (cyan +2), or None


def _slot(frame: np.ndarray, dx: int) -> Slot:
    sx, sy = frame.shape[1] / 1920, frame.shape[0] / 1080

    def patch(box) -> np.ndarray:  # pixels of a box, as RGB rows
        x0, y0, x1, y1 = box
        return frame[int(y0 * sy):int(np.ceil(y1 * sy)), int((x0 + dx) * sx):int(np.ceil((x1 + dx) * sx)),
                     ::-1].reshape(-1, 3).astype(np.int32)

    ring = np.median([frame[int(y * sy), int((x + dx) * sx), ::-1] for x, y in RING], axis=0)
    r, g, b = np.median(patch(IDENTITY), axis=0)
    who = "mordex" if b > r + 25 and r < 110 else "wu_shang" if r > b + 40 and r > 110 else None
    p = patch(BADGE)
    red = float(((p[:, 0] > 180) & (p[:, 1] < 90) & (p[:, 2] < 90)).mean())
    cyan = float(((p[:, 0] < 120) & (p[:, 1] > 170) & (p[:, 2] > 200)).mean())
    return Slot(who, damage(ring), "lost" if red > 0.1 else "scored" if cyan > 0.15 else None)


def read(frame: np.ndarray) -> dict[str, Slot] | None:
    """Each fighter's slot in a BGR frame, by name. None when the frame does not show the HUD as
    it should be: a ring off the colour curve, or the two portraits not one of each fighter."""
    left, right = _slot(frame, 0), _slot(frame, SLOT_DX)
    if left.damage is None or right.damage is None or {left.who, right.who} != {"wu_shang", "mordex"}:
        return None
    return {left.who: left, right.who: right}
