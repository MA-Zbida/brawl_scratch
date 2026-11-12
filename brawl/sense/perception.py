"""One call per decision frame: fighters, weapons, damage and events (research/DESIGN.md section 4).

The environment and the demonstration reader both call this, so a recorded frame and a live frame
become the same ``Sense`` by the same code. Recording frames are half size; the detector takes them
as they are and the boxes are scaled to the 1920 x 1080 screen that every position is given in.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from brawl.ledger import Events, Ledger
from brawl.sense import hud
from brawl.sense.track import Fighter, Tracker
from brawl.sense.truth import Box

WIDTH = 1920


@dataclass
class Sense:
    fighters: dict[str, Fighter]   # those present: seen this frame, or carried (track.py)
    weapons: list[Box]             # spawn markers and thrown weapons
    damage: dict[str, float]       # each fighter's last readable damage (0 before the first)
    events: Events                 # since the previous call
    readable: bool                 # the HUD was read on this frame


def _scaled(b: Box, f: float) -> Box:
    return replace(b, x0=b.x0 * f, y0=b.y0 * f, x1=b.x1 * f, y1=b.y1 * f,
                   point=None if b.point is None else (b.point[0] * f, b.point[1] * f))


class Perception:
    def __init__(self, detector, step: int = 3) -> None:
        """``detector``: brawl.sense.detectors.Yolo at confidence 0.05; ``step``: game frames per call."""
        self.detector, self.step = detector, step
        self.reset()

    def reset(self) -> None:
        """Forget everything (a new session, or a jump in time the state cannot follow)."""
        self.tracker, self.ledger = Tracker(self.step), Ledger()

    def __call__(self, frame: np.ndarray, index: int) -> Sense:
        """``index``: the game frame number, for the ledger's one-KO-per-fighter window."""
        reading = hud.read(frame)
        events = self.ledger.update(reading, index)
        lost = {who for who, slot in (reading or {}).items() if slot.badge == "lost"}
        lost |= {who for who, ko in (("wu_shang", events.ko_lost), ("mordex", events.ko_scored)) if ko}
        boxes = self.detector(frame)
        if frame.shape[1] != WIDTH:
            boxes = [_scaled(b, WIDTH / frame.shape[1]) for b in boxes]
        fighters = self.tracker.update(boxes, lost)
        damage = {who: float(d or 0.0) for who, d in self.ledger.damage.items()}
        return Sense(fighters, list(self.tracker.weapons), damage, events, reading is not None)
