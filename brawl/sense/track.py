"""The two fighters across frames: the detector reads one frame, the game is continuous.

Each update takes one frame's boxes from the detector (run at a low confidence, 0.05) and keeps one
state per fighter. There is one Wu Shang and one Mordex, so nothing has to be matched up:

- A sure box (score >= SURE) is taken, unless it lies further from where the fighter should be
  than a fighter moves in a frame (LEAP): that box is taken only if the next frame confirms it.
- A doubtful box is taken when it lies within REACH of where the fighter should be: a fighter
  half hidden by a hit flash is there, at low confidence.
- With no box taken, the fighter is carried at its last velocity for up to CARRY frames and
  marked ``seen=False``. After that, or when the HUD reports its lost stock, or when the carried
  position leaves the frame, it is absent.
- The held weapon changes only when MAJORITY of the last VOTE boxes taken agree: it changes at
  pickups, throws and KOs, not from one frame to the next.

Weapons not held are in ``tracker.weapons`` after each update. A spawn marker stays where it is
for seconds, so a spawn box is reported only when its marker was seen in 2 of the last 3 frames, and
one box per marker. It is not carried through frames without a box: a marker vanishes at once when
picked up (5 of 10 false markers on the test frames were carried ones). A thrown weapon moves too
fast to wait for: it is reported as detected.

The constants were measured on 3,000 consecutive frames (research/LAB_NOTEBOOK.md, 2026-11-11).
The state is plain data: ``copy.deepcopy(tracker)`` before a step restores it after a rewind.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from brawl.sense.truth import FIGHTERS, Box, identity

SURE = 0.25      # a box at or above this score needs no support from the state
REACH = 60.0     # pixels per frame: a doubtful box this close to the expected position is the fighter
LEAP = 150.0     # pixels per frame: no fighter moves further (99% of steps are under 62)
CARRY = 8        # frames a fighter is carried without a box (hit flashes hide one for 1-8)
VOTE, MAJORITY = 7, 5
WEAPON = 0.25    # weapon boxes below this score are ignored
SPAWN_MATCH = 40.0  # pixels per frame: a spawn box this close to a known marker is that marker


@dataclass(frozen=True)
class Fighter:
    box: Box                       # its class is the voted held weapon, its point the torso
    velocity: tuple[float, float]  # pixels per update
    seen: bool                     # False: carried from earlier frames, no box this frame
    age: int                       # updates since a box was last taken


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class Tracker:
    def __init__(self, step: int = 1, size: tuple[int, int] = (1920, 1080)) -> None:
        """``step``: game frames between updates; distances and the carry time scale with it."""
        self.reach, self.leap, self.carry, self.size = REACH * step, LEAP * step, max(1, round(CARRY / step)), size
        self.last: dict[str, tuple[Box, tuple[float, float], int]] = {}  # box taken, velocity, age
        self.recent: dict[str, list[str]] = {who: [] for who in FIGHTERS}
        self.held: dict[str, str | None] = {who: None for who in FIGHTERS}
        self.pending: dict[str, tuple[float, float] | None] = {who: None for who in FIGHTERS}
        self.match = SPAWN_MATCH * step
        self.spawns: list[dict] = []   # box, seen (the last 3 frames)
        self.weapons: list[Box] = []   # confirmed spawn markers and thrown weapons, after the last update

    def _forget(self, who: str) -> None:
        self.last.pop(who, None)
        self.recent[who], self.held[who] = [], None  # it comes back as whatever it is first seen as

    def update(self, boxes: list[Box], lost: tuple[str, ...] | set[str] = ()) -> dict[str, Fighter]:
        """The fighters present after this frame. ``lost``: fighters whose lost stock the HUD shows."""
        self.weapons = self._spawns([b for b in boxes if b.cls == "weapon_spawn" and b.score >= WEAPON]) + [
            b for b in boxes if b.cls == "weapon_thrown" and b.score >= WEAPON]
        out: dict[str, Fighter] = {}
        for who in FIGHTERS:
            if who in lost:
                self._forget(who)
            mine = [b for b in boxes if identity(b.cls) == who]
            box = max(mine, key=lambda b: b.score) if mine else None
            state = self.last.get(who)
            expect = age = None
            if state is not None:
                before, velocity, age = state
                expect = (before.position[0] + velocity[0] * (age + 1), before.position[1] + velocity[1] * (age + 1))
            take = confirmed = False
            if box is not None and box.score >= SURE:
                confirmed = self.pending[who] is not None and _distance(box.position, self.pending[who]) <= self.leap
                take = expect is None or _distance(box.position, expect) <= self.leap * (age + 1) or confirmed
            elif box is not None:
                take = expect is not None and _distance(box.position, expect) <= self.reach
            self.pending[who] = box.position if box is not None and box.score >= SURE and not take else None
            if take:
                if confirmed:  # it really is somewhere else: nothing known about how it moves there
                    self._forget(who)
                    state = None
                velocity = (0.0, 0.0) if state is None else (
                    (box.position[0] - state[0].position[0]) / (age + 1),
                    (box.position[1] - state[0].position[1]) / (age + 1))
                self.recent[who] = (self.recent[who] + [box.cls])[-VOTE:]
                cls, votes = Counter(self.recent[who]).most_common(1)[0]
                if self.held[who] is None or votes >= MAJORITY:
                    self.held[who] = cls
                self.last[who] = (box, velocity, 0)
                out[who] = Fighter(replace(box, cls=self.held[who]), velocity, True, 0)
            elif state is not None and age + 1 <= self.carry and 0 <= expect[0] < self.size[0] \
                    and 0 <= expect[1] < self.size[1]:
                before, velocity, age = state[0], state[1], age + 1
                dx, dy = velocity[0] * age, velocity[1] * age
                carried = Box(self.held[who], before.x0 + dx, before.y0 + dy, before.x1 + dx, before.y1 + dy,
                              score=0.0, point=expect)
                self.last[who] = (before, velocity, age)
                out[who] = Fighter(carried, velocity, False, age)
            else:
                self._forget(who)
        return out

    def _spawns(self, boxes: list[Box]) -> list[Box]:
        kept: list[Box] = []  # one box per marker: the most confident of those that cover each other
        for b in sorted(boxes, key=lambda b: -b.score):
            if not any(k.contains(*b.centre) or b.contains(*k.centre) for k in kept):
                kept.append(b)
        for spawn in self.spawns:
            near = [b for b in kept if _distance(b.centre, spawn["box"].centre) <= self.match]
            if near:
                spawn["box"] = min(near, key=lambda b: _distance(b.centre, spawn["box"].centre))
                kept.remove(spawn["box"])
            spawn["seen"] = (spawn["seen"] + [bool(near)])[-3:]
        self.spawns = [s for s in self.spawns if any(s["seen"])] + [{"box": b, "seen": [True]} for b in kept]
        return [s["box"] for s in self.spawns if s["seen"][-1] and sum(s["seen"]) >= 2]
