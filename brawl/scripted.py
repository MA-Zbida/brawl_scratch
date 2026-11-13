"""A scripted player that shows every move, to label what a fighter is doing (scripts/probe_state.py).

It does not try to win. It picks one of the 54 actions, holds it for a few decisions, then stays
neutral for a pause, so each move is seen from its start to its end from a clean state; heavies are
sometimes held long (charged). Near a ledge it walks back; off the stage it tries to recover; when
unarmed it sometimes fetches a weapon, so all three weapon states appear. Every decision's action
is the label: exact, and at the agent's own timing.
"""
from __future__ import annotations

import numpy as np

from brawl import actions, stage

NONE, JUMP, LIGHT, HEAVY, DODGE, THROW = range(6)
LEFT, RIGHT, UP_LEFT, UP_RIGHT = (actions.DIRECTION_NAMES.index(n) for n in ("left", "right", "up-left", "up-right"))
MARGIN = 90.0  # pixels from a ledge at which it turns back


class Scripted:
    def __init__(self, who: str, seed: int = 0) -> None:
        """``who``: the fighter it plays, ``wu_shang`` or ``mordex``."""
        self.who, self.rng = who, np.random.default_rng(seed)
        self.action, self.hold, self.pause, self.fetch, self.air, self.picked = (0, 0), 0, 0, 0, 0, 0

    def __call__(self, sense) -> tuple[int, int]:
        me = sense.fighters.get(self.who)
        if me is None:  # KO'd or not seen: nothing to show
            self.hold = self.pause = self.fetch = 0
            return 0, 0
        x, y = me.box.position
        self.picked = max(0, self.picked - 1)
        place = stage.place(x, y)
        toward = RIGHT if x < stage.CENTRE[0] else LEFT
        if place.offstage or place.below:  # recover: up and toward the stage, a new jump press every 5 decisions
            self.hold = self.pause = self.fetch = 0
            self.air += 1
            up = UP_RIGHT if toward == RIGHT else UP_LEFT
            return (up, JUMP) if self.air % 5 == 1 else (up, DODGE) if self.air % 23 == 12 else (toward, NONE)
        self.air = 0
        if min(place.to_left_ledge, place.to_right_ledge) < MARGIN and not self.hold:
            self.pause = 0
            return toward, NONE
        if self.hold:
            self.hold -= 1
            return self.action
        if self.pause:
            self.pause -= 1
            return 0, 0
        spawns = [b.centre[0] for b in sense.weapons if b.cls == "weapon_spawn"]
        # after a pickup the held weapon is read a few frames late: do not press again, it would throw the weapon
        if not self.picked and (self.fetch or (spawns and me.box.cls.endswith("_unarmed") and self.rng.random() < 0.3)):
            self.fetch = self.fetch - 1 if self.fetch else 60  # at most 3 s per attempt
            if not spawns or not me.box.cls.endswith("_unarmed"):
                self.fetch = 0
            else:
                target = min(spawns, key=lambda s: abs(s - x))
                if abs(target - x) < 25:
                    self.fetch, self.pause, self.picked = 0, 4, 40
                    return 0, THROW
                return (RIGHT if target > x else LEFT), NONE
        armed = not me.box.cls.endswith("_unarmed")  # a throw while armed loses the weapon: keep it for about a minute
        button = int(self.rng.choice(6, p=(0.24, 0.15, 0.24, 0.21, 0.15, 0.01) if armed or self.picked else
                                     (0.22, 0.14, 0.22, 0.20, 0.14, 0.08)))
        self.action = (int(self.rng.integers(9)), button)
        self.hold = int(self.rng.integers(2, 10)) if button == NONE else int(self.rng.choice((0, 0, 1, 3, 7))) if button == HEAVY else 0
        self.pause = int(self.rng.integers(2, 11))
        return self.action
