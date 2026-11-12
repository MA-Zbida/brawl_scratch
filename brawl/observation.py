"""The observation: one vector of measured quantities per decision (research/DESIGN.md section 5.2).

FEATURES names every entry, in order. Positions are relative to the stage centre and scaled so
that the screen spans about -1 to 1; a fighter that is not present keeps its last seen position
with ``present`` 0. The agent is Wu Shang.

What cannot be seen in one frame (a dodge on cooldown, jumps already used) is carried two ways:
the last ``HISTORY`` actions, and TIMERS, which count decisions since things that were measured.
The timers are evidence, not game state: a press is an attempt (it can fall in stun), and the
opponent's jumps are guessed from its upward starts. Measured on held-out sessions 2026-11-11
(results/probes/memory.md): they improve the prediction of the agent being hit in every run.
"""
from __future__ import annotations

import numpy as np

from brawl import actions, stage
from brawl.sense.perception import Sense

AGENT, OPPONENT = "wu_shang", "mordex"
HISTORY = 4
X, Y = stage.WIDTH / 2, stage.HEIGHT / 2  # position scale
V = 60.0                                   # velocity scale, pixels per decision
LEDGE = stage.RIGHT - stage.LEFT
HELD = {"unarmed": 0, "spear": 1, "scythe": 1, "gauntlets": 2}  # none, the legend's first weapon, its second
CLIP = 80      # decisions (4 s at k = 3): longer than any cooldown
RISING = -0.25  # scaled vertical velocity below which a fighter has started upwards

_FIGHTER = ("x", "y", "vx", "vy", "present", "seen", "unarmed", "weapon_1", "weapon_2", "grounded", "offstage",
            "below", "to_left_ledge", "to_right_ledge", "damage")
BASE: tuple[str, ...] = (
    *(f"agent_{n}" for n in _FIGHTER), *(f"opponent_{n}" for n in _FIGHTER),
    "opponent_dx", "opponent_dy", "opponent_distance",
    "spawn_dx", "spawn_dy", "spawn_present", "thrown_dx", "thrown_dy", "thrown_present",
    *(f"action_{i}_{n}" for i in range(1, HISTORY + 1)
      for n in (*("dir_" + d for d in actions.DIRECTION_NAMES), *("btn_" + b for b in actions.BUTTON_NAMES))))
_SINCE = (*(f"since_press_{b}" for b in actions.BUTTON_NAMES[1:]), "agent_since_ground", "opponent_since_ground",
          "agent_since_hit", "opponent_since_hit", "agent_since_respawn", "opponent_since_respawn")
TIMERS: tuple[str, ...] = (*_SINCE, "agent_jumps_since_ground", "opponent_rises_since_ground")
FEATURES: tuple[str, ...] = (*BASE, *TIMERS)
SIZE = len(FEATURES)


class Timers:
    """Decisions since measured events (scaled by CLIP), and jumps since the ground (up to 4, scaled)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.since = dict.fromkeys(_SINCE, CLIP)
        self.jumps = self.rises = self.button = 0
        self.present = {"agent": True, "opponent": True}
        self.rising = False

    def acted(self, action) -> None:
        """The action chosen after the last update. A press is a button that was not held before."""
        button = int(action[1])
        if button and button != self.button:
            self.since[f"since_press_{actions.BUTTON_NAMES[button]}"] = 0
            self.jumps += actions.BUTTON_NAMES[button] == "jump"
        self.button = button

    def update(self, grounded, present, opponent_vy: float, hit) -> list[float]:
        """One decision later. ``grounded``, ``present``, ``hit`` are (agent, opponent); ``hit``:
        took damage during the decision that just ended."""
        for name in self.since:
            self.since[name] = min(CLIP, self.since[name] + 1)
        for who, g, p, h in zip(("agent", "opponent"), grounded, present, hit):
            if g:
                self.since[f"{who}_since_ground"] = 0
            if h:
                self.since[f"{who}_since_hit"] = 1
            if p and not self.present[who]:
                self.since[f"{who}_since_respawn"] = 0
            self.present[who] = bool(p)
        if grounded[0]:
            self.jumps = 0
        if grounded[1]:
            self.rises = 0
        rising = opponent_vy < RISING
        self.rises += rising and not self.rising
        self.rising = rising
        return [*(self.since[n] / CLIP for n in _SINCE), min(self.jumps, 4) / 4, min(self.rises, 4) / 4]


def add_timers(obs: np.ndarray, action: np.ndarray, taken: np.ndarray, dealt: np.ndarray) -> np.ndarray:
    """Stored decisions from before the timers existed, with the timers appended: the stored rows
    are replayed through the same ``Timers`` the builder uses."""
    col = {n: obs[:, BASE.index(n)] for n in ("agent_grounded", "opponent_grounded", "agent_present",
                                              "opponent_present", "opponent_vy")}
    timers, out = Timers(), np.zeros((len(obs), len(TIMERS)), np.float32)
    for i in range(len(obs)):
        hit = (taken[i - 1] > 0, dealt[i - 1] > 0) if i else (False, False)
        out[i] = timers.update((col["agent_grounded"][i] > 0, col["opponent_grounded"][i] > 0),
                               (col["agent_present"][i] > 0, col["opponent_present"][i] > 0), col["opponent_vy"][i], hit)
        timers.acted(action[i])
    return np.hstack([obs[:, :len(BASE)], out])


class Observation:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last = {AGENT: stage.CENTRE, OPPONENT: stage.CENTRE}
        self.history: list[tuple[int, int]] = [(0, 0)] * HISTORY
        self.timers = Timers()

    def acted(self, action) -> None:
        """Tell the builder which action followed the last observation."""
        self.history = (self.history + [(int(action[0]), int(action[1]))])[-HISTORY:]
        self.timers.acted(action)

    def _fighter(self, sense: Sense, who: str) -> list[float]:
        f = sense.fighters.get(who)
        held = [0.0, 0.0, 0.0]
        vx = vy = 0.0
        if f is not None:
            self.last[who] = f.box.position
            vx, vy = f.velocity
            held[HELD[f.box.cls.rsplit("_", 1)[1]]] = 1.0
        x, y = self.last[who]
        p = stage.place(x, y)
        return [(x - stage.CENTRE[0]) / X, (y - stage.CENTRE[1]) / Y, float(np.clip(vx / V, -3, 3)),
                float(np.clip(vy / V, -3, 3)), float(f is not None), float(f is not None and f.seen), *held,
                float(p.grounded), float(p.offstage), float(p.below), float(np.clip(p.to_left_ledge / LEDGE, -1, 2)),
                float(np.clip(p.to_right_ledge / LEDGE, -1, 2)), min(sense.damage.get(who, 0.0) / 300.0, 1.0)]

    def _nearest(self, sense: Sense, cls: str) -> list[float]:
        ax, ay = self.last[AGENT]
        found = [b.centre for b in sense.weapons if b.cls == cls]
        if not found:
            return [0.0, 0.0, 0.0]
        x, y = min(found, key=lambda c: (c[0] - ax) ** 2 + (c[1] - ay) ** 2)
        return [(x - ax) / X, (y - ay) / Y, 1.0]

    def build(self, sense: Sense) -> np.ndarray:
        agent, opponent = self._fighter(sense, AGENT), self._fighter(sense, OPPONENT)
        (ax, ay), (ox, oy) = self.last[AGENT], self.last[OPPONENT]
        dx, dy = (ox - ax) / X, (oy - ay) / Y
        history: list[float] = []
        for d, b in self.history:
            history += [float(i == d) for i in range(actions.SHAPE[0])] + [float(i == b) for i in range(actions.SHAPE[1])]
        at = _FIGHTER.index
        timers = self.timers.update((agent[at("grounded")] > 0, opponent[at("grounded")] > 0),
                                    (agent[at("present")] > 0, opponent[at("present")] > 0), opponent[at("vy")],
                                    (sense.events.damage_taken > 0, sense.events.damage_dealt > 0))
        out = np.array([*agent, *opponent, dx, dy, float(np.hypot(dx, dy)), *self._nearest(sense, "weapon_spawn"),
                        *self._nearest(sense, "weapon_thrown"), *history, *timers], dtype=np.float32)
        assert out.shape == (SIZE,)
        return out
