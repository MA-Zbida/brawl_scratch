"""Candidate semantic action contract; NOT a reinterpretation of discrete27.

Names distinguish intent even when the game uses the same physical button. This
registry is for control qualification first. Existing policies keep action_space.py.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Literal

VERSION = "semantic57_candidate_v1"
ActionContext = Literal["any", "ground", "air"]


@dataclass(frozen=True)
class ControlContext:
    # None means uncertain, not "grounded/unarmed/dodge unavailable".
    grounded: bool | None = None
    armed: bool | None = None
    dodge_ready: bool | None = None
    controllable: bool = True
    mirrored: bool = False


@dataclass(frozen=True)
class SemanticAction:
    id: int
    name: str
    context: ActionContext
    hdir: int = 0
    vdir: int = 0
    button: str | None = None
    gravity_cancel: bool = False
    possession: bool | None = None

    def keys(self, *, mirrored: bool) -> tuple[set[str], set[str]]:
        hdir = -self.hdir if mirrored else self.hdir
        held = set()
        if hdir:
            held.add("d" if hdir > 0 else "a")
        if self.vdir:
            held.add("s" if self.vdir > 0 else "w")
        return held, {self.button} if self.button else set()


def _registry() -> tuple[SemanticAction, ...]:
    rows: list[SemanticAction] = []

    def add(name: str, context: ActionContext = "any", hdir: int = 0, vdir: int = 0,
            button: str | None = None, *, gc: bool = False,
            possession: bool | None = None) -> None:
        rows.append(SemanticAction(len(rows), name, context, hdir, vdir, button, gc, possession))

    add("NOOP")
    add("MOVE_TOWARD", hdir=1)
    add("MOVE_AWAY", hdir=-1)
    for suffix, hdir in [("", 0), ("_TOWARD", 1), ("_AWAY", -1)]:
        add("FAST_FALL" + suffix, "air", hdir, 1)
    for suffix, hdir in [("", 0), ("_TOWARD", 1), ("_AWAY", -1)]:
        add("JUMP" + suffix, hdir=hdir, button="space")
    add("GROUND_SPOT_DODGE", "ground", button="e")
    add("DASH_TOWARD", "ground", hdir=1, button="e")
    add("DASH_AWAY", "ground", hdir=-1, button="e")
    directions = [("SPOT", 0, 0), ("TOWARD", 1, 0), ("AWAY", -1, 0),
                  ("UP", 0, -1), ("DOWN", 0, 1), ("UP_TOWARD", 1, -1),
                  ("UP_AWAY", -1, -1), ("DOWN_TOWARD", 1, 1), ("DOWN_AWAY", -1, 1)]
    for name, hdir, vdir in directions:
        add("AIR_DODGE_" + name, "air", hdir, vdir, "e")
    attack_directions = [("NEUTRAL", 0, 0), ("TOWARD", 1, 0), ("AWAY", -1, 0), ("DOWN", 0, 1)]
    for family, context, button in [("GROUND_LIGHT", "ground", "num4"),
                                     ("SIGNATURE", "ground", "num6"),
                                     ("AIR_LIGHT", "air", "num4")]:
        for direction, hdir, vdir in attack_directions:
            add(f"{family}_{direction}", context, hdir, vdir, button)
    for family, vdir in [("RECOVERY", 0), ("GROUND_POUND", 1)]:
        for direction, hdir in [("NEUTRAL", 0), ("TOWARD", 1), ("AWAY", -1)]:
            add(f"{family}_{direction}", "air", hdir, vdir, "num6")
    for family, button in [("LIGHT", "num4"), ("SIGNATURE", "num6")]:
        for direction, hdir, vdir in attack_directions:
            add(f"GC_{family}_{direction}", "air", hdir, vdir, button, gc=True)
    add("PICKUP", button="num5", possession=False)
    for direction, hdir, vdir in directions:
        add("THROW_" + ("NEUTRAL" if direction == "SPOT" else direction),
            hdir=hdir, vdir=vdir, button="num5", possession=True)
    return tuple(rows)


ACTIONS = _registry()
BY_NAME = MappingProxyType({action.name: action for action in ACTIONS})


def action_mask(context: ControlContext) -> tuple[bool, ...]:
    result = []
    for action in ACTIONS:
        allowed = context.controllable or action.id == 0
        # If ground/air is unknown, do not promise a contextual move will occur.
        if action.context == "ground":
            allowed &= context.grounded is True
        elif action.context == "air":
            allowed &= context.grounded is False
        if action.possession is not None and context.armed is not None:
            allowed &= action.possession == context.armed
        if action.gravity_cancel or (action.button == "e" and action.context == "air"):
            allowed &= context.dodge_ready is not False
        result.append(bool(allowed))
    return tuple(result)


def contract() -> dict:
    payload = {"version": VERSION, "action_dim": len(ACTIONS),
               "actions": [asdict(action) for action in ACTIONS]}
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def require_contract(metadata: dict) -> None:
    # A matching dimension alone does not establish matching action meanings.
    if metadata != contract():
        raise ValueError("Incompatible semantic action contract; do not relabel legacy checkpoints/demos")
