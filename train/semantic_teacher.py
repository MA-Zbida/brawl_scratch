"""A single goal-free teacher; collection drills only control discarded setup.

The same visible state always uses the same priority rules in every drill. There
is no hidden goal target stripped away when its action becomes a BC label.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from action_space import Action, components
from control.semantic_actions import BY_NAME
from feature_extractor.memory.state_spec import StateSpec
from train.curriculum_goals import GOAL_INDEX, default_goal_target
from train.heuristic_teachers import HeuristicTeacher
from train.semantic_contract import ACTION_DIM


def semantic_from_legacy(action: int, obs: np.ndarray) -> int:
    name = Action(action).name
    air = StateSpec.get(obs, "player_grounded") < .5
    if name.startswith("DODGE_"):
        suffix = name.removeprefix("DODGE_")
        if air:
            name = "AIR_DODGE_" + suffix
        else:
            hdir = components(action).hdir
            name = "DASH_TOWARD" if hdir > 0 else "DASH_AWAY" if hdir < 0 else "GROUND_SPOT_DODGE"
    elif name.startswith("LIGHT_"):
        name = ("AIR_LIGHT_" if air else "GROUND_LIGHT_") + name.removeprefix("LIGHT_")
    elif name.startswith("HEAVY_"):
        suffix = name.removeprefix("HEAVY_")
        name = ("GROUND_POUND_NEUTRAL" if suffix == "DOWN" else "RECOVERY_" + suffix) if air else "SIGNATURE_" + suffix
    elif name == "PICKUP" and StateSpec.get(obs, "player_has_weapon") > .5:
        name = "THROW_NEUTRAL"
    return BY_NAME[name].id


@dataclass(frozen=True)
class TeacherDecision:
    action: int
    family: str


class IntegratedTeacher:
    def __init__(self):
        self.teacher = HeuristicTeacher()

    def action(self, obs: np.ndarray) -> TeacherDecision:
        mask = obs[-ACTION_DIM:].astype(bool)
        if mask.sum() == 1:
            return TeacherDecision(int(np.flatnonzero(mask)[0]), "continuation")
        value = lambda name: StateSpec.get(obs, name)
        goal = default_goal_target()
        goal[GOAL_INDEX["rel_distance"]] = .12
        goal[GOAL_INDEX["rel_dy"]] = .5
        if value("player_is_offstage") > .5:
            phase, family = "recovery_mastery", "recovery"
        elif value("player_has_weapon") < .5 and value("weapon_on_ground") > .5:
            phase, family = "weapon_acquisition", "weapon"
        elif value("opponent_exists") < .5 or value("opponent_is_offstage") > .5:
            # Do not teach chasing an offstage opponent past the ledge.
            phase, family = "movement_fluency", "movement"
            goal[GOAL_INDEX["player_x"]] = np.clip(value("player_x") + value("signed_dx_to_stage_center"), 0., 1.)
            goal[GOAL_INDEX["player_y"]] = value("player_y")
        elif value("rel_distance") < .065 and value("player_grounded") > .5:
            phase, family = "spacing_neutral", "spacing"
        else:
            phase = "combat_execution"
            family = "movement" if value("rel_distance") > .18 else "combat"
        legacy = self.teacher.action(phase, obs, {"goal_target": goal})
        action = semantic_from_legacy(legacy, obs)
        if family == "recovery" and components(legacy).heavy:
            dx = value("signed_dx_to_ledge")
            name = "RECOVERY_TOWARD" if dx > 0 else "RECOVERY_AWAY" if dx < 0 else "RECOVERY_NEUTRAL"
            action = BY_NAME[name].id
        if not mask[action]:
            dx = value("signed_dx_to_stage_center")
            fallback = BY_NAME["MOVE_TOWARD" if dx >= 0 else "MOVE_AWAY"].id
            action = fallback if mask[fallback] else 0
        return TeacherDecision(action, family)
