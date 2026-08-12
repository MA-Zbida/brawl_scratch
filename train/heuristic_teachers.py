"""Scripted teachers over the 27-action space.

These are reactive controllers on the same observation the policy receives, so
behaviour cloning against them distils a hand-written function rather than
transferring knowledge the network could not otherwise obtain. Their value is
removing the initial random-walk phase, not solving exploration.

Everything is expressed in **canonical** space: TOWARD/AWAY, never left/right. The
observation is horizontally canonicalised, so a teacher written this way is
automatically correct on both sides of the stage and the demonstrations it produces
are mirror-invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from action_space import Action
from feature_extractor.memory.state_spec import StateSpec
from train.curriculum_goals import GOAL_DIM, GOAL_INDEX


@dataclass(frozen=True)
class HeuristicConfig:
    x_deadband: float = 0.015
    y_deadband: float = 0.025
    spacing_deadband: float = 0.035
    pickup_distance: float = 0.05
    combat_light_range: float = 0.055
    combat_heavy_range: float = 0.075
    combat_vertical_tolerance: float = 0.04
    min_jumps_norm: float = 0.05
    rising_speed: float = 0.02


def heuristic_action(
    phase: str,
    obs: np.ndarray,
    info: Mapping[str, Any] | None = None,
    config: HeuristicConfig | None = None,
) -> int:
    """Return an action index from the 27-action space."""
    cfg = config or HeuristicConfig()
    phase_key = _resolve_phase_key(phase, info)
    base_obs = _base_obs(obs)
    goal = _goal_target(obs, info)

    if phase_key == "recovery":
        return _recovery_action(base_obs, cfg)
    if phase_key == "movement":
        return _movement_action(base_obs, goal, cfg)
    if phase_key == "weapon_acquisition":
        return _weapon_action(base_obs, cfg)
    if phase_key == "spacing":
        return _spacing_action(base_obs, goal, cfg)
    if phase_key == "combat":
        return _combat_action(base_obs, cfg)

    return int(Action.NOOP)


def _resolve_phase_key(phase: str, info: Mapping[str, Any] | None) -> str:
    if info is not None:
        goal_type = str(info.get("goal_type", "")).strip().lower()
        if goal_type:
            return goal_type

    key = str(phase).strip().lower()
    return {
        "recovery_mastery": "recovery",
        "movement_fluency": "movement",
        "weapon_acquisition": "weapon_acquisition",
        "spacing_neutral": "spacing",
        "combat_execution": "combat",
    }.get(key, key)


def _base_obs(obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if arr.shape[0] < StateSpec.dim():
        raise ValueError(f"Expected at least {StateSpec.dim()} observation values, got {arr.shape[0]}")
    return arr[: StateSpec.dim()]


def _goal_target(obs: np.ndarray, info: Mapping[str, Any] | None) -> np.ndarray:
    if info is not None and "goal_target" in info:
        target = np.asarray(info["goal_target"], dtype=np.float32).reshape(-1)
        if target.shape[0] >= GOAL_DIM:
            return target[:GOAL_DIM]

    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    start = StateSpec.dim()
    stop = start + GOAL_DIM
    if arr.shape[0] >= stop:
        return arr[start:stop]

    return np.full((GOAL_DIM,), 0.5, dtype=np.float32)


# ── direction helpers, in canonical space ───────────────────────────────────

def _horizontal(dx: float, deadband: float) -> int:
    """+1 toward, -1 away, 0 neutral, for a canonical-frame offset."""
    if dx > deadband:
        return +1
    if dx < -deadband:
        return -1
    return 0


def _can_jump(obs: np.ndarray, cfg: HeuristicConfig) -> bool:
    """Spend a jump only when one remains and we are not already rising.

    Requesting a jump every step burns all three in three frames and then does
    nothing, so airborne goals were never reached -- and because only successful
    episodes are saved, they vanished from the dataset rather than showing up as
    failures.
    """
    if StateSpec.get(obs, "player_jumps_norm") <= cfg.min_jumps_norm:
        return False
    grounded = StateSpec.get(obs, "player_grounded") > 0.5
    rising = StateSpec.get(obs, "player_vy") < -cfg.rising_speed
    return grounded or not rising


_MOVE = {0: Action.NOOP, +1: Action.MOVE_TOWARD, -1: Action.MOVE_AWAY}
_JUMP = {0: Action.JUMP, +1: Action.JUMP_TOWARD, -1: Action.JUMP_AWAY}
_FALL = {0: Action.FAST_FALL, +1: Action.FAST_FALL_TOWARD, -1: Action.FAST_FALL_AWAY}
_LIGHT = {0: Action.LIGHT_NEUTRAL, +1: Action.LIGHT_TOWARD, -1: Action.LIGHT_AWAY}
_HEAVY = {0: Action.HEAVY_NEUTRAL, +1: Action.HEAVY_TOWARD, -1: Action.HEAVY_AWAY}


def _navigate(hdir: int, want_up: bool, want_down: bool, obs, cfg) -> int:
    """Compose a locomotion action from a direction and a vertical intent."""
    if want_up and _can_jump(obs, cfg):
        return int(_JUMP[hdir])
    if want_down:
        return int(_FALL[hdir])
    return int(_MOVE[hdir])


# ── per-phase teachers ──────────────────────────────────────────────────────

def _movement_action(obs: np.ndarray, goal: np.ndarray, cfg: HeuristicConfig) -> int:
    player_x = StateSpec.get(obs, "player_x")
    player_y = StateSpec.get(obs, "player_y")
    target_x = float(goal[GOAL_INDEX["player_x"]])
    target_y = float(goal[GOAL_INDEX["player_y"]])

    hdir = _horizontal(target_x - player_x, cfg.x_deadband)
    want_up = target_y < player_y - cfg.y_deadband
    want_down = target_y > player_y + cfg.y_deadband
    return _navigate(hdir, want_up, want_down, obs, cfg)


def _recovery_action(obs: np.ndarray, cfg: HeuristicConfig) -> int:
    dx_to_ledge = StateSpec.get(obs, "signed_dx_to_ledge")
    dy_to_ledge = StateSpec.get(obs, "dy_to_ledge")
    offstage = StateSpec.get(obs, "player_is_offstage") > 0.5

    hdir = _horizontal(dx_to_ledge, cfg.x_deadband)
    want_up = offstage or dy_to_ledge > cfg.y_deadband
    return _navigate(hdir, want_up, False, obs, cfg)


def _weapon_action(obs: np.ndarray, cfg: HeuristicConfig) -> int:
    has_weapon = StateSpec.get(obs, "player_has_weapon") > 0.5
    if has_weapon:
        # Hold position, but never emit a bare NOOP: only successful episodes are
        # saved, so an idle here becomes a behaviour-cloning anchor that teaches the
        # agent to freeze for the whole hold window after every pickup.
        return _hold_position(obs, cfg)

    weapon_dx = StateSpec.get(obs, "weapon_dx")
    weapon_dy = StateSpec.get(obs, "weapon_dy")
    distance = float(np.hypot(weapon_dx, weapon_dy))
    if distance <= cfg.pickup_distance:
        return int(Action.PICKUP)

    hdir = _horizontal(weapon_dx, cfg.x_deadband)
    return _navigate(hdir, weapon_dy < -cfg.y_deadband, False, obs, cfg)


def _hold_position(obs: np.ndarray, cfg: HeuristicConfig) -> int:
    """Stay put while armed without teaching a freeze reflex."""
    if StateSpec.get(obs, "player_is_offstage") <= 0.5:
        return int(Action.NOOP)

    dx_to_ledge = StateSpec.get(obs, "signed_dx_to_ledge")
    hdir = _horizontal(dx_to_ledge, cfg.x_deadband)
    return _navigate(hdir, StateSpec.get(obs, "dy_to_ledge") > cfg.y_deadband, False, obs, cfg)


def _spacing_action(obs: np.ndarray, goal: np.ndarray, cfg: HeuristicConfig) -> int:
    rel_dy = StateSpec.get(obs, "rel_dy")
    current_distance = StateSpec.get(obs, "rel_distance")
    target_distance = float(goal[GOAL_INDEX["rel_distance"]]) * 2.0
    target_dy = (float(goal[GOAL_INDEX["rel_dy"]]) * 2.0) - 1.0

    distance_error = current_distance - target_distance
    if abs(distance_error) <= cfg.spacing_deadband:
        hdir = 0
    else:
        # Canonical space: the opponent is always toward +1, so closing the gap is
        # always TOWARD and opening it is always AWAY. No sign juggling.
        hdir = +1 if distance_error > 0.0 else -1

    vertical_error = target_dy - rel_dy
    return _navigate(hdir, vertical_error > cfg.y_deadband, vertical_error < -cfg.y_deadband, obs, cfg)


def _combat_action(obs: np.ndarray, cfg: HeuristicConfig) -> int:
    """Approach, then commit inside range.

    Facing needs no feature: canonicalisation puts the opponent at positive rel_dx
    always, so "toward" is a single fixed input. The old `facing_opponent` gate was
    derived from the agent's own previous input, making it circular.
    """
    rel_dx = StateSpec.get(obs, "rel_dx")
    rel_dy = StateSpec.get(obs, "rel_dy")
    abs_dx, abs_dy = abs(rel_dx), abs(rel_dy)

    vertically_aligned = abs_dy <= cfg.combat_vertical_tolerance
    if abs_dx > cfg.combat_heavy_range:
        return _navigate(_horizontal(rel_dx, cfg.x_deadband),
                         rel_dy < -cfg.y_deadband and not vertically_aligned, False, obs, cfg)

    if not vertically_aligned:
        # In range horizontally but off the same plane: line up rather than whiff.
        if rel_dy < -cfg.y_deadband:
            return int(Action.LIGHT_NEUTRAL) if _can_jump(obs, cfg) is False else int(Action.JUMP)
        return int(Action.LIGHT_DOWN)

    if abs_dx <= cfg.combat_light_range:
        return int(_LIGHT[_horizontal(rel_dx, cfg.x_deadband)])
    return int(_HEAVY[_horizontal(rel_dx, cfg.x_deadband)])
