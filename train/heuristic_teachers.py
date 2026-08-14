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
    recovery_dodge_ready_threshold: float = 0.05  # Zero-ish cooldown means the air dodge is available.
    recovery_heavy_cooldown_steps: int = 10  # Space recovery inputs so a held heavy does not dominate demos.
    recovery_setup_jump_distance: float = 0.035  # Jump near the ledge so setup reliably creates an offstage start.
    dash_distance: float = 0.20  # Reserve ground dashes for targets well beyond walking range.
    dash_cooldown_steps: int = 12  # Space dash inputs so one approach does not become dodge spam.
    pickup_retry_cooldown_steps: int = 8  # Retry a missed grab without pressing PICKUP every frame.
    combat_threat_animation_delta: float = 0.04  # Treat a visible silhouette change as attack startup.
    combat_threat_closing_speed: float = 0.04  # Defend when an in-range opponent is closing quickly.
    combat_defense_interval: int = 5  # Mix one dodge and one retreat into each threat-response cycle.


@dataclass
class HeuristicState:
    """Episode-local cadence state for actions that must not be spammed."""

    step: int = 0
    last_dash_step: int = -1_000_000
    pickup_in_range: bool = False
    last_pickup_step: int = -1_000_000
    combat_threat_steps: int = 0
    last_recovery_heavy_step: int = -1_000_000


class HeuristicTeacher:
    """Stateful facade for one collection episode."""

    def __init__(self, config: HeuristicConfig | None = None) -> None:
        self.config = config or HeuristicConfig()
        self.state = HeuristicState()

    def reset(self) -> None:
        self.state = HeuristicState()

    def action(
        self,
        phase: str,
        obs: np.ndarray,
        info: Mapping[str, Any] | None = None,
    ) -> int:
        action = _heuristic_action(phase, obs, info, self.config, self.state)
        self.state.step += 1
        return action


@dataclass
class RecoverySetupController:
    """Create an offstage start before recovery recording is armed.

    Setup transitions are deliberately discarded. Keeping this controller separate
    prevents its outward inputs from becoming recovery labels and lets the recovery
    teacher begin with clean episode-local cadence state.
    """

    config: HeuristicConfig = HeuristicConfig()
    outward_hdir: int = 0
    crossing_committed: bool = False

    def reset(self) -> None:
        self.outward_hdir = 0
        self.crossing_committed = False

    def action(self, obs: np.ndarray) -> int:
        base_obs = _base_obs(obs)
        dx_to_ledge = StateSpec.get(base_obs, "signed_dx_to_ledge")
        measured_hdir = _horizontal(dx_to_ledge, self.config.x_deadband)
        center_hdir = _horizontal(
            StateSpec.get(base_obs, "signed_dx_to_stage_center"),
            self.config.x_deadband,
        )
        if center_hdir != 0:
            # This is recomputed every frame because canonicalisation can mirror
            # when the opponent crosses sides. Opposite stage-centre remains the
            # same physical outward direction after canonical action inversion.
            self.outward_hdir = -center_hdir
        elif measured_hdir != 0 and not self.crossing_committed:
            self.outward_hdir = measured_hdir
        elif self.outward_hdir == 0:
            # At the exact ledge, the ledge offset loses its sign. Stage centre
            # still points inward, so its opposite is the outward crossing input.
            self.outward_hdir = -center_hdir if center_hdir != 0 else +1

        grounded = StateSpec.get(base_obs, "player_grounded") > 0.5
        near_ledge = abs(dx_to_ledge) <= self.config.recovery_setup_jump_distance
        if grounded and near_ledge and not self.crossing_committed:
            # Commit before leaving the platform: after crossing, the vector back
            # to the same ledge reverses sign even though outward momentum should
            # continue until the collector observes a valid offstage state.
            self.crossing_committed = True
            return int(_JUMP[self.outward_hdir])
        return int(_MOVE[self.outward_hdir])


def heuristic_action(
    phase: str,
    obs: np.ndarray,
    info: Mapping[str, Any] | None = None,
    config: HeuristicConfig | None = None,
) -> int:
    """Return a one-shot action; collectors should retain a ``HeuristicTeacher``."""
    return HeuristicTeacher(config).action(phase, obs, info)


def _heuristic_action(
    phase: str,
    obs: np.ndarray,
    info: Mapping[str, Any] | None,
    cfg: HeuristicConfig,
    state: HeuristicState,
) -> int:
    phase_key = _resolve_phase_key(phase, info)
    base_obs = _base_obs(obs)
    goal = _goal_target(obs, info)

    if phase_key == "recovery":
        return _recovery_action(base_obs, cfg, state)
    if phase_key == "movement":
        return _movement_action(base_obs, goal, cfg, state)
    if phase_key == "weapon_acquisition":
        return _weapon_action(base_obs, cfg, state)
    if phase_key == "spacing":
        return _spacing_action(base_obs, goal, cfg, state)
    if phase_key == "combat":
        return _combat_action(base_obs, cfg, state)

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
_DODGE = {0: Action.DODGE_SPOT, +1: Action.DODGE_TOWARD, -1: Action.DODGE_AWAY}
_DODGE_UP = {0: Action.DODGE_UP, +1: Action.DODGE_UP_TOWARD, -1: Action.DODGE_UP_AWAY}


def _navigate(hdir: int, want_up: bool, want_down: bool, obs, cfg) -> int:
    """Compose a locomotion action from a direction and a vertical intent."""
    if want_up and _can_jump(obs, cfg):
        return int(_JUMP[hdir])
    if want_down:
        return int(_FALL[hdir])
    return int(_MOVE[hdir])


# ── per-phase teachers ──────────────────────────────────────────────────────

def _dash_ready(state: HeuristicState, cfg: HeuristicConfig) -> bool:
    return state.step - state.last_dash_step >= max(1, int(cfg.dash_cooldown_steps))


def _movement_action(
    obs: np.ndarray,
    goal: np.ndarray,
    cfg: HeuristicConfig,
    state: HeuristicState,
) -> int:
    player_x = StateSpec.get(obs, "player_x")
    player_y = StateSpec.get(obs, "player_y")
    target_x = float(goal[GOAL_INDEX["player_x"]])
    target_y = float(goal[GOAL_INDEX["player_y"]])

    hdir = _horizontal(target_x - player_x, cfg.x_deadband)
    want_up = target_y < player_y - cfg.y_deadband
    want_down = target_y > player_y + cfg.y_deadband
    grounded = StateSpec.get(obs, "player_grounded") > 0.5
    if (
        grounded
        and hdir != 0
        and abs(target_x - player_x) >= cfg.dash_distance
        and _dash_ready(state, cfg)
    ):
        state.last_dash_step = state.step
        return int(_DODGE[hdir])
    return _navigate(hdir, want_up, want_down, obs, cfg)


def _recovery_action(
    obs: np.ndarray,
    cfg: HeuristicConfig,
    state: HeuristicState,
) -> int:
    dx_to_ledge = StateSpec.get(obs, "signed_dx_to_ledge")
    dy_to_ledge = StateSpec.get(obs, "dy_to_ledge")
    offstage = StateSpec.get(obs, "player_is_offstage") > 0.5

    hdir = _horizontal(dx_to_ledge, cfg.x_deadband)
    out_of_jumps = StateSpec.get(obs, "player_jumps_norm") <= cfg.min_jumps_norm
    if offstage and out_of_jumps:
        ledge_is_above = dy_to_ledge > cfg.y_deadband
        dodge_ready = (
            StateSpec.get(obs, "dodge_cooldown_norm")
            <= cfg.recovery_dodge_ready_threshold
        )
        if dodge_ready:
            return int((_DODGE_UP if ledge_is_above else _DODGE)[hdir])

        heavy_ready = (
            state.step - state.last_recovery_heavy_step
            >= max(1, int(cfg.recovery_heavy_cooldown_steps))
        )
        if heavy_ready:
            state.last_recovery_heavy_step = state.step
            # Recovery is the remaining vertical resource after jumps and air
            # dodge are gone. Requiring the ledge to already be above suppressed
            # it from shallow horizontal edge returns, even though those states
            # are precisely where the dedicated recovery move should be taught.
            # The recovery profile deliberately excludes HEAVY_AWAY: when the
            # ledge is not on canonical TOWARD, neutral recovery gains height
            # without steering the agent farther from stage.
            return int(Action.HEAVY_TOWARD if hdir > 0 else Action.HEAVY_NEUTRAL)

        # Every vertical resource is spent or cooling down. Drift horizontally --
        # but never NOOP, which `_MOVE[0]` would give inside the deadband. The
        # deadband exists to stop direction chatter while walking on the ground; in
        # the air it produces a fighter that stops steering and falls straight past
        # the ledge. Off the stage, any horizontal component beats none, so fall
        # back to the raw sign, and to canonical TOWARD when the offset is exactly
        # zero (with no opponent, TOWARD points at stage centre).
        if hdir == 0:
            drift = _horizontal(dx_to_ledge, 0.0)
            return int(_MOVE[drift] if drift != 0 else Action.MOVE_TOWARD)
        return int(_MOVE[hdir])
    want_up = offstage or dy_to_ledge > cfg.y_deadband
    return _navigate(hdir, want_up, False, obs, cfg)


def _weapon_action(obs: np.ndarray, cfg: HeuristicConfig, state: HeuristicState) -> int:
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
        entering_range = not state.pickup_in_range
        retry_ready = (
            state.step - state.last_pickup_step
            >= max(1, int(cfg.pickup_retry_cooldown_steps))
        )
        state.pickup_in_range = True
        if entering_range or retry_ready:
            state.last_pickup_step = state.step
            return int(Action.PICKUP)
        return _hold_position(obs, cfg)

    state.pickup_in_range = False
    hdir = _horizontal(weapon_dx, cfg.x_deadband)
    return _navigate(hdir, weapon_dy < -cfg.y_deadband, False, obs, cfg)


def _hold_position(obs: np.ndarray, cfg: HeuristicConfig) -> int:
    """Stay put while armed without teaching a freeze reflex."""
    if StateSpec.get(obs, "player_is_offstage") <= 0.5:
        center_dx = StateSpec.get(obs, "signed_dx_to_stage_center")
        hdir = _horizontal(center_dx, cfg.x_deadband)
        if hdir == 0:
            # A small correction at centre is preferable to making NOOP one third
            # of the successful weapon archive. Reverse current drift when possible.
            hdir = -1 if StateSpec.get(obs, "player_vx") > 0.0 else +1
        return int(_MOVE[hdir])

    dx_to_ledge = StateSpec.get(obs, "signed_dx_to_ledge")
    hdir = _horizontal(dx_to_ledge, cfg.x_deadband)
    return _navigate(hdir, StateSpec.get(obs, "dy_to_ledge") > cfg.y_deadband, False, obs, cfg)


def _spacing_action(
    obs: np.ndarray,
    goal: np.ndarray,
    cfg: HeuristicConfig,
    state: HeuristicState,
) -> int:
    rel_dy = StateSpec.get(obs, "rel_dy")
    current_distance = StateSpec.get(obs, "rel_distance")
    target_distance = float(goal[GOAL_INDEX["rel_distance"]]) * 2.0
    target_dy = (float(goal[GOAL_INDEX["rel_dy"]]) * 2.0) - 1.0

    distance_error = current_distance - target_distance
    grounded = StateSpec.get(obs, "player_grounded") > 0.5
    if distance_error < -2.0 * cfg.spacing_deadband and grounded and _dash_ready(state, cfg):
        state.last_dash_step = state.step
        return int(Action.DODGE_AWAY)
    if abs(distance_error) <= cfg.spacing_deadband:
        hdir = 0
    else:
        # Canonical space: the opponent is always toward +1, so closing the gap is
        # always TOWARD and opening it is always AWAY. No sign juggling.
        hdir = +1 if distance_error > 0.0 else -1

    vertical_error = target_dy - rel_dy
    return _navigate(hdir, vertical_error > cfg.y_deadband, vertical_error < -cfg.y_deadband, obs, cfg)


def _combat_action(obs: np.ndarray, cfg: HeuristicConfig, state: HeuristicState) -> int:
    """Approach, then commit inside range.

    Facing needs no feature: canonicalisation puts the opponent at positive rel_dx
    always, so "toward" is a single fixed input. The old `facing_opponent` gate was
    derived from the agent's own previous input, making it circular.
    """
    rel_dx = StateSpec.get(obs, "rel_dx")
    rel_dy = StateSpec.get(obs, "rel_dy")
    abs_dx, abs_dy = abs(rel_dx), abs(rel_dy)

    in_strike_range = StateSpec.get(obs, "in_strike_range") > 0.5
    opponent_exists = StateSpec.get(obs, "opponent_exists") > 0.5
    animation_delta = abs(StateSpec.get(obs, "opponent_dw")) + abs(
        StateSpec.get(obs, "opponent_dh")
    )
    opponent_closing = StateSpec.get(obs, "rel_vx") < -cfg.combat_threat_closing_speed
    opponent_is_threat = bool(
        opponent_exists
        and in_strike_range
        and (
            animation_delta >= cfg.combat_threat_animation_delta
            or opponent_closing
        )
    )
    if opponent_is_threat:
        interval = max(3, int(cfg.combat_defense_interval))
        slot = state.combat_threat_steps % interval
        defense_cycle = state.combat_threat_steps // interval
        state.combat_threat_steps += 1
        if slot == 0:
            return int(Action.DODGE_AWAY if defense_cycle % 2 == 0 else Action.DODGE_SPOT)
        if slot == 1:
            return int(Action.MOVE_AWAY)

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


__all__ = [
    "HeuristicConfig",
    "HeuristicState",
    "HeuristicTeacher",
    "heuristic_action",
]
