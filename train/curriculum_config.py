from __future__ import annotations

from typing import Callable

import numpy as np

from config import PLATFORM_BOUNDS
from feature_extractor.memory.state_spec import StateSpec
from train.curriculum_goals import (
    CURRICULUM_GOAL_FEATURES,
    GOAL_DIM,
    GOAL_INDEX,
    clip_goal_target,
    default_goal_target,
    extract_curriculum_goal_features,
)
from train.llc_stage_common import StageSpec


TargetSampler = Callable[[np.ndarray], np.ndarray]

PHASES = (
    "locomotion_grounded",
    "locomotion_airborne",
    "locomotion_recovery",
    "locomotion",
    "weapon_control",
    "damage_static",
    "damage_dynamic",
)

LOCO_PLATFORM_X_MARGIN = 0.03
LOCO_GROUNDED_Y_EPS = 0.015
LOCO_AIRBORNE_Y_DELTA = 0.08
LOCO_RECOVERY_OUTSIDE_PROB = 0.40
LOCO_RECOVERY_OUTSIDE_BAND = 0.10
LOCO_RECOVERY_SIDE_OFFSET = 0.01
LOCO_RECOVERY_DOWN_SHIFT = 0.12


def _base_target() -> np.ndarray:
    return default_goal_target()


def _sample_platform_x(margin: float = LOCO_PLATFORM_X_MARGIN) -> float:
    x_min = float(PLATFORM_BOUNDS["x_min"]) + float(max(0.0, margin))
    x_max = float(PLATFORM_BOUNDS["x_max"]) - float(max(0.0, margin))
    if x_max <= x_min:
        x_min = float(PLATFORM_BOUNDS["x_min"])
        x_max = float(PLATFORM_BOUNDS["x_max"])
    return float(np.random.uniform(x_min, x_max))


def _sample_inside_platform_xy() -> tuple[float, float]:
    x = _sample_platform_x()
    y_floor = float(PLATFORM_BOUNDS["y_min"])
    y = float(np.clip(y_floor + LOCO_GROUNDED_Y_EPS, 0.0, 1.0))
    return x, y


def _sample_airborne_platform_xy() -> tuple[float, float]:
    x = _sample_platform_x()
    y_center = float(PLATFORM_BOUNDS["y_min"])
    y = float(np.random.uniform(y_center - LOCO_AIRBORNE_Y_DELTA, y_center + LOCO_AIRBORNE_Y_DELTA))
    y = float(np.clip(y, 0.0, 1.0))
    return x, y


def _sample_outside_platform_xy(side: str) -> tuple[float, float]:
    x_min = float(PLATFORM_BOUNDS["x_min"])
    x_max = float(PLATFORM_BOUNDS["x_max"])
    y_min = float(PLATFORM_BOUNDS["y_min"])

    band = float(max(0.02, LOCO_RECOVERY_OUTSIDE_BAND))
    offset = float(max(0.005, LOCO_RECOVERY_SIDE_OFFSET))

    if side == "left":
        lo = max(0.0, x_min - band)
        hi = max(lo + 1e-3, x_min - offset)
    else:
        lo = min(1.0, x_max + offset)
        hi = min(1.0, x_max + band)
        if hi <= lo:
            hi = min(1.0, lo + 1e-3)

    x = float(np.random.uniform(lo, hi))
    y_lo = float(np.clip(y_min + 0.02, 0.0, 1.0))
    y_hi = float(np.clip(y_min + LOCO_RECOVERY_DOWN_SHIFT, y_lo + 1e-3, 1.0))
    y = float(np.random.uniform(y_lo, y_hi))
    return x, y


def _sampler_locomotion_grounded(_: np.ndarray) -> np.ndarray:
    t = _base_target()
    tx, ty = _sample_inside_platform_xy()
    t[GOAL_INDEX["player_x"]] = tx
    t[GOAL_INDEX["player_y"]] = ty
    return clip_goal_target(t)


def _sampler_locomotion_airborne(_: np.ndarray) -> np.ndarray:
    t = _base_target()
    tx, ty = _sample_airborne_platform_xy()
    t[GOAL_INDEX["player_x"]] = tx
    t[GOAL_INDEX["player_y"]] = ty
    return clip_goal_target(t)


def _sampler_locomotion_recovery(obs: np.ndarray) -> np.ndarray:
    t = _base_target()

    offstage = False
    try:
        offstage = float(StateSpec.get(np.asarray(obs, dtype=np.float32), "player_is_offstage")) > 0.5
    except Exception:
        offstage = False

    if offstage:
        tx, ty = _sample_inside_platform_xy()
    else:
        if float(np.random.rand()) < LOCO_RECOVERY_OUTSIDE_PROB:
            side = "left" if float(np.random.rand()) < 0.5 else "right"
            tx, ty = _sample_outside_platform_xy(side)
        else:
            tx, ty = _sample_inside_platform_xy()

    t[GOAL_INDEX["player_x"]] = tx
    t[GOAL_INDEX["player_y"]] = ty
    return clip_goal_target(t)


def _sampler_weapon(_: np.ndarray) -> np.ndarray:
    t = _base_target()
    t[GOAL_INDEX["player_has_weapon"]] = 1.0
    t[GOAL_INDEX["weapon_dx"]] = 0.5
    t[GOAL_INDEX["weapon_dy"]] = 0.5
    t[GOAL_INDEX["player_is_offstage"]] = 0.0
    return clip_goal_target(t)


def _sampler_damage_static(_: np.ndarray) -> np.ndarray:
    t = _base_target()
    t[GOAL_INDEX["player_has_weapon"]] = 1.0
    t[GOAL_INDEX["in_strike_range"]] = 1.0
    t[GOAL_INDEX["rel_distance"]] = float(np.random.uniform(0.10, 0.24))
    t[GOAL_INDEX["opponent_damage_pct"]] = float(np.random.uniform(0.60, 1.00))
    t[GOAL_INDEX["player_is_offstage"]] = 0.0
    return clip_goal_target(t)


def _sampler_damage_dynamic(_: np.ndarray) -> np.ndarray:
    t = _base_target()
    t[GOAL_INDEX["player_has_weapon"]] = 1.0
    t[GOAL_INDEX["in_strike_range"]] = 1.0
    t[GOAL_INDEX["rel_distance"]] = float(np.random.uniform(0.10, 0.28))
    t[GOAL_INDEX["frame_advantage_estimate"]] = float(np.random.uniform(0.60, 1.00))
    t[GOAL_INDEX["opponent_damage_pct"]] = float(np.random.uniform(0.60, 1.00))
    t[GOAL_INDEX["player_is_offstage"]] = 0.0
    return clip_goal_target(t)


def _mask_for(*active: tuple[str, float]) -> np.ndarray:
    mask = np.zeros((GOAL_DIM,), dtype=np.float32)
    for name, weight in active:
        mask[GOAL_INDEX[name]] = float(np.clip(weight, 0.0, 1.0))
    return mask


def build_phase_spec(
    phase: str,
    death_penalty: float = 1.0,
    terminate_on_death: bool = True,
) -> StageSpec:
    phase = phase.strip().lower()

    if phase == "locomotion":
        phase = "locomotion_grounded"

    if phase == "locomotion_grounded":
        return StageSpec(
            stage_id=1,
            name="phase1_locomotion_grounded",
            mask=_mask_for(("player_x", 1.0), ("player_y", 1.0)),
            target_sampler=_sampler_locomotion_grounded,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            min_goal_duration=20,
            max_goal_duration=40,
            success_threshold=0.12,
            success_bonus=1.2,
            proximity_scale=0.0,
            use_l2_error=True,
            vertical_velocity_penalty_scale=0.10,
            death_penalty=float(death_penalty),
            reward_clip=3.0,
            disable_attack=True,
            disable_dodge=True,
            disable_jump=True,
            reset_perturb_steps=0,
            idle_movement_penalty=0.01,
            idle_action_index=3,
            terminate_on_death=bool(terminate_on_death),
            terminate_on_goal_success=True,
            resample_goal_on_timer=False,
        )

    if phase == "locomotion_airborne":
        return StageSpec(
            stage_id=2,
            name="phase2_locomotion_airborne",
            mask=_mask_for(("player_x", 1.0), ("player_y", 1.0)),
            target_sampler=_sampler_locomotion_airborne,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            min_goal_duration=20,
            max_goal_duration=40,
            success_threshold=0.10,
            success_bonus=1.3,
            proximity_scale=0.0,
            use_l2_error=True,
            jump_usage_penalty_scale=0.05,
            velocity_penalty_scale=0.02,
            velocity_penalty_radius=1.5,
            death_penalty=float(death_penalty),
            reward_clip=3.0,
            disable_attack=True,
            disable_dodge=True,
            disable_jump=False,
            reset_perturb_steps=0,
            idle_movement_penalty=0.0,
            terminate_on_death=bool(terminate_on_death),
            terminate_on_goal_success=True,
            resample_goal_on_timer=False,
        )

    if phase == "locomotion_recovery":
        return StageSpec(
            stage_id=3,
            name="phase3_locomotion_recovery",
            mask=_mask_for(("player_x", 1.0), ("player_y", 1.0)),
            target_sampler=_sampler_locomotion_recovery,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            min_goal_duration=20,
            max_goal_duration=40,
            success_threshold=0.11,
            success_bonus=1.4,
            proximity_scale=0.0,
            use_l2_error=True,
            jump_usage_penalty_scale=0.03,
            velocity_penalty_scale=0.02,
            velocity_penalty_radius=1.5,
            death_penalty=float(death_penalty),
            reward_clip=3.2,
            disable_attack=True,
            disable_dodge=False,
            disable_jump=False,
            reset_perturb_steps=0,
            idle_movement_penalty=0.005,
            idle_action_index=3,
            terminate_on_death=bool(terminate_on_death),
            terminate_on_goal_success=True,
            resample_goal_on_timer=False,
        )

    if phase == "weapon_control":
        return StageSpec(
            stage_id=4,
            name="phase4_weapon_control",
            mask=_mask_for(
                ("player_has_weapon", 1.0),
                ("weapon_dx", 0.4),
                ("weapon_dy", 0.4),
                ("player_is_offstage", 0.6),
            ),
            target_sampler=_sampler_weapon,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            min_goal_duration=20,
            max_goal_duration=40,
            progress_scale=1.8,
            progress_clip_min=-0.2,
            progress_clip_max=0.7,
            success_threshold=0.16,
            success_bonus=1.2,
            proximity_scale=0.8,
            death_penalty=float(death_penalty),
            reward_clip=2.2,
            disable_attack=True,
            disable_dodge=False,
            disable_jump=False,
            reset_perturb_steps=4,
            terminate_on_death=bool(terminate_on_death),
            terminate_on_goal_success=True,
            resample_goal_on_timer=False,
        )

    if phase == "damage_static":
        return StageSpec(
            stage_id=5,
            name="phase5_damage_static",
            mask=_mask_for(
                ("player_has_weapon", 0.4),
                ("in_strike_range", 1.0),
                ("rel_distance", 0.8),
                ("opponent_damage_pct", 1.0),
                ("player_is_offstage", 0.7),
            ),
            target_sampler=_sampler_damage_static,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            min_goal_duration=16,
            max_goal_duration=30,
            progress_scale=2.0,
            progress_clip_min=-0.25,
            progress_clip_max=0.8,
            success_threshold=0.20,
            success_bonus=1.5,
            proximity_scale=0.4,
            death_penalty=float(death_penalty),
            reward_clip=2.5,
            disable_attack=False,
            disable_dodge=False,
            disable_jump=False,
            reset_perturb_steps=0,
            terminate_on_death=bool(terminate_on_death),
            terminate_on_goal_success=True,
            resample_goal_on_timer=False,
        )

    if phase == "damage_dynamic":
        return StageSpec(
            stage_id=6,
            name="phase6_damage_dynamic",
            mask=_mask_for(
                ("player_has_weapon", 0.3),
                ("in_strike_range", 1.0),
                ("rel_distance", 0.8),
                ("frame_advantage_estimate", 0.9),
                ("opponent_damage_pct", 1.0),
                ("player_is_offstage", 0.8),
            ),
            target_sampler=_sampler_damage_dynamic,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            min_goal_duration=12,
            max_goal_duration=24,
            progress_scale=2.2,
            progress_clip_min=-0.25,
            progress_clip_max=0.9,
            success_threshold=0.22,
            success_bonus=1.7,
            proximity_scale=0.35,
            death_penalty=float(death_penalty),
            reward_clip=2.8,
            disable_attack=False,
            disable_dodge=False,
            disable_jump=False,
            reset_perturb_steps=0,
            terminate_on_death=bool(terminate_on_death),
            terminate_on_goal_success=True,
            resample_goal_on_timer=False,
        )

    raise ValueError(f"Unknown phase '{phase}'. Expected one of: {', '.join(PHASES)}")
