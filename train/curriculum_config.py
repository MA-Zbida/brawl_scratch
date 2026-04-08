from __future__ import annotations

from typing import Callable

import numpy as np

from config import LOCOMOTION_GREEN_BOUNDS, LOCOMOTION_RED_EXCLUSION_MARGIN, PLATFORM_BOUNDS
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
    "locomotion",
    "weapon_control",
    "damage_static",
    "damage_dynamic",
)


def _base_target() -> np.ndarray:
    return default_goal_target()


def _sample_feasible_locomotion_xy() -> tuple[float, float]:
    gx_min = float(LOCOMOTION_GREEN_BOUNDS["x_min"])
    gx_max = float(LOCOMOTION_GREEN_BOUNDS["x_max"])
    gy_min = float(LOCOMOTION_GREEN_BOUNDS["y_min"])
    gy_max = float(LOCOMOTION_GREEN_BOUNDS["y_max"])

    rx_min = float(PLATFORM_BOUNDS["x_min"])
    rx_max = float(PLATFORM_BOUNDS["x_max"])
    ry_min = float(PLATFORM_BOUNDS["y_min"])
    ry_max = float(PLATFORM_BOUNDS["y_max"])

    margin = float(max(0.0, LOCOMOTION_RED_EXCLUSION_MARGIN))
    rx_min -= margin
    rx_max += margin
    ry_min -= margin
    ry_max += margin

    if not (gx_min < gx_max and gy_min < gy_max):
        raise ValueError("Invalid LOCOMOTION_GREEN_BOUNDS in config.py")

    # A valid target must be inside the green rectangle and outside the red hole.
    for _ in range(256):
        x = float(np.random.uniform(gx_min, gx_max))
        y = float(np.random.uniform(gy_min, gy_max))
        in_red = (rx_min <= x <= rx_max) and (ry_min <= y <= ry_max)
        if not in_red:
            return x, y

    # Deterministic fallback: left-side corridor, guaranteed outside red hole.
    y_mid = float(np.clip((gy_min + gy_max) * 0.5, gy_min, gy_max))
    x_left = float(np.clip(rx_min - max(0.02, margin), gx_min, gx_max))
    return x_left, y_mid


def _sampler_locomotion(_: np.ndarray) -> np.ndarray:
    t = _base_target()
    tx, ty = _sample_feasible_locomotion_xy()
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
        return StageSpec(
            stage_id=1,
            name="phase1_locomotion",
            mask=_mask_for(("player_x", 1.0), ("player_y", 1.0)),
            target_sampler=_sampler_locomotion,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            min_goal_duration=20,
            max_goal_duration=40,
            progress_scale=2.0,
            progress_clip_min=-10.0,
            progress_clip_max=10.0,
            clip_progress_reward=False,
            success_threshold=0.1,
            success_bonus=2.0,
            proximity_scale=0.0,
            death_penalty=float(death_penalty),
            reward_clip=10.0,
            disable_attack=True,
            disable_dodge=False,
            disable_jump=False,
            reset_perturb_steps=0,
            idle_movement_penalty=0.01,
            idle_action_index=3,
            terminate_on_death=bool(terminate_on_death),
            terminate_on_goal_success=True,
            resample_goal_on_timer=False,
        )

    if phase == "weapon_control":
        return StageSpec(
            stage_id=2,
            name="phase2_weapon_control",
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
            stage_id=3,
            name="phase3_damage_static",
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
            stage_id=4,
            name="phase4_damage_dynamic",
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
