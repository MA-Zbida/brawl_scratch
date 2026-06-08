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
    normalize_goal_type,
)
from train.llc_stage_common import StageSpec


TargetSampler = Callable[[np.ndarray], np.ndarray]

PHASES = (
    "recovery_mastery",
    "movement_fluency",
    "weapon_acquisition",
    "spacing_neutral",
    "combat_execution",
)

LOCO_PLATFORM_X_MARGIN = 0.03
LOCO_GROUNDED_Y_EPS = 0.015
LOCO_AIRBORNE_Y_DELTA = 0.28


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
    y = float(np.random.uniform(y_center - LOCO_AIRBORNE_Y_DELTA, y_center - LOCO_AIRBORNE_Y_DELTA / 4.0))
    y = float(np.clip(y, 0.0, 1.0))
    return x, y


def _sample_outside_platform_xy(side: str) -> tuple[float, float]:
    x_min = float(PLATFORM_BOUNDS["x_min"])
    x_max = float(PLATFORM_BOUNDS["x_max"])
    y_min = float(PLATFORM_BOUNDS["y_min"])

    band = float(max(0.02, LOCO_RECOVERY_STEP1_OUTSIDE_BAND))
    offset = float(max(0.005, LOCO_RECOVERY_STEP1_SIDE_OFFSET))

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
    y_hi = float(np.clip(y_min + LOCO_RECOVERY_STEP2_PLATFORM_Y_EPS + 0.10, y_lo + 1e-3, 1.0))
    y = float(np.random.uniform(y_lo, y_hi))
    return x, y


def _nearest_platform_side(player_x: float) -> str:
    x_min = float(PLATFORM_BOUNDS["x_min"])
    x_max = float(PLATFORM_BOUNDS["x_max"])
    d_left = abs(float(player_x) - x_min)
    d_right = abs(float(player_x) - x_max)
    return "left" if d_left <= d_right else "right"


# ─── Samplers ──────────────────────────────────────────────────────────────────

def _sampler_recovery(_: np.ndarray) -> np.ndarray:
    """Recovery goal: reach the nearest ledge (zero signed offset in both axes)."""
    t = _base_target()
    # 0.5 in normalised [0,1] == 0 in raw [-1,1] == "directly at the ledge"
    t[GOAL_INDEX["signed_dx_to_ledge"]] = 0.5
    t[GOAL_INDEX["dy_to_ledge"]] = 0.5
    return clip_goal_target(t)


def _sampler_movement(_: np.ndarray) -> np.ndarray:
    """Movement goal: reach a random position on or above the platform."""
    t = _base_target()
    if float(np.random.rand()) < 0.7:
        tx, ty = _sample_inside_platform_xy()
    else:
        tx, ty = _sample_airborne_platform_xy()
    t[GOAL_INDEX["player_x"]] = tx
    t[GOAL_INDEX["player_y"]] = ty
    return clip_goal_target(t)


def _sampler_weapon_acq(_: np.ndarray) -> np.ndarray:
    """Weapon goal: acquire and hold the weapon."""
    t = _base_target()
    t[GOAL_INDEX["player_has_weapon"]] = 1.0
    # 0.5 == zero offset in the normalised [-1,1]→[0,1] space
    t[GOAL_INDEX["weapon_dx"]] = 0.5
    t[GOAL_INDEX["weapon_dy"]] = 0.5
    return clip_goal_target(t)


def _sampler_spacing(_: np.ndarray) -> np.ndarray:
    """Spacing goal: maintain a desired neutral distance from the opponent."""
    t = _base_target()
    # raw rel_distance in [0.08, 0.35]; normalised = raw / 2.0
    raw_dist = float(np.random.uniform(0.08, 0.35))
    t[GOAL_INDEX["rel_distance"]] = float(np.clip(raw_dist / 2.0, 0.0, 1.0))
    t[GOAL_INDEX["rel_dy"]] = 0.5  # neutral vertical offset
    return clip_goal_target(t)


def _sampler_combat(_: np.ndarray) -> np.ndarray:
    """Combat goal: enter strike range with a favourable frame advantage."""
    t = _base_target()
    t[GOAL_INDEX["in_strike_range"]] = 1.0
    # normalised frame advantage in [0.55, 0.90] == slight-to-strong advantage
    t[GOAL_INDEX["frame_advantage_estimate"]] = float(np.random.uniform(0.55, 0.90))
    return clip_goal_target(t)


# ─── Mask helper ───────────────────────────────────────────────────────────────

def _mask_for(*active: tuple[str, float]) -> np.ndarray:
    mask = np.zeros((GOAL_DIM,), dtype=np.float32)
    for name, weight in active:
        mask[GOAL_INDEX[name]] = float(np.clip(weight, 0.0, 1.0))
    return mask


# ─── Phase specs ───────────────────────────────────────────────────────────────

def build_phase_spec(
    phase: str,
    death_penalty: float = 2.0,
    terminate_on_death: bool = True,
) -> StageSpec:
    phase = phase.strip().lower()

    if phase == "recovery_mastery":
        return StageSpec(
            stage_id=1,
            name="stage1_recovery_mastery",
            goal_type=normalize_goal_type("recovery"),
            mask=_mask_for(
                ("signed_dx_to_ledge", 1.0),
                ("dy_to_ledge", 1.0),
            ),
            target_sampler=_sampler_recovery,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            min_goal_duration=30,
            max_goal_duration=80,
            success_threshold=0.08,
            success_bonus=2.5,
            use_l2_error=True,
            reward_from_goal_progress=False,
            progress_scale=3.0,
            jump_usage_penalty_scale=0.10,
            velocity_penalty_scale=0.05,
            velocity_penalty_radius=1.5,
            offstage_penalty_scale=0.0,
            death_penalty=float(death_penalty),
            reward_clip=4.0,
            disable_attack=True,
            disable_dodge=False,
            disable_jump=False,
            step_penalty=0.05,
            terminate_on_death=bool(terminate_on_death),
        )

    if phase == "movement_fluency":
        return StageSpec(
            stage_id=2,
            name="stage2_movement_fluency",
            goal_type=normalize_goal_type("movement"),
            mask=_mask_for(("player_x", 1.0), ("player_y", 1.0)),
            target_sampler=_sampler_movement,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            min_goal_duration=20,
            max_goal_duration=40,
            success_threshold=0.04,
            success_bonus=1.5,
            use_l2_error=True,
            reward_from_goal_progress=False,
            progress_scale=2.0,
            vertical_velocity_penalty_scale=0.10,
            death_penalty=float(death_penalty),
            reward_clip=3.0,
            disable_attack=True,
            disable_dodge=True,
            disable_jump=False,
            step_penalty=0.1,
            terminate_on_death=bool(terminate_on_death),
        )

    if phase == "weapon_acquisition":
        return StageSpec(
            stage_id=3,
            name="stage3_weapon_acquisition",
            goal_type=normalize_goal_type("weapon_acquisition"),
            mask=_mask_for(
                ("player_has_weapon", 1.0),
                ("weapon_dx", 0.5),
                ("weapon_dy", 0.5),
            ),
            target_sampler=_sampler_weapon_acq,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            success_threshold=0.1,
            success_bonus=1.0,
            reward_from_goal_progress=False,
            player_has_weapon_bonus=0.1,
            weapon_pickup_bonus=2.0,
            step_penalty=0.05,
            offstage_penalty_scale=0.1,
            proximity_scale=0.0,
            conditional_weapon_guidance_when_unarmed=True,
            unarmed_weapon_dx_weight=0.5,
            unarmed_weapon_dy_weight=0.5,
            death_penalty=float(death_penalty),
            reward_clip=10,
            disable_attack=False,
            allowed_attack_actions=(0, 3),
            disable_dodge=False,
            disable_jump=False,
            agent_weapon_drop_penalty=1.0,
            force_drop_weapon_on_timeout=True,
            drop_weapon_key="num5",
            terminate_on_death=bool(terminate_on_death),
        )

    if phase == "spacing_neutral":
        return StageSpec(
            stage_id=4,
            name="stage4_spacing_neutral",
            goal_type=normalize_goal_type("spacing"),
            mask=_mask_for(("rel_distance", 1.0), ("rel_dy", 0.5)),
            target_sampler=_sampler_spacing,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            min_goal_duration=30,
            max_goal_duration=60,
            success_threshold=0.06,
            success_bonus=1.2,
            use_l2_error=True,
            reward_from_goal_progress=False,
            progress_scale=2.0,
            death_penalty=float(death_penalty),
            reward_clip=3.0,
            disable_attack=True,
            disable_dodge=False,
            disable_jump=False,
            step_penalty=0.05,
            terminate_on_death=bool(terminate_on_death),
        )

    if phase == "combat_execution":
        return StageSpec(
            stage_id=5,
            name="stage5_combat_execution",
            goal_type=normalize_goal_type("combat"),
            mask=_mask_for(
                ("in_strike_range", 1.0),
                ("frame_advantage_estimate", 0.8),
            ),
            target_sampler=_sampler_combat,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            min_goal_duration=40,
            max_goal_duration=80,
            progress_scale=1.5,
            progress_clip_min=-0.20,
            progress_clip_max=0.40,
            success_threshold=0.15,
            success_bonus=1.5,
            reward_from_goal_progress=True,
            in_strike_range_bonus=0.03,
            hit_event_bonus=0.5,
            damage_dealt_scale=2.0,
            self_damage_penalty_scale=0.5,
            offstage_penalty_scale=0.05,
            death_penalty=float(death_penalty),
            reward_clip=6.0,
            disable_attack=False,
            disable_dodge=False,
            disable_jump=False,
            step_penalty=0.003,
            terminate_on_death=bool(terminate_on_death),
        )

    raise ValueError(f"Unknown phase '{phase}'. Expected one of: {', '.join(PHASES)}")
