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
    "locomotion_grounded",
    "locomotion_airborne",
    "locomotion_recovery",
    "locomotion",
    "weapon_control",
    "damage_static_fist",
    "damage_static_weapon",
    "damage_dynamic",
    "damage_static",  # backward-compatible alias of damage_static_weapon
)

LOCO_PLATFORM_X_MARGIN = 0.03
LOCO_GROUNDED_Y_EPS = 0.015
LOCO_AIRBORNE_Y_DELTA = 0.28
LOCO_RECOVERY_STEP1_Y_MIN = 0.30
LOCO_RECOVERY_STEP1_Y_MAX = 0.70
LOCO_RECOVERY_STEP1_OUTSIDE_BAND = 0.14
LOCO_RECOVERY_STEP1_SIDE_OFFSET = 0.02
LOCO_RECOVERY_STEP1_FLIP_SIDE_PROB = 0.20
LOCO_RECOVERY_STEP2_LEDGE_PROB = 0.70
LOCO_RECOVERY_STEP2_LEDGE_X_INSET = 0.012
LOCO_RECOVERY_STEP2_PLATFORM_MARGIN = 0.03
LOCO_RECOVERY_STEP2_PLATFORM_Y_EPS = 0.015


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


def _sample_recovery_offstage_xy(player_x: float) -> tuple[float, float]:
    x_min = float(PLATFORM_BOUNDS["x_min"])
    x_max = float(PLATFORM_BOUNDS["x_max"])

    side = _nearest_platform_side(player_x)
    if float(np.random.rand()) < float(np.clip(LOCO_RECOVERY_STEP1_FLIP_SIDE_PROB, 0.0, 1.0)):
        side = "right" if side == "left" else "left"

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
    y = float(
        np.random.uniform(
            float(np.clip(LOCO_RECOVERY_STEP1_Y_MIN, 0.0, 1.0)),
            float(np.clip(LOCO_RECOVERY_STEP1_Y_MAX, 0.0, 1.0)),
        )
    )
    return x, y


def _sample_recovery_return_xy(player_x: float) -> tuple[float, float]:
    x_min = float(PLATFORM_BOUNDS["x_min"])
    x_max = float(PLATFORM_BOUNDS["x_max"])
    y_platform = float(np.clip(float(PLATFORM_BOUNDS["y_min"]) + LOCO_RECOVERY_STEP2_PLATFORM_Y_EPS, 0.0, 1.0))

    if float(np.random.rand()) < float(np.clip(LOCO_RECOVERY_STEP2_LEDGE_PROB, 0.0, 1.0)):
        side = _nearest_platform_side(player_x)
        if side == "left":
            tx = float(np.clip(x_min + LOCO_RECOVERY_STEP2_LEDGE_X_INSET, 0.0, 1.0))
        else:
            tx = float(np.clip(x_max - LOCO_RECOVERY_STEP2_LEDGE_X_INSET, 0.0, 1.0))
        return tx, y_platform

    margin = float(max(0.0, LOCO_RECOVERY_STEP2_PLATFORM_MARGIN))
    lo = x_min + margin
    hi = x_max - margin
    if hi <= lo:
        lo = x_min
        hi = x_max

    tx = float(np.random.uniform(lo, hi))
    return tx, y_platform


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


def _sampler_locomotion_recovery_offstage(obs: np.ndarray) -> np.ndarray:
    t = _base_target()

    player_x = 0.5 * (float(PLATFORM_BOUNDS["x_min"]) + float(PLATFORM_BOUNDS["x_max"]))
    try:
        o = np.asarray(obs, dtype=np.float32)
        player_x = float(np.clip(StateSpec.get(o, "player_x"), 0.0, 1.0))
    except Exception:
        pass

    tx, ty = _sample_recovery_offstage_xy(player_x)

    t[GOAL_INDEX["player_x"]] = tx
    t[GOAL_INDEX["player_y"]] = ty
    return clip_goal_target(t)


def _sampler_locomotion_recovery_return(obs: np.ndarray) -> np.ndarray:
    t = _base_target()

    player_x = 0.5 * (float(PLATFORM_BOUNDS["x_min"]) + float(PLATFORM_BOUNDS["x_max"]))
    try:
        o = np.asarray(obs, dtype=np.float32)
        player_x = float(np.clip(StateSpec.get(o, "player_x"), 0.0, 1.0))
    except Exception:
        pass

    tx, ty = _sample_recovery_return_xy(player_x)

    t[GOAL_INDEX["player_x"]] = tx
    t[GOAL_INDEX["player_y"]] = ty
    return clip_goal_target(t)


def _sampler_weapon(obs: np.ndarray) -> np.ndarray:
    t = _base_target()
    t[GOAL_INDEX["player_has_weapon"]] = 1.0
    # weapon_dx / weapon_dy targets are normalized to [0,1], where 0.5 means ~zero offset.
    t[GOAL_INDEX["weapon_dx"]] = 0.5
    t[GOAL_INDEX["weapon_dy"]] = 0.5
    t[GOAL_INDEX["player_is_offstage"]] = 0.0
    return clip_goal_target(t)


def _set_combat_relational_target(
    t: np.ndarray,
    *,
    requires_weapon: bool,
    rel_distance_range: tuple[float, float],
    frame_advantage_range: tuple[float, float],
) -> np.ndarray:
    t[GOAL_INDEX["player_has_weapon"]] = 1.0 if requires_weapon else 0.0
    t[GOAL_INDEX["weapon_dx"]] = 0.5 if requires_weapon else 0.0
    t[GOAL_INDEX["weapon_dy"]] = 0.5 if requires_weapon else 0.0
    t[GOAL_INDEX["in_strike_range"]] = 1.0

    dist_lo, dist_hi = rel_distance_range
    adv_lo, adv_hi = frame_advantage_range

    t[GOAL_INDEX["rel_distance"]] = float(np.random.uniform(max(0.0, dist_lo), min(1.0, dist_hi)))
    t[GOAL_INDEX["facing_opponent"]] = 1.0
    t[GOAL_INDEX["frame_advantage_estimate"]] = float(np.random.uniform(max(0.0, adv_lo), min(1.0, adv_hi)))
    t[GOAL_INDEX["player_is_offstage"]] = 0.0
    return t


def _sampler_damage_static_fist(obs: np.ndarray) -> np.ndarray:
    t = _base_target()
    # Stage A (attack trigger): learn WHEN to attack, independent of damage target.
    t[GOAL_INDEX["player_has_weapon"]] = 0.0
    t[GOAL_INDEX["weapon_dx"]] = 0.0
    t[GOAL_INDEX["weapon_dy"]] = 0.0
    t[GOAL_INDEX["in_strike_range"]] = 1.0
    t[GOAL_INDEX["rel_distance"]] = float(np.random.uniform(0.004, 0.050))
    t[GOAL_INDEX["facing_opponent"]] = 1.0
    t[GOAL_INDEX["frame_advantage_estimate"]] = 0.5
    t[GOAL_INDEX["player_is_offstage"]] = 0.0
    return clip_goal_target(t)


def _sampler_damage_static_weapon(obs: np.ndarray) -> np.ndarray:
    t = _base_target()
    _set_combat_relational_target(
        t,
        requires_weapon=True,
        # 0.008..0.050 ~= raw 0.016..0.100 (extended heavy/weapon window).
        rel_distance_range=(0.008, 0.050),
        frame_advantage_range=(0.55, 0.95),
    )
    return clip_goal_target(t)


def _sampler_damage_dynamic(obs: np.ndarray) -> np.ndarray:
    t = _base_target()
    _set_combat_relational_target(
        t,
        requires_weapon=True,
        # Keep dynamic within realistic hit windows while allowing spacing variance.
        rel_distance_range=(0.006, 0.050),
        frame_advantage_range=(0.55, 1.00),
    )
    return clip_goal_target(t)


def _mask_for(*active: tuple[str, float]) -> np.ndarray:
    mask = np.zeros((GOAL_DIM,), dtype=np.float32)
    for name, weight in active:
        mask[GOAL_INDEX[name]] = float(np.clip(weight, 0.0, 1.0))
    return mask


def build_phase_spec(
    phase: str,
    death_penalty: float = 2.0,
    terminate_on_death: bool = True,
) -> StageSpec:
    phase = phase.strip().lower()

    if phase == "locomotion":
        phase = "locomotion_grounded"

    if phase == "damage_static":
        phase = "damage_static_weapon"

    if phase == "locomotion_grounded":
        return StageSpec(
            stage_id=1,
            name="phase1_locomotion_grounded",
            goal_type=normalize_goal_type("spacing"),
            mask=_mask_for(("player_x", 1.0), ("player_y", 1.0)),
            target_sampler=_sampler_locomotion_grounded,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            min_goal_duration=20,
            max_goal_duration=40,
            success_threshold=0.02,
            success_bonus=1.5,
            proximity_scale=0.0,
            use_l2_error=True,
            vertical_velocity_penalty_scale=0.10,
            death_penalty=float(death_penalty),
            reward_clip=3.0,
            disable_attack=True,
            disable_dodge=True,
            disable_jump=True,
            reset_perturb_steps=0,
            step_penalty=0.1,
            terminate_on_death=bool(terminate_on_death),
        )

    if phase == "locomotion_airborne":
        return StageSpec(
            stage_id=2,
            name="phase2_locomotion_airborne",
            goal_type=normalize_goal_type("approach"),
            mask=_mask_for(("player_x", 1.0), ("player_y", 1.0)),
            target_sampler=_sampler_locomotion_airborne,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            min_goal_duration=20,
            max_goal_duration=40,
            success_threshold=0.05,
            success_bonus=1.5,
            proximity_scale=0.0,
            use_l2_error=True,
            jump_usage_penalty_scale=0.15,
            velocity_penalty_scale=0.1,
            velocity_penalty_radius=1.5,
            death_penalty=float(death_penalty),
            reward_clip=3.0,
            disable_attack=True,
            disable_dodge=True,
            disable_jump=False,
            reset_perturb_steps=0,
            step_penalty=0.1,
            terminate_on_death=bool(terminate_on_death),
        )

    if phase == "locomotion_recovery":
        return StageSpec(
            stage_id=3,
            name="phase3_locomotion_recovery",
            goal_type=normalize_goal_type("recovery"),
            mask=_mask_for(("player_x", 1.0), ("player_y", 1.0)),
            target_sampler=_sampler_locomotion_recovery_offstage,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            min_goal_duration=24,
            max_goal_duration=60,
            success_threshold=0.03,
            success_bonus=2.2,
            proximity_scale=0.0,
            use_l2_error=True,
            jump_usage_penalty_scale=0.15,
            velocity_penalty_scale=0.1,
            velocity_penalty_radius=1.5,
            sequential_goal_enabled=True,
            sequential_target_sampler=_sampler_locomotion_recovery_return,
            sequential_step1_bonus=0.35,
            sequential_failure_penalty=0.75,
            sequential_require_offstage_first=True,
            sequential_require_onstage_second=True,
            death_penalty=float(death_penalty),
            reward_clip=3.2,
            disable_attack=True,
            disable_dodge=True,
            disable_jump=False,
            reset_perturb_steps=0,
            step_penalty=0.1,
            terminate_on_death=bool(terminate_on_death),
        )

    if phase == "weapon_control":
        return StageSpec(
            stage_id=4,
            name="phase4_weapon_control",
            goal_type=normalize_goal_type("weapon_acquisition"),
            mask=_mask_for(
                ("player_has_weapon", 1.0),
                ("player_is_offstage", 1.0),
                ("weapon_dx", 1.0),
                ("weapon_dy", 1.0),
            ),
            target_sampler=_sampler_weapon,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            min_goal_duration=20,
            max_goal_duration=40,
            progress_scale=2.0,
            progress_clip_min=-0.15,
            progress_clip_max=0.25,
            success_threshold=0.16,
            success_bonus=0.0,
            reward_from_goal_progress=False,
            player_has_weapon_bonus=0.1,
            proximity_scale=0.0,
            offstage_penalty_scale=0.08,
            death_penalty=float(death_penalty),
            reward_clip=10,
            disable_attack=False,
            allowed_attack_actions=(0, 3),
            disable_dodge=False,
            disable_jump=False,
            reset_perturb_steps=0,
            step_penalty=0,
            conditional_weapon_guidance_when_unarmed=False,
            unarmed_weapon_dx_weight=0.0,
            unarmed_weapon_dy_weight=0.0,
            player_xy_to_weapon_goal_when_unarmed=False,
            anchor_player_xy_goal_when_armed=False,
            agent_weapon_drop_penalty=1 ,
            force_drop_weapon_on_timeout=True,
            drop_weapon_key="num5",
            terminate_on_death=bool(terminate_on_death),
        )

    if phase == "damage_static_fist":
        return StageSpec(
            stage_id=5,
            name="phase5_damage_static_fist",
            goal_type=normalize_goal_type("attack"),
            mask=_mask_for(
                ("in_strike_range", 1.0),
                ("rel_distance", 0.8),
                ("facing_opponent", 0.7),
            ),
            target_sampler=_sampler_damage_static_fist,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            min_goal_duration=50,
            max_goal_duration=90,
            progress_scale=0.5,
            progress_clip_min=-0.20,
            progress_clip_max=0.30,
            success_threshold=0.5,
            success_bonus=0.6,
            proximity_scale=0.0,
            chase_rel_distance_scale=0.0,
            in_strike_range_bonus=0.0,
            facing_opponent_bonus=0.0,
            hit_event_bonus=0.0,
            damage_dealt_scale=0.0,
            self_damage_penalty_scale=0.6,
            offstage_penalty_scale=0.0,
            combo_penalty_scale=0.0,
            attack_whiff_penalty_scale=0.20,
            attack_commit_threshold=0.7,
            attack_commit_bonus=0.2,
            no_attack_in_range_penalty=0.1,
            attack_out_of_range_threshold=0.5,
            attack_out_of_range_keep_prob=0.25,
            death_penalty=float(death_penalty),
            reward_clip=4,
            disable_attack=False,
            allowed_attack_actions=(0, 1, 2),
            disable_dodge=False,
            disable_jump=False,
            reset_perturb_steps=0,
            step_penalty=0.002,
            reward_from_goal_progress=True,
            terminate_on_death=bool(terminate_on_death),
            terminate_on_hit_event=False,
            require_attack_for_success=True,
            resample_goal_on_opponent_stock_loss=False,
            opponent_ko_bonus=0.0,
        )

    if phase == "damage_static_weapon":
        return StageSpec(
            stage_id=6,
            name="phase6_damage_static_weapon",
            goal_type=normalize_goal_type("attack"),
            mask=_mask_for(
                ("player_has_weapon", 1.0),
                ("weapon_dx", 0.1),
                ("weapon_dy", 0.1),
                ("in_strike_range", 1.0),
                ("rel_distance", 0.5),
                ("facing_opponent", 0.3),
                ("player_is_offstage", 0.2),
            ),
            target_sampler=_sampler_damage_static_weapon,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            min_goal_duration=60,
            max_goal_duration=120,
            progress_scale=1.0,
            progress_clip_min=-0.20,
            progress_clip_max=0.45,
            success_threshold=0.14,
            success_bonus=1.0,
            proximity_scale=0.0,
            chase_rel_distance_scale=0.0,
            in_strike_range_bonus=0.02,
            facing_opponent_bonus=0.0,
            hit_event_bonus=0.8,
            damage_dealt_scale=3.5,
            self_damage_penalty_scale=1.0,
            offstage_penalty_scale=0.05,
            combo_penalty_scale=0.0,
            attack_whiff_penalty_scale=0.20,
            attack_commit_threshold=0.7,
            attack_commit_bonus=0.0,
            no_attack_in_range_penalty=0.0,
            attack_out_of_range_threshold=0.5,
            attack_out_of_range_keep_prob=0.35,
            death_penalty=float(death_penalty),
            reward_clip=8,
            disable_attack=False,
            allowed_attack_actions=(0, 1, 2),
            disable_dodge=False,
            disable_jump=False,
            reset_perturb_steps=0,
            step_penalty=0.003,
            reward_from_goal_progress=True,
            terminate_on_death=bool(terminate_on_death),
            resample_goal_on_opponent_stock_loss=True,
            opponent_ko_bonus=8.0,
        )

    if phase == "damage_dynamic":
        return StageSpec(
            stage_id=7,
            name="phase7_damage_dynamic",
            goal_type=normalize_goal_type("attack"),
            mask=_mask_for(
                ("player_has_weapon", 0.2),
                ("in_strike_range", 1.0),
                ("rel_distance", 0.6),
                ("facing_opponent", 0.4),
                ("frame_advantage_estimate", 0.4),
                ("player_is_offstage", 0.3),
            ),
            target_sampler=_sampler_damage_dynamic,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            min_goal_duration=70,
            max_goal_duration=130,
            progress_scale=1.1,
            progress_clip_min=-0.25,
            progress_clip_max=0.9,
            success_threshold=0.18,
            success_bonus=1.2,
            proximity_scale=0.0,
            chase_rel_distance_scale=0.0,
            in_strike_range_bonus=0.04,
            facing_opponent_bonus=0.0,
            hit_event_bonus=1.0,
            damage_dealt_scale=4.2,
            self_damage_penalty_scale=1.0,
            offstage_penalty_scale=0.05,
            combo_penalty_scale=0.0,
            attack_whiff_penalty_scale=0.12,
            combo_chain_bonus_scale=0.1,
            combo_chain_reset_time_since_hit=0.10,
            attack_commit_threshold=0.7,
            attack_commit_bonus=0.0,
            no_attack_in_range_penalty=0.0,
            attack_out_of_range_threshold=0.5,
            attack_out_of_range_keep_prob=0.45,
            death_penalty=float(death_penalty),
            reward_clip=8,
            disable_attack=False,
            allowed_attack_actions=(0, 1, 2),
            disable_dodge=False,
            disable_jump=False,
            reset_perturb_steps=0,
            step_penalty=0.003,
            reward_from_goal_progress=True,
            terminate_on_death=bool(terminate_on_death),
            resample_goal_on_opponent_stock_loss=True,
            opponent_ko_bonus=12.0,
        )

    raise ValueError(f"Unknown phase '{phase}'. Expected one of: {', '.join(PHASES)}")
