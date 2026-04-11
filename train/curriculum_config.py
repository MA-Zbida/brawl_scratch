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
    "damage_static_fist",
    "damage_static_weapon",
    "damage_dynamic",
    "damage_static",  # backward-compatible alias of damage_static_weapon
)

LOCO_PLATFORM_X_MARGIN = 0.03
LOCO_GROUNDED_Y_EPS = 0.015
LOCO_AIRBORNE_Y_DELTA = 0.35
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
    y = float(np.random.uniform(y_center - LOCO_AIRBORNE_Y_DELTA, y_center - LOCO_AIRBORNE_Y_DELTA / 4.0))
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
    opponent_damage_delta: tuple[float, float],
    frame_advantage_range: tuple[float, float],
    current_opponent_damage: float = 0.0,
) -> np.ndarray:
    t[GOAL_INDEX["player_has_weapon"]] = 1.0 if requires_weapon else 0.0
    t[GOAL_INDEX["weapon_dx"]] = 0.5
    t[GOAL_INDEX["weapon_dy"]] = 0.5
    t[GOAL_INDEX["in_strike_range"]] = 1.0

    dist_lo, dist_hi = rel_distance_range
    delta_lo, delta_hi = opponent_damage_delta
    adv_lo, adv_hi = frame_advantage_range

    t[GOAL_INDEX["rel_distance"]] = float(np.random.uniform(max(0.0, dist_lo), min(1.0, dist_hi)))
    t[GOAL_INDEX["facing_opponent"]] = 1.0
    t[GOAL_INDEX["frame_advantage_estimate"]] = float(np.random.uniform(max(0.0, adv_lo), min(1.0, adv_hi)))
    # Target is current opponent damage + a positive delta (damage only goes up
    # until a KO, and episode resets don't restart the match).
    dmg_target = current_opponent_damage + float(np.random.uniform(delta_lo, delta_hi))
    t[GOAL_INDEX["opponent_damage_pct"]] = float(np.clip(dmg_target, 0.0, 1.0))
    t[GOAL_INDEX["player_is_offstage"]] = 0.0
    return t


def _sampler_damage_static_fist(obs: np.ndarray) -> np.ndarray:
    t = _base_target()
    cur_dmg = float(np.clip(StateSpec.get(obs, "opponent_damage_pct"), 0.0, 1.0))
    _set_combat_relational_target(
        t,
        requires_weapon=False,
        rel_distance_range=(0.10, 0.30),
        opponent_damage_delta=(0.05, 0.20),
        frame_advantage_range=(0.40, 0.90),
        current_opponent_damage=cur_dmg,
    )
    return clip_goal_target(t)


def _sampler_damage_static_weapon(obs: np.ndarray) -> np.ndarray:
    t = _base_target()
    cur_dmg = float(np.clip(StateSpec.get(obs, "opponent_damage_pct"), 0.0, 1.0))
    _set_combat_relational_target(
        t,
        requires_weapon=True,
        rel_distance_range=(0.08, 0.22),
        opponent_damage_delta=(0.10, 0.20),
        frame_advantage_range=(0.60, 1.00),
        current_opponent_damage=cur_dmg,
    )
    return clip_goal_target(t)


def _sampler_damage_dynamic(obs: np.ndarray) -> np.ndarray:
    t = _base_target()
    cur_dmg = float(np.clip(StateSpec.get(obs, "opponent_damage_pct"), 0.0, 1.0))
    _set_combat_relational_target(
        t,
        requires_weapon=True,
        rel_distance_range=(0.10, 0.28),
        opponent_damage_delta=(0.15, 0.35),
        frame_advantage_range=(0.70, 1.00),
        current_opponent_damage=cur_dmg,
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
            mask=_mask_for(("player_x", 1.0), ("player_y", 1.0)),
            target_sampler=_sampler_locomotion_grounded,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            min_goal_duration=20,
            max_goal_duration=40,
            success_threshold=0.02,
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
            step_penalty=0.01,
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
            success_threshold=0.02,
            success_bonus=1.5,
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
            step_penalty=0.01,
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
            success_threshold=0.02,
            success_bonus=1.5,
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
            step_penalty=0.01,
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
            terminate_on_goal_success=False,
            resample_goal_on_timer=False,
        )

    if phase == "damage_static_fist":
        return StageSpec(
            stage_id=5,
            name="phase5_damage_static_fist",
            mask=_mask_for(
                ("player_has_weapon", 0.0),
                ("in_strike_range", 1.0),
                ("rel_distance", 0.8),
                ("facing_opponent", 0.9),
                ("opponent_damage_pct", 1.0),
                ("player_is_offstage", 0.8),
            ),
            target_sampler=_sampler_damage_static_fist,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            min_goal_duration=16,
            max_goal_duration=30,
            progress_scale=2.0,
            progress_clip_min=-0.25,
            progress_clip_max=0.8,
            success_threshold=0.10,
            success_bonus=1.2,
            proximity_scale=0.35,
            chase_rel_distance_scale=0.30,
            in_strike_range_bonus=0.08,
            facing_opponent_bonus=0.1,
            hit_event_bonus=3.0,
            damage_dealt_scale=2.8,
            self_damage_penalty_scale=1.3,
            offstage_penalty_scale=0.12,
            death_penalty=float(death_penalty),
            reward_clip=5,
            disable_attack=False,
            allowed_attack_actions=(0, 1, 2),
            disable_dodge=False,
            disable_jump=False,
            reset_perturb_steps=0,
            step_penalty=0.01,
            terminate_on_death=bool(terminate_on_death),
            terminate_on_goal_success=False,
            terminate_on_hit_event=False,
            resample_goal_on_timer=False,
            resample_goal_on_opponent_stock_loss=True,
            opponent_ko_bonus=5.0,
        )

    if phase == "damage_static_weapon":
        return StageSpec(
            stage_id=6,
            name="phase6_damage_static_weapon",
            mask=_mask_for(
                ("player_has_weapon", 1.0),
                ("weapon_dx", 0.5),
                ("weapon_dy", 0.5),
                ("in_strike_range", 1.0),
                ("rel_distance", 0.9),
                ("facing_opponent", 0.9),
                ("opponent_damage_pct", 1.0),
                ("player_is_offstage", 0.8),
            ),
            target_sampler=_sampler_damage_static_weapon,
            feature_names=list(CURRICULUM_GOAL_FEATURES),
            goal_extractor=extract_curriculum_goal_features,
            min_goal_duration=16,
            max_goal_duration=30,
            progress_scale=2.0,
            progress_clip_min=-0.25,
            progress_clip_max=0.8,
            success_threshold=0.06,
            success_bonus=1.5,
            proximity_scale=0.35,
            chase_rel_distance_scale=0.25,
            in_strike_range_bonus=0.10,
            facing_opponent_bonus=0.06,
            hit_event_bonus=0.14,
            damage_dealt_scale=2.5,
            self_damage_penalty_scale=1.4,
            offstage_penalty_scale=0.12,
            death_penalty=float(death_penalty),
            reward_clip=5,
            disable_attack=False,
            disable_dodge=False,
            disable_jump=False,
            reset_perturb_steps=0,
            terminate_on_death=bool(terminate_on_death),
            terminate_on_goal_success=True,
            resample_goal_on_timer=False,
            resample_goal_on_opponent_stock_loss=True,
            opponent_ko_bonus=5.0,
        )

    if phase == "damage_dynamic":
        return StageSpec(
            stage_id=7,
            name="phase7_damage_dynamic",
            mask=_mask_for(
                ("player_has_weapon", 0.3),
                ("in_strike_range", 1.0),
                ("rel_distance", 0.8),
                ("facing_opponent", 0.8),
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
            chase_rel_distance_scale=0.18,
            in_strike_range_bonus=0.07,
            facing_opponent_bonus=0.04,
            hit_event_bonus=0.10,
            damage_dealt_scale=1.8,
            self_damage_penalty_scale=1.2,
            offstage_penalty_scale=0.12,
            death_penalty=float(death_penalty),
            reward_clip=2.8,
            disable_attack=False,
            disable_dodge=False,
            disable_jump=False,
            reset_perturb_steps=0,
            terminate_on_death=bool(terminate_on_death),
            terminate_on_goal_success=True,
            resample_goal_on_timer=False,
            resample_goal_on_opponent_stock_loss=True,
            opponent_ko_bonus=5.0,
        )

    raise ValueError(f"Unknown phase '{phase}'. Expected one of: {', '.join(PHASES)}")
