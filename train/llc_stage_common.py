from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, cast

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from algo.anchored_replay_ppo import AnchoredReplayPPO
from env import BrawlDeepEnv, EnvConfig
from feature_extractor.memory.state_spec import StateSpec
from hierarchical.goals import GOAL_FEATURE_NAMES, GOAL_STATE_SPEC_NAMES, GOAL_TARGET_DIM, extract_goal_features
from train.curriculum_goals import GOAL_TYPE_INDEX, goal_type_onehot, normalize_goal_type
from train.retention import (
    load_best_scores,
    parse_phase_list,
    previous_phases,
    retention_and_amnesia,
    save_best_scores,
    skill_score_for_phase,
    update_best_scores,
)


ActionAdapter = Callable[[np.ndarray], np.ndarray]
TargetSampler = Callable[[np.ndarray], np.ndarray]
GoalFamilySampler = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, str]]
GoalExtractor = Callable[[np.ndarray], np.ndarray]


FEATURE_SCALE: dict[str, float] = {
    "player_x": 1.0,
    "player_y": 1.0,
    "dist_to_nearest_ledge": 2.0,
    "player_is_offstage": 1.0,
    "player_has_weapon": 1.0,
    "player_jumps_norm": 1.0,
    "dy_to_ledge": 1.0,
    "rel_dx": 2.0,
    "rel_dy": 2.0,
    "weapon_dx": 2.0,
    "weapon_dy": 2.0,
    "rel_distance": 2.0,
    "facing_opponent": 1.0,
    "in_strike_range": 1.0,
    "opponent_damage_pct": 1.0,
    "opponent_hitstun": 1.0,
    "frame_advantage_estimate": 2.0,
}


@dataclass
class StageSpec:
    """Parametrises one LLC training stage.

    Legacy stages use the unified 7-dim goal space from hierarchical/goals.py.
    New curriculum stages may provide a custom goal extractor and feature list.
    ``mask`` selects which goal dimensions are active for this stage.
    """

    stage_id: int
    name: str
    mask: np.ndarray          # goal_dim, values in [0, 1]
    target_sampler: TargetSampler  # must return a goal_dim array in [0, 1]
    goal_type: str = "spacing"
    goal_family_sampler: Optional[GoalFamilySampler] = None
    feature_names: Optional[list[str]] = None  # logging + goal feature order
    goal_extractor: Optional[GoalExtractor] = None
    min_goal_duration: int = 100
    max_goal_duration: int = 120
    progress_scale: float = 1.0
    progress_clip_min: float = -0.1
    progress_clip_max: float = 0.3
    clip_progress_reward: bool = True
    success_threshold: float = 0.12
    success_bonus: float = 0.25
    proximity_scale: float = 0.0  # penalty for current error: -proximity_scale * error
    chase_rel_distance_scale: float = 0.0
    in_strike_range_bonus: float = 0.0
    facing_opponent_bonus: float = 0.0
    hit_event_bonus: float = 0.0
    damage_dealt_scale: float = 0.0
    self_damage_penalty_scale: float = 0.0
    offstage_penalty_scale: float = 0.0
    combo_penalty_scale: float = 0.0  # penalize opponent_time_since_hit to encourage combos
    attack_whiff_penalty_scale: float = 0.0  # penalize attack inputs that do not connect
    attack_commit_threshold: float = 0.7
    attack_commit_bonus: float = 0.0
    no_attack_in_range_penalty: float = 0.0
    attack_out_of_range_threshold: float = 0.5
    attack_out_of_range_keep_prob: float = 1.0
    combo_chain_bonus_scale: float = 0.0
    combo_chain_reset_time_since_hit: float = 0.12
    require_attack_for_success: bool = False
    require_opponent_damage_for_success: bool = False
    opponent_damage_success_tolerance: float = 0.01
    use_l2_error: bool = False
    death_penalty: float = 0.0    # penalty applied when agent loses a stock
    velocity_penalty_scale: float = 0.0   # penalise speed when near goal
    velocity_penalty_radius: float = 2.0  # error threshold to start damping
    vertical_velocity_penalty_scale: float = 0.0  # penalise |player_vy|
    jump_usage_penalty_scale: float = 0.0  # penalise jumps_used = (1 - player_jumps_norm)
    stay_bonus: float = 0.0               # bonus each step the agent stays inside success radius with low vel
    velocity_threshold: float = 0.15      # speed considered "stopped"
    reward_clip: float = 1.0
    disable_attack: bool = False
    allowed_attack_actions: Optional[tuple[int, ...]] = None  # attack channel values allowed (0=none,1=num4,2=num6,3=num5)
    disable_dodge: bool = False
    disable_jump: bool = False
    reset_perturb_steps: int = 0
    step_penalty: float = 0.0  # constant per-step penalty to discourage time wastage
    idle_movement_penalty: float = 0.0
    idle_action_index: int = 3
    reward_from_goal_progress: bool = False
    player_has_weapon_bonus: float = 0.0
    weapon_pickup_bonus: float = 0.0
    conditional_weapon_guidance_when_unarmed: bool = False
    unarmed_weapon_dx_weight: float = 0.0
    unarmed_weapon_dy_weight: float = 0.0
    player_xy_to_weapon_goal_when_unarmed: bool = False
    anchor_player_xy_goal_when_armed: bool = False
    player_xy_to_opponent_goal: bool = False
    anchor_player_xy_goal_when_in_strike_range: bool = False
    agent_weapon_drop_penalty: float = 0.0
    force_drop_weapon_on_timeout: bool = False
    drop_weapon_key: str = "num5"
    terminate_on_death: bool = False
    terminate_on_goal_success: bool = False
    terminate_on_hit_event: bool = False
    hit_event_damage_threshold: float = 1e-6
    resample_goal_on_timer: bool = False
    resample_goal_on_opponent_stock_loss: bool = False
    opponent_ko_bonus: float = 0.0
    sample_goal_only_when_player_exists: bool = True
    sequential_goal_enabled: bool = False
    sequential_target_sampler: Optional[TargetSampler] = None
    sequential_step1_bonus: float = 0.0
    sequential_failure_penalty: float = 0.0
    sequential_require_offstage_first: bool = False
    sequential_require_onstage_second: bool = False


class StageGoalEnv(gym.Wrapper):
    """Stage-specific LLC training wrapper with masked goal-error shaping."""

    def __init__(
        self,
        env: gym.Env,
        spec: StageSpec,
        action_adapter: Optional[ActionAdapter] = None,
    ):
        super().__init__(env)
        self.stage_spec = spec
        self.action_adapter = action_adapter
        self.goal_extractor = spec.goal_extractor
        self.goal_type = "spacing"
        self.goal_type_index = int(GOAL_TYPE_INDEX[self.goal_type])
        self._goal_type_onehot = goal_type_onehot(self.goal_type)
        self._set_goal_type(spec.goal_type)

        if spec.feature_names is not None:
            self.goal_dim = int(len(spec.feature_names))
            self.feature_names = list(spec.feature_names)
        else:
            # Default behavior for legacy stages.
            self.goal_dim = GOAL_TARGET_DIM
            self.feature_names = list(GOAL_FEATURE_NAMES)

        self.mask = np.asarray(spec.mask, dtype=np.float32).reshape(self.goal_dim)
        self.mask = np.clip(self.mask, 0.0, 1.0)
        self._sampled_mask = self.mask.copy()
        self._active_mask = self.mask.copy()
        self._feature_index = {name: idx for idx, name in enumerate(self.feature_names)}

        obs_shape = getattr(env.observation_space, "shape", None)
        if obs_shape is None or len(obs_shape) == 0:
            raise ValueError("Underlying env must expose a 1D observation space with known shape")

        base_dim = int(obs_shape[0])
        self._base_dim = base_dim
        self._aug_dim = base_dim + (2 * self.goal_dim)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._aug_dim,),
            dtype=np.float32,
        )

        self._obs_buf = np.zeros((self._aug_dim,), dtype=np.float32)
        self._goal_target = np.zeros((self.goal_dim,), dtype=np.float32)
        self._goal_steps_left = 0
        self._goal_active = False
        self._step_count = 0
        self._awaiting_respawn_after_death = False
        self._prev_error: float | None = None
        self._prev_has_weapon: Optional[bool] = None
        self._prev_in_range: float = 0.0
        self._combo_chain_hits: int = 0
        self._seq_phase: int = 1
        self._seq_step1_completed: bool = False
        self._allowed_attack_actions: Optional[set[int]] = None
        if spec.allowed_attack_actions is not None:
            allowed = {int(v) for v in spec.allowed_attack_actions}
            allowed = {v for v in allowed if 0 <= v <= 3}
            if not allowed:
                allowed = {0}
            self._allowed_attack_actions = allowed

    def _set_goal_type(self, goal_type: str) -> None:
        self.goal_type = normalize_goal_type(goal_type)
        self.goal_type_index = int(GOAL_TYPE_INDEX[self.goal_type])
        self._goal_type_onehot = goal_type_onehot(self.goal_type)

    def _sequential_enabled(self) -> bool:
        return bool(self.stage_spec.sequential_goal_enabled and self.stage_spec.sequential_target_sampler is not None)

    def _sample_goal_for_sequence_step(self, obs: np.ndarray, step: int) -> None:
        if int(step) == 2 and self.stage_spec.sequential_target_sampler is not None:
            self._seq_phase = 2
            self._sample_goal(obs, sampler=self.stage_spec.sequential_target_sampler)
            return
        self._seq_phase = 1
        self._seq_step1_completed = False
        self._sample_goal(obs, sampler=self.stage_spec.target_sampler)

    def _extract(self, obs: np.ndarray) -> np.ndarray:
        if self.goal_extractor is not None:
            feats = np.asarray(self.goal_extractor(obs), dtype=np.float32).reshape(-1)
        else:
            feats = np.asarray(extract_goal_features(obs), dtype=np.float32).reshape(-1)
        if feats.shape[0] != self.goal_dim:
            raise ValueError(
                f"Goal extractor returned dim={feats.shape[0]}, expected {self.goal_dim}"
            )
        return feats

    def _error(self, obs: np.ndarray, target: np.ndarray) -> float:
        feats = self._extract(obs)  # already in [0, 1]; no scaling needed
        return self._goal_error_from_feats(feats, target)

    def _goal_error_from_feats(self, feats: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        use_mask = self.mask if mask is None else np.asarray(mask, dtype=np.float32).reshape(self.goal_dim)
        delta = np.asarray(feats, dtype=np.float32) - np.asarray(target, dtype=np.float32)
        if self.stage_spec.use_l2_error:
            return float(np.sqrt(np.sum(use_mask * (delta ** 2))))
        return float(np.sum(use_mask * np.abs(delta)))

    def _feature_value(self, feats: np.ndarray, name: str, default: float = 0.0) -> float:
        idx = self._feature_index.get(name)
        if idx is None:
            return float(default)
        return float(feats[idx])

    def _effective_mask_for_step(self, feats: np.ndarray) -> np.ndarray:
        mask = self._sampled_mask.copy()
        if not self.stage_spec.conditional_weapon_guidance_when_unarmed:
            return mask

        has_weapon = self._feature_value(feats, "player_has_weapon", 0.0) > 0.5
        dx_idx = self._feature_index.get("weapon_dx")
        dy_idx = self._feature_index.get("weapon_dy")

        if dx_idx is None and dy_idx is None:
            return mask

        if has_weapon:
            if dx_idx is not None:
                mask[dx_idx] = 0.0
            if dy_idx is not None:
                mask[dy_idx] = 0.0
            return mask

        if dx_idx is not None:
            mask[dx_idx] = float(np.clip(self.stage_spec.unarmed_weapon_dx_weight, 0.0, 1.0))
        if dy_idx is not None:
            mask[dy_idx] = float(np.clip(self.stage_spec.unarmed_weapon_dy_weight, 0.0, 1.0))
        return mask

    def _maybe_update_player_xy_goal(self, obs: np.ndarray) -> None:
        x_idx = self._feature_index.get("player_x")
        y_idx = self._feature_index.get("player_y")
        if x_idx is None or y_idx is None:
            return

        o = np.asarray(obs, dtype=np.float32)
        try:
            player_x = float(np.clip(StateSpec.get(o, "player_x"), 0.0, 1.0))
            player_y = float(np.clip(StateSpec.get(o, "player_y"), 0.0, 1.0))
        except Exception:
            return

        # Weapon-acquisition retargeting: move toward weapon while unarmed.
        if self.stage_spec.player_xy_to_weapon_goal_when_unarmed:
            try:
                has_weapon = float(np.clip(StateSpec.get(o, "player_has_weapon"), 0.0, 1.0)) > 0.5
                if has_weapon:
                    if self.stage_spec.anchor_player_xy_goal_when_armed:
                        self._goal_target[x_idx] = player_x
                        self._goal_target[y_idx] = player_y
                else:
                    weapon_dx = float(StateSpec.get(o, "weapon_dx"))
                    weapon_dy = float(StateSpec.get(o, "weapon_dy"))
                    self._goal_target[x_idx] = float(np.clip(player_x + weapon_dx, 0.0, 1.0))
                    self._goal_target[y_idx] = float(np.clip(player_y + weapon_dy, 0.0, 1.0))
                    # Keep weapon reacquire as top priority while unarmed.
                    return
            except Exception:
                pass

        # Combat retargeting: move toward opponent position.
        if self.stage_spec.player_xy_to_opponent_goal:
            try:
                in_range = float(np.clip(StateSpec.get(o, "in_strike_range"), 0.0, 1.0)) > 0.5
                if in_range and self.stage_spec.anchor_player_xy_goal_when_in_strike_range:
                    self._goal_target[x_idx] = player_x
                    self._goal_target[y_idx] = player_y
                else:
                    rel_dx = float(StateSpec.get(o, "rel_dx"))
                    rel_dy = float(StateSpec.get(o, "rel_dy"))
                    self._goal_target[x_idx] = float(np.clip(player_x + rel_dx, 0.0, 1.0))
                    self._goal_target[y_idx] = float(np.clip(player_y + rel_dy, 0.0, 1.0))
            except Exception:
                pass

    def _force_drop_weapon_if_needed(self, feats: np.ndarray, truncated: bool) -> float:
        if not truncated or not self.stage_spec.force_drop_weapon_on_timeout:
            return 0.0
        has_weapon = self._feature_value(feats, "player_has_weapon", 0.0) > 0.5
        if not has_weapon:
            return 0.0

        key = str(self.stage_spec.drop_weapon_key or "num5").strip().lower() or "num5"
        try:
            controller = getattr(self.unwrapped, "input_controller", None)
            if controller is not None and hasattr(controller, "tap"):
                controller.tap({key})
                return 1.0
        except Exception:
            return 0.0
        return 0.0

    @staticmethod
    def _attack_action_value_for_key(key: str) -> int:
        key_to_action = {
            "num4": 1,
            "num_4": 1,
            "num6": 2,
            "num_6": 2,
            "num5": 3,
            "num_5": 3,
        }
        return int(key_to_action.get(str(key or "").strip().lower(), 0))

    def _sample_goal(self, obs: np.ndarray, sampler: Optional[TargetSampler] = None) -> None:
        if sampler is None and self.stage_spec.goal_family_sampler is not None:
            target, mask, goal_type = self.stage_spec.goal_family_sampler(obs)
            self._goal_target = np.asarray(target, dtype=np.float32).reshape(-1)
            self._sampled_mask = np.clip(np.asarray(mask, dtype=np.float32).reshape(self.goal_dim), 0.0, 1.0)
            self._set_goal_type(goal_type)
        else:
            sampler_fn = self.stage_spec.target_sampler if sampler is None else sampler
            self._goal_target = np.asarray(sampler_fn(obs), dtype=np.float32).reshape(-1)
            self._sampled_mask = self.mask.copy()
            self._set_goal_type(self.stage_spec.goal_type)
        if self._goal_target.shape[0] != self.goal_dim:
            raise ValueError(
                f"Target sampler returned dim={self._goal_target.shape[0]}, expected {self.goal_dim}"
            )
        self._goal_active = True
        self._goal_steps_left = -1

    def _activate_goal(self, obs: np.ndarray, feats: np.ndarray, *, sequence_step: Optional[int] = None) -> None:
        if sequence_step is None:
            self._sample_goal(obs)
        else:
            self._sample_goal_for_sequence_step(obs, step=sequence_step)
        self._maybe_update_player_xy_goal(obs)
        self._active_mask = self._effective_mask_for_step(feats)

    def _player_is_controllable(self, info: Optional[dict] = None) -> bool:
        """Returns whether the player can currently execute actions."""
        info = info or {}

        timer = info.get("player_respawn_timer")
        if timer is not None:
            try:
                if float(timer) > 1e-6:
                    return False
            except Exception:
                pass

        marker = info.get("player_exists")
        if marker is not None:
            try:
                return float(marker) > 0.5
            except Exception:
                pass

        mem = getattr(self.env, "memory", None)
        if mem is not None:
            mem_timer = getattr(mem, "player_respawn_timer", None)
            if mem_timer is not None:
                try:
                    if float(mem_timer) > 1e-6:
                        return False
                except Exception:
                    pass

            player = getattr(mem, "player", None)
            if player is not None and hasattr(player, "exists"):
                try:
                    return bool(player.exists)
                except Exception:
                    pass

        return True

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        # Emit [base_obs | goal_target(goal_dim) | mask(goal_dim)].
        # goal_target is already in [0, 1] (extract_goal_features space).
        np.copyto(self._obs_buf[: self._base_dim], obs)
        np.copyto(self._obs_buf[self._base_dim : self._base_dim + self.goal_dim], self._goal_target)
        np.copyto(self._obs_buf[self._base_dim + self.goal_dim :], self._active_mask)
        return self._obs_buf

    def _perturb_reset(self) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset()
        if self.stage_spec.reset_perturb_steps <= 0:
            return obs, info

        direction = 0 if np.random.rand() < 0.5 else 1
        for _ in range(self.stage_spec.reset_perturb_steps):
            jump = 1 if np.random.rand() < 0.35 else 0
            action = np.array([direction, jump, 0, 0], dtype=np.int64)
            obs, _, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                obs, info = self.env.reset()
                break
        return obs, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            obs, info = self.env.reset(seed=seed, options=options)
        else:
            obs, info = self._perturb_reset()

        # Guarantee agent starts weaponless regardless of force-drop timing.
        _unwrapped = self.unwrapped
        _mem = getattr(_unwrapped, "memory", None)
        if _mem is not None:
            _player = getattr(_mem, "player", None)
            if _player is not None:
                _player.weapon_state = 0.0

        obs = np.asarray(obs, dtype=np.float32)
        init_feats = self._extract(obs)
        self._step_count = 0
        self._seq_phase = 1
        self._seq_step1_completed = False
        if self.stage_spec.sample_goal_only_when_player_exists and not self._player_is_controllable(info):
            self._goal_target.fill(0.0)
            self._goal_steps_left = -1
            self._goal_active = False
            self._active_mask = self.mask.copy()
        else:
            if self._sequential_enabled():
                self._activate_goal(obs, init_feats, sequence_step=1)
            else:
                self._activate_goal(obs, init_feats)
        self._prev_has_weapon = self._feature_value(init_feats, "player_has_weapon", 0.0) > 0.5
        self._prev_in_range = float(np.clip(self._feature_value(init_feats, "in_strike_range", 0.0), 0.0, 1.0))
        self._combo_chain_hits = 0
        self._prev_error = None

        info["stage_name"] = self.stage_spec.name
        info["goal_type"] = self.goal_type
        info["goal_type_index"] = int(self.goal_type_index)
        info["goal_type_onehot"] = self._goal_type_onehot.copy()
        info["goal_target"] = self._goal_target.copy()
        info["goal_mask"] = self._active_mask.copy()
        info["goal_active"] = float(1.0 if self._goal_active else 0.0)
        return self._augment(obs), info

    def step(self, action: Sequence[int]):
        action_arr = np.asarray(action, dtype=np.int64).copy()
        if self.stage_spec.disable_attack:
            action_arr[3] = 0
        elif self._allowed_attack_actions is not None and int(action_arr[3]) not in self._allowed_attack_actions:
            action_arr[3] = 0

        # Action grounding: when clearly out of range, reduce attack frequency
        # to bias the policy toward spacing first, then commit.
        if action_arr.shape[0] > 3 and int(action_arr[3]) != 0:
            keep_prob = float(np.clip(self.stage_spec.attack_out_of_range_keep_prob, 0.0, 1.0))
            if keep_prob < 1.0:
                if self._prev_in_range < float(np.clip(self.stage_spec.attack_out_of_range_threshold, 0.0, 1.0)):
                    if float(np.random.rand()) > keep_prob:
                        action_arr[3] = 0

        attack_input = int(action_arr[3]) if action_arr.shape[0] > 3 else 0
        if self.stage_spec.disable_dodge:
            action_arr[2] = 0
        if self.stage_spec.disable_jump:
            action_arr[1] = 0
        if self.action_adapter is not None:
            action_arr = self.action_adapter(action_arr)

        obs, _, terminated, truncated, info = self.env.step(action_arr)
        obs = np.asarray(obs, dtype=np.float32)
        self._step_count += 1
        player_controllable = self._player_is_controllable(info)
        if player_controllable:
            self._awaiting_respawn_after_death = False

        raw_self_stock_lost = float(info.get("self_stock_lost_step", 0.0))
        is_new_death_event = raw_self_stock_lost > 0.0 and not self._awaiting_respawn_after_death
        if is_new_death_event:
            self._awaiting_respawn_after_death = True
        effective_self_stock_lost = raw_self_stock_lost if is_new_death_event else 0.0

        op_stock_lost = float(max(0.0, info.get("op_stock_lost_step", 0.0)))

        seq_enabled = self._sequential_enabled()
        seq_phase_before_step = int(self._seq_phase)
        seq_step1_transition = False
        seq_failure_penalty = 0.0

        goal_new_sampled = False
        goal_resampled_after_reward = False
        resample_due_to_stock_loss = False
        resample_due_to_timer = False
        resample_goal_after_reward = False
        resample_goal_sequence_step: Optional[int] = None
        curr_feats = self._extract(obs)  # compute once; reused for reward, resampling, and HER buffer
        if player_controllable:
            if not self._goal_active:
                if seq_enabled:
                    self._activate_goal(obs, curr_feats, sequence_step=1)
                else:
                    self._activate_goal(obs, curr_feats)
                goal_new_sampled = True
        # When player is not controllable (YOLO miss / respawning), keep the
        # existing goal active.  The goal is NEVER resampled mid-episode.
        curr_has_weapon = self._feature_value(curr_feats, "player_has_weapon", 0.0) > 0.5
        op_delta_damage = float(max(0.0, info.get("op_delta_damage", 0.0)))
        self_delta_damage = float(max(0.0, info.get("self_delta_damage", 0.0)))
        chase_bonus = 0.0
        strike_bonus = 0.0
        facing_bonus = 0.0
        hit_bonus = 0.0
        damage_bonus = 0.0
        self_damage_penalty = 0.0
        offstage_penalty = 0.0
        combo_delay_penalty = 0.0
        attack_whiff_penalty = 0.0
        attack_commit_bonus_applied = 0.0
        no_attack_in_range_penalty_applied = 0.0
        combo_chain_bonus = 0.0
        if self._goal_active and player_controllable:
            self._maybe_update_player_xy_goal(obs)
            self._active_mask = self._effective_mask_for_step(curr_feats)
            curr_error = self._goal_error_from_feats(curr_feats, self._goal_target, mask=self._active_mask)

            goal_progress = 0.0
            if self._prev_error is not None and not goal_new_sampled:
                goal_progress = float(self._prev_error - curr_error)

            if self.stage_spec.reward_from_goal_progress:
                reward = float(self.stage_spec.progress_scale * goal_progress)
                if self.stage_spec.clip_progress_reward:
                    reward = float(np.clip(reward, self.stage_spec.progress_clip_min, self.stage_spec.progress_clip_max))
            else:
                reward = - curr_error

            # Proximity penalty: consistent "closer is better" signal even without progress.
            if self.stage_spec.proximity_scale > 0.0:
                reward -= self.stage_spec.proximity_scale * curr_error

            if self.stage_spec.player_has_weapon_bonus > 0.0 and self._feature_value(curr_feats, "player_has_weapon", 0.0) > 0.5:
                reward += self.stage_spec.player_has_weapon_bonus

            # One-time bonus when agent picks up a weapon this step.
            if self.stage_spec.weapon_pickup_bonus > 0.0 and curr_has_weapon and not self._prev_has_weapon:
                reward += self.stage_spec.weapon_pickup_bonus

            # Explicit combat shaping (phase-configurable): chase, orient, connect, and trade favorably.
            rel_distance = float(np.clip(self._feature_value(curr_feats, "rel_distance", 1.0), 0.0, 1.0))
            in_range = float(np.clip(self._feature_value(curr_feats, "in_strike_range", 0.0), 0.0, 1.0))
            facing_norm = float(np.clip(self._feature_value(curr_feats, "facing_opponent", 0.5), 0.0, 1.0))
            facing_score = max(0.0, (2.0 * facing_norm) - 1.0)
            offstage = float(np.clip(self._feature_value(curr_feats, "player_is_offstage", 0.0), 0.0, 1.0))
            time_since_hit = float(np.clip(StateSpec.get(obs, "opponent_time_since_hit"), 0.0, 1.0))

            # Stage A: immediate supervision on attack timing.
            commit_threshold = float(np.clip(self.stage_spec.attack_commit_threshold, 0.0, 1.0))
            if in_range > commit_threshold:
                if attack_input != 0 and self.stage_spec.attack_commit_bonus > 0.0:
                    attack_commit_bonus_applied = float(self.stage_spec.attack_commit_bonus)
                    reward += attack_commit_bonus_applied
                elif attack_input == 0 and self.stage_spec.no_attack_in_range_penalty > 0.0:
                    no_attack_in_range_penalty_applied = float(self.stage_spec.no_attack_in_range_penalty)
                    reward -= no_attack_in_range_penalty_applied

            if self.stage_spec.chase_rel_distance_scale > 0.0:
                chase_bonus = float(self.stage_spec.chase_rel_distance_scale * (1.0 - rel_distance))
                reward += chase_bonus

            if self.stage_spec.in_strike_range_bonus > 0.0 and in_range > 0.5:
                strike_bonus = float(self.stage_spec.in_strike_range_bonus)
                reward += strike_bonus

            if self.stage_spec.facing_opponent_bonus > 0.0:
                facing_bonus = float(self.stage_spec.facing_opponent_bonus * facing_score)
                reward += facing_bonus

            if self.stage_spec.hit_event_bonus > 0.0 and op_delta_damage > 1e-6:
                hit_bonus = float(self.stage_spec.hit_event_bonus)
                reward += hit_bonus

            if self.stage_spec.damage_dealt_scale > 0.0:
                damage_bonus = float(self.stage_spec.damage_dealt_scale * op_delta_damage)
                reward += damage_bonus

            if self.stage_spec.opponent_ko_bonus > 0.0 and op_stock_lost > 0.0:
                reward += float(self.stage_spec.opponent_ko_bonus)

            if self.stage_spec.self_damage_penalty_scale > 0.0:
                self_damage_penalty = float(self.stage_spec.self_damage_penalty_scale * self_delta_damage)
                reward -= self_damage_penalty

            if self.stage_spec.offstage_penalty_scale > 0.0:
                offstage_penalty = float(self.stage_spec.offstage_penalty_scale * offstage)
                reward -= offstage_penalty

            if self.stage_spec.combo_penalty_scale > 0.0:
                combo_delay_penalty = float(self.stage_spec.combo_penalty_scale * time_since_hit)
                reward -= combo_delay_penalty

            if self.stage_spec.attack_whiff_penalty_scale > 0.0 and attack_input != 0 and op_delta_damage <= 1e-6:
                attack_whiff_penalty = float(self.stage_spec.attack_whiff_penalty_scale)
                reward -= attack_whiff_penalty

            # Stage C: reward longer hit chains.
            if op_delta_damage > 1e-6:
                self._combo_chain_hits += 1
                if self.stage_spec.combo_chain_bonus_scale > 0.0:
                    combo_chain_bonus = float(self.stage_spec.combo_chain_bonus_scale * self._combo_chain_hits)
                    reward += combo_chain_bonus
            elif time_since_hit > float(np.clip(self.stage_spec.combo_chain_reset_time_since_hit, 0.0, 1.0)):
                self._combo_chain_hits = 0

            if op_stock_lost > 0.0:
                self._combo_chain_hits = 0

            success = bool(curr_error < self.stage_spec.success_threshold)
            if success and self.stage_spec.require_attack_for_success:
                commit_threshold = float(np.clip(self.stage_spec.attack_commit_threshold, 0.0, 1.0))
                success = bool(attack_input != 0 and in_range > commit_threshold)

            if success and self.stage_spec.require_opponent_damage_for_success:
                op_dmg_idx = self._feature_index.get("opponent_damage_pct")
                if op_dmg_idx is None or self._active_mask[op_dmg_idx] <= 0.0:
                    success = False
                else:
                    op_dmg_target = float(self._goal_target[op_dmg_idx])
                    op_dmg_curr = float(curr_feats[op_dmg_idx])
                    tol = float(max(0.0, self.stage_spec.opponent_damage_success_tolerance))
                    success = bool((op_dmg_curr + tol) >= op_dmg_target)

            if seq_enabled:
                offstage_now = self._feature_value(curr_feats, "player_is_offstage", 0.0) > 0.5
                if self._seq_phase == 1:
                    step1_ok = bool(success)
                    if self.stage_spec.sequential_require_offstage_first:
                        step1_ok = bool(step1_ok and offstage_now)

                    if step1_ok:
                        if self.stage_spec.sequential_step1_bonus > 0.0:
                            reward += float(self.stage_spec.sequential_step1_bonus)
                        self._seq_step1_completed = True
                        if not (terminated or truncated):
                            resample_goal_after_reward = True
                            resample_goal_sequence_step = 2
                        seq_step1_transition = True

                    # Step-1 is never terminal success for the episode.
                    success = False
                else:
                    step2_ok = bool(success)
                    if self.stage_spec.sequential_require_onstage_second:
                        step2_ok = bool(step2_ok and (not offstage_now))
                    success = bool(step2_ok)

            if success:
                reward += self.stage_spec.success_bonus
        else:
            self._active_mask = self.mask.copy()
            curr_error = 0.0
            goal_progress = 0.0
            reward = 0.0
            success = False

        if self._goal_active and self.stage_spec.step_penalty > 0.0:
            reward -= self.stage_spec.step_penalty

        if self._goal_active and self.stage_spec.vertical_velocity_penalty_scale > 0.0 and obs.shape[0] > 3:
            reward -= self.stage_spec.vertical_velocity_penalty_scale * abs(float(obs[3]))

        if self._goal_active and self.stage_spec.jump_usage_penalty_scale > 0.0 and obs.shape[0] > 7:
            jumps_norm = float(np.clip(obs[7], 0.0, 1.0))
            jumps_used = 1.0 - jumps_norm
            reward -= self.stage_spec.jump_usage_penalty_scale * jumps_used

        # Episode ends when goal is reached.
        if self._goal_active and success:
            truncated = True

        terminated_by_hit_event = False
        if self.stage_spec.terminate_on_hit_event and op_delta_damage >= float(self.stage_spec.hit_event_damage_threshold):
            terminated = True
            terminated_by_hit_event = True

        # Velocity damping: penalise speed when close to target
        if self._goal_active and self.stage_spec.velocity_penalty_scale > 0.0 and curr_error < self.stage_spec.velocity_penalty_radius * self.stage_spec.success_threshold:
            speed = np.sqrt(obs[2] ** 2 + obs[3]** 2)  # player_vx, player_vy
            reward -= self.stage_spec.velocity_penalty_scale * speed

        # Stay bonus: reward holding position with low velocity
        if self._goal_active and self.stage_spec.stay_bonus > 0.0 and success:
            speed = np.sqrt(obs[2] ** 2 + obs[3] ** 2)
            if speed < self.stage_spec.velocity_threshold:
                reward += self.stage_spec.stay_bonus

        # Death penalty: punish losing a stock
        self_stock_lost = 0.0
        if self.stage_spec.death_penalty > 0.0:
            self_stock_lost = effective_self_stock_lost
            if self_stock_lost > 0.0:
                reward -= self.stage_spec.death_penalty

        if self.stage_spec.terminate_on_death and effective_self_stock_lost > 0.0:
            terminated = True

        if seq_enabled and seq_phase_before_step == 2 and (not success) and effective_self_stock_lost > 0.0:
            if self.stage_spec.sequential_failure_penalty > 0.0:
                seq_failure_penalty = max(seq_failure_penalty, float(self.stage_spec.sequential_failure_penalty))
            self._seq_phase = 1
            self._seq_step1_completed = False

        if seq_failure_penalty > 0.0:
            reward -= float(seq_failure_penalty)

        forced_weapon_drop = float(self._force_drop_weapon_if_needed(curr_feats, truncated))
        agent_weapon_drop_event = 0.0
        agent_weapon_drop_penalty_applied = 0.0
        if self.stage_spec.agent_weapon_drop_penalty > 0.0:
            drop_attack_action = self._attack_action_value_for_key(self.stage_spec.drop_weapon_key)
            dropped_now = bool(self._prev_has_weapon) and not curr_has_weapon
            agent_drop_input = drop_attack_action > 0 and attack_input == drop_attack_action
            if (
                dropped_now
                and agent_drop_input
                and effective_self_stock_lost <= 0.0
                and forced_weapon_drop <= 0.0
            ):
                reward -= float(self.stage_spec.agent_weapon_drop_penalty)
                agent_weapon_drop_event = 1.0
                agent_weapon_drop_penalty_applied = float(self.stage_spec.agent_weapon_drop_penalty)

        reward = float(np.clip(reward, -self.stage_spec.reward_clip, self.stage_spec.reward_clip))

        if resample_goal_after_reward:
            if resample_goal_sequence_step is None:
                self._activate_goal(obs, curr_feats)
            else:
                self._activate_goal(obs, curr_feats, sequence_step=resample_goal_sequence_step)
            goal_new_sampled = True
            goal_resampled_after_reward = True

        if not self._goal_active:
            self._prev_error = None
        elif not player_controllable:
            pass  # Preserve prev_error so progress resumes correctly when player returns
        elif goal_resampled_after_reward:
            self._prev_error = self._goal_error_from_feats(curr_feats, self._goal_target, mask=self._active_mask)
        else:
            self._prev_error = curr_error
        self._prev_has_weapon = curr_has_weapon
        self._prev_in_range = float(np.clip(self._feature_value(curr_feats, "in_strike_range", 0.0), 0.0, 1.0))

        info["stage_name"] = self.stage_spec.name
        info["goal_type"] = self.goal_type
        info["goal_type_index"] = int(self.goal_type_index)
        info["goal_type_onehot"] = self._goal_type_onehot.copy()
        info["goal_target"] = self._goal_target.copy()
        info["goal_mask"] = self._active_mask.copy()
        info["goal_error"] = float(curr_error)
        info["goal_progress"] = float(goal_progress)
        info["goal_success"] = float(1.0 if success else 0.0)
        info["goal_steps_left"] = int(self._goal_steps_left)
        info["llc_reward"] = float(reward)
        info["stage_feature_names"] = list(self.feature_names)
        info["goal_new_sampled"] = goal_new_sampled
        info["goal_active"] = float(1.0 if self._goal_active else 0.0)
        info["raw_goal_feats"] = curr_feats  # already computed above - no second call
        info["active_goal_feature_errors"] = (
            np.abs(curr_feats - self._goal_target) * self._active_mask
        ).astype(np.float32)
        info["stage_action"] = action_arr.astype(np.int64).copy()
        info["stage_action_movement"] = int(action_arr[0]) if action_arr.shape[0] > 0 else 3
        info["stage_action_jump"] = int(action_arr[1]) if action_arr.shape[0] > 1 else 0
        info["stage_action_dodge"] = int(action_arr[2]) if action_arr.shape[0] > 2 else 0
        info["stage_action_attack"] = int(action_arr[3]) if action_arr.shape[0] > 3 else 0
        info["self_stock_lost_step_raw"] = raw_self_stock_lost
        info["self_stock_lost_step_effective"] = effective_self_stock_lost
        info["death_event"] = float(1.0 if effective_self_stock_lost > 0.0 else 0.0)
        info["duplicate_death_suppressed"] = float(1.0 if raw_self_stock_lost > 0.0 and effective_self_stock_lost <= 0.0 else 0.0)
        info["terminal_success"] = float(1.0 if (self._goal_active and success) else 0.0)
        info["terminal_hit_event"] = float(1.0 if terminated_by_hit_event else 0.0)
        info["forced_weapon_drop"] = forced_weapon_drop
        info["agent_weapon_drop_event"] = agent_weapon_drop_event
        info["agent_weapon_drop_penalty"] = agent_weapon_drop_penalty_applied
        info["combat_bonus_chase"] = float(chase_bonus)
        info["combat_bonus_strike_range"] = float(strike_bonus)
        info["combat_bonus_facing"] = float(facing_bonus)
        info["combat_bonus_hit_event"] = float(hit_bonus)
        info["combat_bonus_damage_dealt"] = float(damage_bonus)
        info["combat_bonus_attack_commit"] = float(attack_commit_bonus_applied)
        info["combat_bonus_combo_chain"] = float(combo_chain_bonus)
        info["combat_penalty_self_damage"] = float(self_damage_penalty)
        info["combat_penalty_offstage"] = float(offstage_penalty)
        info["combat_penalty_combo_delay"] = float(combo_delay_penalty)
        info["combat_penalty_attack_whiff"] = float(attack_whiff_penalty)
        info["combat_penalty_no_attack_in_range"] = float(no_attack_in_range_penalty_applied)
        info["sequential_goal_enabled"] = float(1.0 if seq_enabled else 0.0)
        info["sequential_phase"] = int(self._seq_phase if seq_enabled else 0)
        info["sequential_step1_completed"] = float(1.0 if self._seq_step1_completed else 0.0)
        info["sequential_step1_transition"] = float(1.0 if seq_step1_transition else 0.0)
        info["sequential_failure_penalty"] = float(seq_failure_penalty)

        aug_obs = self._augment(obs)
        return aug_obs, reward, terminated, truncated, info


class StageDashboardCallback(BaseCallback):
    """Informative stage dashboard for learning diagnostics.

    Produces:
    - Step-level CSV (reward, goal error, progress, success and optional combat signals)
    - Episode-level CSV (returns, lengths, success ratios, average errors)
    - PNG dashboard with moving averages and trend diagnostics
    """

    def __init__(
        self,
        save_dir: Path,
        model_name: str,
        stage_spec: Optional[StageSpec] = None,
        plot_every_episodes: int = 5,
        moving_avg_window: int = 300,
        enable_csv: bool = True,
        enable_plot: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.model_name = model_name
        self.stage_spec = stage_spec  # used by on-policy HER; None disables HER
        self.plot_every_episodes = max(1, int(plot_every_episodes))
        self.moving_avg_window = max(10, int(moving_avg_window))
        self.enable_csv = bool(enable_csv)
        self.enable_plot = bool(enable_plot)

        self.step_csv = self.save_dir / f"{self.model_name}_steps.csv"
        self.episode_csv = self.save_dir / f"{self.model_name}_episodes.csv"
        self.plot_path = self.save_dir / f"{self.model_name}_dashboard.png"

        # On-policy HER buffers; accumulated each rollout, cleared in _on_rollout_end.
        self._her_raw_feats: list[np.ndarray] = []  # achieved goal feats per step (stage goal_dim)
        self._her_masks: list[np.ndarray] = []      # active mask per step (dynamic in all_skills_llc)
        self._her_new_goal: list[bool] = []          # True when goal was (re)sampled
        self._her_done: list[bool] = []              # True when episode ended at this step
        self._her_orig_rewards: list[float] = []     # original LLC reward for 50% blend

        self.step_reward: list[float] = []
        self.step_goal_error: list[float] = []
        self.step_goal_progress: list[float] = []
        self.step_goal_success: list[float] = []
        self.step_op_delta: list[float] = []
        self.step_self_delta: list[float] = []
        self.step_time_index: list[int] = []
        self._step_count = 0
        self.stage_name: str = "unknown"
        self.stage_features: list[str] = []
        self.step_idle: list[float] = []
        self.step_whiff: list[float] = []
        self.step_hit: list[float] = []

        self.ep_return: list[float] = []
        self.ep_length: list[int] = []
        self.ep_goal_error_mean: list[float] = []
        self.ep_success_ratio: list[float] = []
        self.ep_success: list[float] = []
        self.ep_op_delta_sum: list[float] = []
        self.ep_self_delta_sum: list[float] = []
        self.ep_time_to_success: list[int] = []
        self.ep_damage_trade: list[float] = []
        self.ep_action_entropy: list[float] = []
        self.ep_idle_rate: list[float] = []
        self.ep_whiff_rate: list[float] = []
        self.ep_attack_precision: list[float] = []

        self._cur_ep_reward = 0.0
        self._cur_ep_len = 0
        self._cur_ep_goal_error_sum = 0.0
        self._cur_ep_success_sum = 0.0
        self._cur_ep_had_success = False
        self._cur_ep_op_delta_sum = 0.0
        self._cur_ep_self_delta_sum = 0.0
        self._cur_ep_time_to_success: Optional[int] = None
        self._cur_ep_action_counts = np.zeros((4, 4), dtype=np.int64)
        self._cur_ep_idle_steps = 0
        self._cur_ep_attack_steps = 0
        self._cur_ep_hit_steps = 0
        self._cur_ep_whiff_steps = 0

        self._step_writer: Optional[csv.DictWriter] = None
        self._episode_writer: Optional[csv.DictWriter] = None
        self._step_fh = None
        self._episode_fh = None



    def _on_training_start(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if not self.enable_csv:
            return

        self._step_fh = open(self.step_csv, "w", newline="", encoding="utf-8")
        self._episode_fh = open(self.episode_csv, "w", newline="", encoding="utf-8")

        self._step_writer = csv.DictWriter(
            self._step_fh,
            fieldnames=[
                "step",
                "reward",
                "goal_error",
                "goal_progress",
                "goal_success",
                "op_delta_damage",
                "self_delta_damage",
                "damage_trade",
                "movement",
                "jump",
                "dodge",
                "attack",
                "idle",
                "hit",
                "whiff",
                "death_event",
                "weapon_drop_event",
                "active_feature_errors",
                "raw_goal_feats",
                "goal_target",
                "goal_mask",
                "combat_bonus_hit_event",
                "combat_bonus_damage_dealt",
                "combat_penalty_attack_whiff",
                "combat_penalty_self_damage",
                "stage_name",
                "goal_type",
                "goal_type_index",
            ],
        )
        self._step_writer.writeheader()

        self._episode_writer = csv.DictWriter(
            self._episode_fh,
            fieldnames=[
                "episode",
                "return",
                "length",
                "mean_goal_error",
                "success_ratio",
                "episode_success",
                "op_delta_damage_sum",
                "self_delta_damage_sum",
                "damage_trade",
                "time_to_success",
                "action_entropy",
                "idle_rate",
                "whiff_rate",
                "attack_precision",
            ],
        )
        self._episode_writer.writeheader()

    @staticmethod
    def _moving_average(arr: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
        if arr.size < window:
            return np.array([]), np.array([])
        kernel = np.ones(window, dtype=np.float32) / float(window)
        ma = np.convolve(arr, kernel, mode="valid")
        x = np.arange(window, arr.size + 1)
        return x, ma

    @staticmethod
    def _trend(arr: np.ndarray) -> float:
        if arr.size < 2:
            return 0.0
        x = np.arange(arr.size, dtype=np.float32)
        x = x - x.mean()
        y = arr.astype(np.float32) - arr.mean()
        denom = float(np.dot(x, x))
        if denom <= 1e-8:
            return 0.0
        return float(np.dot(x, y) / denom)

    @staticmethod
    def _json_array(value: Any, precision: int = 4) -> str:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        rounded = [round(float(v), precision) for v in arr.tolist()]
        return json.dumps(rounded, separators=(",", ":"))

    @staticmethod
    def _action_entropy(counts: np.ndarray) -> float:
        flat = np.asarray(counts, dtype=np.float64).reshape(-1)
        total = float(flat.sum())
        if total <= 0.0:
            return 0.0
        probs = flat / total
        probs = probs[probs > 0.0]
        denom = np.log(max(2, flat.size))
        return float(-np.sum(probs * np.log(probs)) / denom)

    def _reset_episode_accumulators(self) -> None:
        self._cur_ep_reward = 0.0
        self._cur_ep_len = 0
        self._cur_ep_goal_error_sum = 0.0
        self._cur_ep_success_sum = 0.0
        self._cur_ep_had_success = False
        self._cur_ep_op_delta_sum = 0.0
        self._cur_ep_self_delta_sum = 0.0
        self._cur_ep_time_to_success = None
        self._cur_ep_action_counts.fill(0)
        self._cur_ep_idle_steps = 0
        self._cur_ep_attack_steps = 0
        self._cur_ep_hit_steps = 0
        self._cur_ep_whiff_steps = 0

    def _plot_dashboard(self) -> None:
        if not self.enable_plot:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        if len(self.step_reward) < 10:
            return

        step_idx = np.asarray(self.step_time_index, dtype=np.int32)
        rewards = np.asarray(self.step_reward, dtype=np.float32)
        errors = np.asarray(self.step_goal_error, dtype=np.float32)
        progress = np.asarray(self.step_goal_progress, dtype=np.float32)
        op_delta = np.asarray(self.step_op_delta, dtype=np.float32)
        self_delta = np.asarray(self.step_self_delta, dtype=np.float32)

        win = int(min(self.moving_avg_window, max(20, len(rewards) // 8)))

        fig, axes = plt.subplots(3, 2, figsize=(16, 11))
        axes = axes.ravel()

        axes[0].plot(step_idx, rewards, alpha=0.25, label="reward/step")
        x_ma, y_ma = self._moving_average(rewards, win)
        if x_ma.size:
            axes[0].plot(x_ma, y_ma, linewidth=2, label=f"reward MA({win})")
        axes[0].set_title("Reward Dynamics")
        axes[0].set_xlabel("Step")
        axes[0].legend(loc="best")
        axes[0].grid(alpha=0.25)

        axes[1].plot(step_idx, errors, alpha=0.25, color="tab:red", label="goal_error/step")
        x_ma, y_ma = self._moving_average(errors, win)
        if x_ma.size:
            axes[1].plot(x_ma, y_ma, color="tab:red", linewidth=2, label=f"goal_error MA({win})")
        axes[1].set_title("Goal Error")
        axes[1].set_xlabel("Step")
        axes[1].legend(loc="best")
        axes[1].grid(alpha=0.25)

        axes[2].plot(step_idx, progress, alpha=0.25, color="tab:green", label="goal_progress/step")
        x_ma, y_ma = self._moving_average(progress, win)
        if x_ma.size:
            axes[2].plot(x_ma, y_ma, color="tab:green", linewidth=2, label=f"progress MA({win})")
        axes[2].axhline(0.0, linestyle="--", linewidth=1)
        axes[2].set_title("Goal Progress")
        axes[2].set_xlabel("Step")
        axes[2].legend(loc="best")
        axes[2].grid(alpha=0.25)

        ep_success = np.asarray(self.ep_success, dtype=np.float32)
        if ep_success.size > 0:
            epi_success_idx = np.arange(1, ep_success.size + 1)
            ep_win = int(min(30, max(5, len(ep_success) // 6)))
            ex, ey = self._moving_average(ep_success, ep_win)
            if ex.size:
                axes[3].plot(ex, ey, linewidth=2, color="tab:purple", label=f"episode_success MA({ep_win})")
            cumulative_success = np.cumsum(ep_success) / np.maximum(1.0, np.arange(1, len(ep_success) + 1, dtype=np.float32))
            axes[3].plot(epi_success_idx, cumulative_success, alpha=0.65, label="cumulative episode success")
        else:
            axes[3].text(0.5, 0.5, "Waiting for completed episodes", ha="center", va="center", transform=axes[3].transAxes)
        axes[3].set_ylim(-0.02, 1.02)
        axes[3].set_title("Episode Success Rate")
        axes[3].set_xlabel("Episode")
        axes[3].legend(loc="best")
        axes[3].grid(alpha=0.25)

        if len(self.ep_return) > 0:
            epi = np.arange(1, len(self.ep_return) + 1)
            axes[4].plot(epi, np.asarray(self.ep_return, dtype=np.float32), alpha=0.30, label="episode return")
            ep_ret = np.asarray(self.ep_return, dtype=np.float32)
            ep_win = int(min(30, max(5, len(ep_ret) // 6)))
            ex, ey = self._moving_average(ep_ret, ep_win)
            if ex.size:
                axes[4].plot(ex, ey, linewidth=2, label=f"return MA({ep_win})")
            axes[4].set_title("Episode Return")
            axes[4].set_xlabel("Episode")
            axes[4].legend(loc="best")
            axes[4].grid(alpha=0.25)

        reward_trend = self._trend(np.asarray(rewards[-win:], dtype=np.float32))
        error_trend = self._trend(np.asarray(errors[-win:], dtype=np.float32))
        progress_trend = self._trend(np.asarray(progress[-win:], dtype=np.float32))
        ep_success = np.asarray(self.ep_success, dtype=np.float32)
        if ep_success.size > 0:
            ep_recent_win = int(min(30, max(5, len(ep_success) // 6)))
            success_recent = float(np.mean(ep_success[-ep_recent_win:]))
        else:
            success_recent = 0.0
        op_recent = float(np.mean(op_delta[-win:])) if len(op_delta) >= win else float(np.mean(op_delta))
        self_recent = float(np.mean(self_delta[-win:])) if len(self_delta) >= win else float(np.mean(self_delta))

        diagnostics: list[str] = []
        diagnostics.append(f"Stage: {self.stage_name}")
        # diagnostics.append(f"Features: {', '.join(self.stage_features) if self.stage_features else 'n/a'}")
        diagnostics.append(f"Recent episode success rate: {success_recent:.3f}")
        diagnostics.append(f"Reward trend (recent): {reward_trend:+.5f}")
        diagnostics.append(f"Error trend (recent): {error_trend:+.5f}")
        diagnostics.append(f"Progress trend (recent): {progress_trend:+.5f}")
        diagnostics.append(f"Recent op/self damage delta: {op_recent:.4f} / {self_recent:.4f}")

        if reward_trend < 0 and error_trend > 0:
            diagnostics.append("Warning: reward declining while error rises (likely policy drift).")
        if len(self.ep_success) >= 20 and success_recent < 0.20:
            diagnostics.append("Warning: low episode success rate (consider easier goals / larger success region).")
        if abs(progress_trend) < 1e-4:
            diagnostics.append("Warning: near-zero progress trend (possible local optimum or weak exploration).")

        axes[5].axis("off")
        axes[5].set_title("Critical Diagnostics", loc="left")
        axes[5].text(
            0.01,
            0.98,
            "\n".join(diagnostics),
            va="top",
            ha="left",
            family="monospace",
            fontsize=10,
        )

        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=120)
        plt.close(fig)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        if len(infos) == 0:
            return True

        for i, info in enumerate(infos):
            reward = float(rewards[i]) if i < len(rewards) else float(info.get("llc_reward", 0.0))
            goal_error = float(info.get("goal_error", 0.0))
            goal_progress = float(info.get("goal_progress", 0.0))
            goal_success = float(info.get("goal_success", 0.0))
            op_delta = float(info.get("op_delta_damage", 0.0))
            self_delta = float(info.get("self_delta_damage", 0.0))
            damage_trade = float(op_delta - self_delta)
            stage_action = np.asarray(info.get("stage_action", [3, 0, 0, 0]), dtype=np.int64).reshape(-1)
            movement = int(stage_action[0]) if stage_action.shape[0] > 0 else 3
            jump = int(stage_action[1]) if stage_action.shape[0] > 1 else 0
            dodge = int(stage_action[2]) if stage_action.shape[0] > 2 else 0
            attack = int(stage_action[3]) if stage_action.shape[0] > 3 else 0
            idle = 1.0 if movement == 3 and jump == 0 and dodge == 0 and attack == 0 else 0.0
            hit = 1.0 if op_delta > 1e-6 else 0.0
            whiff = 1.0 if attack != 0 and hit <= 0.0 else 0.0
            death_event = float(info.get("death_event", 0.0))
            weapon_drop_event = float(info.get("agent_weapon_drop_event", 0.0))

            self.stage_name = str(info.get("stage_name", self.stage_name))
            stage_features = info.get("stage_feature_names")
            if isinstance(stage_features, list):
                self.stage_features = [str(v) for v in stage_features]

            step_number = self._step_count + 1
            self._step_count = step_number
            if self.enable_plot:
                self.step_time_index.append(step_number)
                self.step_reward.append(reward)
                self.step_goal_error.append(goal_error)
                self.step_goal_progress.append(goal_progress)
                self.step_goal_success.append(goal_success)
                self.step_op_delta.append(op_delta)
                self.step_self_delta.append(self_delta)
                self.step_idle.append(idle)
                self.step_hit.append(hit)
                self.step_whiff.append(whiff)

            if self._step_writer is not None:
                self._step_writer.writerow(
                    {
                        "step": step_number,
                        "reward": reward,
                        "goal_error": goal_error,
                        "goal_progress": goal_progress,
                        "goal_success": goal_success,
                        "op_delta_damage": op_delta,
                        "self_delta_damage": self_delta,
                        "damage_trade": damage_trade,
                        "movement": movement,
                        "jump": jump,
                        "dodge": dodge,
                        "attack": attack,
                        "idle": idle,
                        "hit": hit,
                        "whiff": whiff,
                        "death_event": death_event,
                        "weapon_drop_event": weapon_drop_event,
                        "active_feature_errors": self._json_array(info.get("active_goal_feature_errors", [])),
                        "raw_goal_feats": self._json_array(info.get("raw_goal_feats", [])),
                        "goal_target": self._json_array(info.get("goal_target", [])),
                        "goal_mask": self._json_array(info.get("goal_mask", [])),
                        "combat_bonus_hit_event": float(info.get("combat_bonus_hit_event", 0.0)),
                        "combat_bonus_damage_dealt": float(info.get("combat_bonus_damage_dealt", 0.0)),
                        "combat_penalty_attack_whiff": float(info.get("combat_penalty_attack_whiff", 0.0)),
                        "combat_penalty_self_damage": float(info.get("combat_penalty_self_damage", 0.0)),
                        "stage_name": self.stage_name,
                        "goal_type": str(info.get("goal_type", "unknown")),
                        "goal_type_index": int(info.get("goal_type_index", -1)),
                    }
                )

            self._cur_ep_reward += reward
            self._cur_ep_len += 1
            self._cur_ep_goal_error_sum += goal_error
            self._cur_ep_success_sum += goal_success
            if goal_success > 0.5 or float(info.get("terminal_success", 0.0)) > 0.5:
                self._cur_ep_had_success = True
                if self._cur_ep_time_to_success is None:
                    self._cur_ep_time_to_success = self._cur_ep_len
            self._cur_ep_op_delta_sum += op_delta
            self._cur_ep_self_delta_sum += self_delta
            self._cur_ep_action_counts[0, int(np.clip(movement, 0, 3))] += 1
            self._cur_ep_action_counts[1, int(np.clip(jump, 0, 3))] += 1
            self._cur_ep_action_counts[2, int(np.clip(dodge, 0, 3))] += 1
            self._cur_ep_action_counts[3, int(np.clip(attack, 0, 3))] += 1
            self._cur_ep_idle_steps += int(idle > 0.5)
            self._cur_ep_attack_steps += int(attack != 0)
            self._cur_ep_hit_steps += int(hit > 0.5)
            self._cur_ep_whiff_steps += int(whiff > 0.5)

            done = bool(dones[i]) if i < len(dones) else False

            # On-policy HER: accumulate per-step data for rollout-end relabeling.
            _raw_feats = info.get("raw_goal_feats")
            self._her_raw_feats.append(
                np.asarray(_raw_feats, dtype=np.float32).copy()
                if _raw_feats is not None
                else np.zeros(int(len(self.stage_spec.mask)) if self.stage_spec is not None else GOAL_TARGET_DIM, dtype=np.float32)
            )
            _goal_mask = info.get("goal_mask")
            self._her_masks.append(
                np.asarray(_goal_mask, dtype=np.float32).copy()
                if _goal_mask is not None
                else np.asarray(self.stage_spec.mask if self.stage_spec is not None else np.ones(GOAL_TARGET_DIM), dtype=np.float32)
            )
            self._her_new_goal.append(bool(info.get("goal_new_sampled", False)))
            self._her_done.append(done)
            self._her_orig_rewards.append(reward)

            if done:
                ep_idx = len(self.ep_return) + 1
                ep_len = max(1, self._cur_ep_len)
                mean_err = self._cur_ep_goal_error_sum / float(ep_len)
                episode_success = 1.0 if self._cur_ep_had_success else 0.0
                # Keep the historical step-wise metric but ensure terminal success
                # is counted as full success at episode level.
                success_ratio = max(self._cur_ep_success_sum / float(ep_len), episode_success)
                damage_trade_sum = float(self._cur_ep_op_delta_sum - self._cur_ep_self_delta_sum)
                time_to_success = int(self._cur_ep_time_to_success or 0)
                action_entropy = self._action_entropy(self._cur_ep_action_counts)
                idle_rate = float(self._cur_ep_idle_steps / float(ep_len))
                whiff_rate = float(self._cur_ep_whiff_steps / float(max(1, self._cur_ep_attack_steps)))
                attack_precision = float(self._cur_ep_hit_steps / float(max(1, self._cur_ep_attack_steps)))

                self.ep_return.append(float(self._cur_ep_reward))
                self.ep_length.append(int(self._cur_ep_len))
                self.ep_goal_error_mean.append(float(mean_err))
                self.ep_success_ratio.append(float(success_ratio))
                self.ep_success.append(float(episode_success))
                self.ep_op_delta_sum.append(float(self._cur_ep_op_delta_sum))
                self.ep_self_delta_sum.append(float(self._cur_ep_self_delta_sum))
                self.ep_time_to_success.append(time_to_success)
                self.ep_damage_trade.append(damage_trade_sum)
                self.ep_action_entropy.append(action_entropy)
                self.ep_idle_rate.append(idle_rate)
                self.ep_whiff_rate.append(whiff_rate)
                self.ep_attack_precision.append(attack_precision)

                if self._episode_writer is not None:
                    self._episode_writer.writerow(
                        {
                            "episode": ep_idx,
                            "return": self._cur_ep_reward,
                            "length": self._cur_ep_len,
                            "mean_goal_error": mean_err,
                            "success_ratio": success_ratio,
                            "episode_success": episode_success,
                            "op_delta_damage_sum": self._cur_ep_op_delta_sum,
                            "self_delta_damage_sum": self._cur_ep_self_delta_sum,
                            "damage_trade": damage_trade_sum,
                            "time_to_success": time_to_success,
                            "action_entropy": action_entropy,
                            "idle_rate": idle_rate,
                            "whiff_rate": whiff_rate,
                            "attack_precision": attack_precision,
                        }
                    )

                self._reset_episode_accumulators()

                if self.enable_plot and ep_idx % self.plot_every_episodes == 0:
                    self._plot_dashboard()

        return True

    def _on_rollout_end(self) -> None:
        """On-policy HER: relabel goal epochs with hindsight achieved goals.

        Called by SB3 BEFORE compute_returns_and_advantage(), so patched
        rewards propagate correctly into GAE advantage estimates.
        Flushes CSV buffers here instead of per-step to avoid disk I/O stalls.
        """
        if self._step_fh is not None:
            self._step_fh.flush()
        if self._episode_fh is not None:
            self._episode_fh.flush()
        spec = self.stage_spec
        n = len(self._her_raw_feats)
        if spec is None or n == 0:
            self._her_raw_feats.clear()
            self._her_masks.clear()
            self._her_new_goal.clear()
            self._her_done.clear()
            self._her_orig_rewards.clear()
            return

        model_obj = cast(Any, self.model)
        buffer = getattr(model_obj, "rollout_buffer", None)
        if buffer is None:
            self._her_raw_feats.clear()
            self._her_masks.clear()
            self._her_new_goal.clear()
            self._her_done.clear()
            self._her_orig_rewards.clear()
            return

        # Walk steps to identify goal-epoch boundaries.
        # A new epoch starts when: t==0, previous step was done, or goal was resampled.
        epoch_start = 0
        for t in range(n):
            is_new_epoch = (t == 0) or self._her_done[t - 1] or self._her_new_goal[t]
            if is_new_epoch and t > epoch_start:
                self._her_relabel_epoch(buffer, spec, epoch_start, t)
                epoch_start = t
        # Final epoch
        if epoch_start < n:
            self._her_relabel_epoch(buffer, spec, epoch_start, n)

        self._her_raw_feats.clear()
        self._her_masks.clear()
        self._her_new_goal.clear()
        self._her_done.clear()
        self._her_orig_rewards.clear()

    def _her_relabel_epoch(
        self,
        buffer,
        spec: StageSpec,
        t_start: int,
        t_end: int,
    ) -> None:
        """Relabel one goal epoch [t_start, t_end) with hindsight achieved goal.

        Strategy: final state of epoch becomes the retrospective target.
        Blend: 50% original reward + 50% HER reward (preserves on-policy validity).
        """
        if t_end <= t_start + 1:
            return

        # Hindsight goal = the stage-goal state where the agent actually ended up.
        achieved = self._her_raw_feats[t_end - 1]

        prev_her_error: Optional[float] = None
        for t in range(t_start, t_end):
            feats = self._her_raw_feats[t]
            mask = np.asarray(self._her_masks[t], dtype=np.float32)
            curr_her_error = float(np.sum(mask * np.abs(feats - achieved)))

            her_progress = 0.0 if prev_her_error is None else (prev_her_error - curr_her_error)
            her_reward = float(np.clip(
                spec.progress_scale * her_progress,
                spec.progress_clip_min,
                spec.progress_clip_max,
            ))
            if curr_her_error < spec.success_threshold:
                her_reward += spec.success_bonus
            her_reward = float(np.clip(her_reward, -spec.reward_clip, spec.reward_clip))

            # 50/50 blend with original reward.
            buffer.rewards[t, 0] = float(0.5 * self._her_orig_rewards[t] + 0.5 * her_reward)
            prev_her_error = curr_her_error

    def _on_training_end(self) -> None:
        if self.enable_plot:
            self._plot_dashboard()
        if self._step_fh is not None:
            self._step_fh.close()
        if self._episode_fh is not None:
            self._episode_fh.close()


class DiagnosticCallback(BaseCallback):
    """Lightweight diagnostic callback that prints key signals every N steps.

    Logs: raw obs sample, VecNormalize running mean/var, reward stats,
    action distribution, and goal-error statistics.  Useful to verify
    the training loop is sane before long runs.
    """

    def __init__(self, report_every: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self._report_every = max(1, report_every)
        self._rewards: list[float] = []
        self._errors: list[float] = []
        self._actions: list[np.ndarray] = []
        self._successes: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        actions = self.locals.get("actions")

        for i, info in enumerate(infos):
            self._rewards.append(float(rewards[i]) if i < len(rewards) else 0.0)
            self._errors.append(float(info.get("goal_error", 0.0)))
            self._successes.append(float(info.get("goal_success", 0.0)))
            if actions is not None and i < len(actions):
                self._actions.append(np.asarray(actions[i]))

        if self.num_timesteps % self._report_every == 0 and len(self._rewards) > 0:
            r = np.asarray(self._rewards[-self._report_every:], dtype=np.float32)
            e = np.asarray(self._errors[-self._report_every:], dtype=np.float32)
            s = np.asarray(self._successes[-self._report_every:], dtype=np.float32)

            # Action distribution (movement axis = index 0)
            act_str = ""
            if len(self._actions) >= self._report_every:
                recent_acts = np.stack(self._actions[-self._report_every:])
                move_counts = np.bincount(recent_acts[:, 0].astype(int), minlength=4)
                move_pct = 100.0 * move_counts / max(1, move_counts.sum())
                act_str = (
                    f"move=[L:{move_pct[0]:.0f}% R:{move_pct[1]:.0f}% "
                    f"D:{move_pct[2]:.0f}% idle:{move_pct[3]:.0f}%]"
                )

            # VecNormalize stats
            norm_str = ""
            try:
                vecn = self.model.get_vec_normalize_env()
                if vecn is not None and hasattr(vecn, "obs_rms"):
                    rms = vecn.obs_rms
                    if isinstance(rms, dict):
                        rms = next(iter(rms.values()), None)
                    if rms is not None:
                        mean = rms.mean
                        var = rms.var
                        norm_str = (
                            f"VecNorm mean[0:3]={mean[:3].round(3)} "
                            f"var[0:3]={var[:3].round(3)}"
                        )
            except Exception:
                pass

            print(
                f"[Diag @{self.num_timesteps}] "
                f"reward: {r.mean():.4f}+/-{r.std():.4f} "
                f"[{r.min():.3f}, {r.max():.3f}] | "
                f"error: {e.mean():.4f}+/-{e.std():.4f} | "
                f"success: {s.mean():.3f} | "
                f"{act_str} | {norm_str}"
            )

            # Sample one raw obs from the current step
            new_obs = self.locals.get("new_obs")
            if new_obs is not None:
                if isinstance(new_obs, dict):
                    # Dict obs from VecEnv: each value has shape (n_envs, ...)
                    obs0 = {k: v[0] for k, v in new_obs.items()}
                    o = obs0["observation"]
                    print(
                        f"  obs[0:6]={o[:6].round(3)} "
                        f"desired_goal={obs0['desired_goal'].round(3)} "
                        f"achieved_goal={obs0['achieved_goal'].round(3)}"
                    )
                elif len(new_obs) > 0:
                    obs0 = new_obs[0]
                    base = StateSpec.dim()
                    g = GOAL_TARGET_DIM
                    if len(infos) > 0 and isinstance(infos[0].get("goal_target"), np.ndarray):
                        g = int(np.asarray(infos[0].get("goal_target"), dtype=np.float32).shape[0])
                    print(
                        f"  raw_obs[0:6]={obs0[:6].round(3)} "
                        f"goal_target={obs0[base:base+g].round(3)} "
                        f"mask={obs0[base+g:base+2*g].round(3)}"
                    )

        return True


class PeriodicEvalCallback(BaseCallback):
    """Run periodic evaluation episodes during training without stopping learning."""

    def __init__(
        self,
        make_eval_env: Callable[[], gym.Env],
        eval_freq_steps: int,
        eval_episodes: int = 5,
        deterministic: bool = True,
        seed: int = 42,
        save_dir: Optional[Path] = None,
        model_name: str = "model",
        current_phase: str = "",
        eval_phases: Optional[Sequence[str]] = None,
        make_eval_env_for_phase: Optional[Callable[[str], gym.Env]] = None,
        amnesia_threshold: float = 0.15,
        retention_scores_path: Optional[Path] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.make_eval_env = make_eval_env
        self.make_eval_env_for_phase = make_eval_env_for_phase
        self.eval_freq_steps = max(1, int(eval_freq_steps))
        self.eval_episodes = max(1, int(eval_episodes))
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.save_dir = save_dir
        self.model_name = str(model_name)
        self.current_phase = str(current_phase or model_name).strip().lower()
        self.eval_phases = list(eval_phases or [self.current_phase])
        self.amnesia_threshold = float(max(0.0, amnesia_threshold))

        self._next_eval_step = self.eval_freq_steps
        self._eval_index = 0
        self._eval_env: Optional[gym.Env] = None
        self._eval_envs: dict[str, gym.Env] = {}

        self.best_mean_reward = float("-inf")
        self.best_model_path = (
            (self.save_dir / f"{self.model_name}_best_eval.zip")
            if self.save_dir is not None
            else None
        )

        self.eval_csv_path = (
            (self.save_dir / f"{self.model_name}_eval.csv")
            if self.save_dir is not None
            else None
        )
        self._eval_fh = None
        self._eval_writer: Optional[csv.DictWriter] = None
        if retention_scores_path is not None:
            self.retention_path = Path(retention_scores_path)
        elif self.save_dir is not None:
            self.retention_path = self.save_dir / "llc_retention_best.json"
        else:
            self.retention_path = None
        self._best_scores: dict[str, float] = {}

    @staticmethod
    def _resolve_outcome(self_stocks: float, op_stocks: float, truncated: bool) -> str:
        if op_stocks <= 0.0 and self_stocks > 0.0:
            return "WIN"
        if self_stocks <= 0.0 and op_stocks > 0.0:
            return "LOSS"
        if self_stocks <= 0.0 and op_stocks <= 0.0:
            return "DRAW"
        if truncated:
            return "TRUNC"
        return "UNRESOLVED"

    @staticmethod
    def _prepare_action(env: gym.Env, action: Any) -> Any:
        if isinstance(action, np.ndarray) and action.ndim > 1:
            action = action[0]

        action_space = env.action_space
        if isinstance(action_space, gym.spaces.Discrete):
            return int(np.asarray(action, dtype=np.int64).reshape(-1)[0])

        if isinstance(action_space, gym.spaces.MultiDiscrete):
            arr = np.asarray(action, dtype=np.int64).reshape(-1)
            expected_dim = int(np.asarray(action_space.nvec).reshape(-1).shape[0])
            if arr.shape[0] != expected_dim:
                raise ValueError(
                    f"Evaluation action shape mismatch: expected {expected_dim}, got {arr.shape[0]}"
                )
            return arr

        return action

    @staticmethod
    def _extract_base_env(env: gym.Env) -> Optional[BrawlDeepEnv]:
        base = env.unwrapped
        return base if isinstance(base, BrawlDeepEnv) else None

    def _on_training_start(self) -> None:
        self._eval_envs = {}
        for phase in self.eval_phases:
            phase_key = str(phase).strip().lower()
            if not phase_key:
                continue
            if self.make_eval_env_for_phase is not None:
                self._eval_envs[phase_key] = self.make_eval_env_for_phase(phase_key)
            else:
                self._eval_envs[phase_key] = self.make_eval_env()
        if not self._eval_envs:
            self._eval_envs[self.current_phase] = self.make_eval_env()
        self._eval_env = self._eval_envs.get(self.current_phase) or next(iter(self._eval_envs.values()))
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.retention_path is not None:
            self._best_scores = load_best_scores(self.retention_path)

        if self.eval_csv_path is None:
            return

        self._eval_fh = open(self.eval_csv_path, "w", newline="", encoding="utf-8")
        self._eval_writer = csv.DictWriter(
            self._eval_fh,
            fieldnames=[
                "eval_index",
                "train_steps",
                "phase",
                "episodes",
                "mean_reward",
                "std_reward",
                "mean_steps",
                "mean_goal_error",
                "mean_goal_success",
                "episode_success_rate",
                "hit_rate",
                "whiff_rate",
                "idle_rate",
                "attack_precision",
                "weapon_pickup_rate",
                "mean_damage_trade",
                "skill_score",
                "best_skill_score",
                "retention",
                "amnesia",
                "amnesia_gate_pass",
                "mean_op_damage",
                "mean_self_damage",
                "mean_self_stocks",
                "mean_op_stocks",
                "wins",
                "losses",
                "draws",
                "truncs",
                "win_rate",
            ],
        )
        self._eval_writer.writeheader()

    def _run_eval(self, phase: str, env: gym.Env) -> dict[str, float | int | str]:
        rewards: list[float] = []
        lengths: list[int] = []
        self_stocks_list: list[float] = []
        op_stocks_list: list[float] = []
        op_dmg_totals: list[float] = []
        self_dmg_totals: list[float] = []
        goal_error_means: list[float] = []
        goal_success_means: list[float] = []
        episode_successes: list[float] = []
        hit_rates: list[float] = []
        whiff_rates: list[float] = []
        idle_rates: list[float] = []
        attack_precisions: list[float] = []
        weapon_pickups: list[float] = []
        outcomes: list[str] = []

        for ep in range(self.eval_episodes):
            eval_seed = int(self.seed + (self._eval_index * 10_000) + ep + 1)
            obs, _ = env.reset(seed=eval_seed)
            terminated = False
            truncated = False
            ep_reward = 0.0
            ep_len = 0
            ep_op_dmg = 0.0
            ep_self_dmg = 0.0
            ep_goal_error_sum = 0.0
            ep_goal_success_sum = 0.0
            ep_had_success = False
            ep_attack_steps = 0
            ep_hit_steps = 0
            ep_whiff_steps = 0
            ep_idle_steps = 0
            ep_weapon_pickup = False
            prev_has_weapon = False

            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                env_action = self._prepare_action(env, action)
                obs, reward, terminated, truncated, info = env.step(env_action)
                ep_reward += float(reward)
                ep_len += 1
                op_delta = float(info.get("op_delta_damage", 0.0))
                self_delta = float(info.get("self_delta_damage", 0.0))
                ep_op_dmg += op_delta
                ep_self_dmg += self_delta
                ep_goal_error_sum += float(info.get("goal_error", 0.0))
                goal_success = float(info.get("goal_success", 0.0))
                ep_goal_success_sum += goal_success
                ep_had_success = bool(ep_had_success or goal_success > 0.5 or float(info.get("terminal_success", 0.0)) > 0.5)

                stage_action = np.asarray(info.get("stage_action", env_action), dtype=np.int64).reshape(-1)
                movement = int(stage_action[0]) if stage_action.shape[0] > 0 else 3
                jump = int(stage_action[1]) if stage_action.shape[0] > 1 else 0
                dodge = int(stage_action[2]) if stage_action.shape[0] > 2 else 0
                attack = int(stage_action[3]) if stage_action.shape[0] > 3 else 0
                ep_idle_steps += int(movement == 3 and jump == 0 and dodge == 0 and attack == 0)
                ep_attack_steps += int(attack != 0)
                ep_hit_steps += int(op_delta > 1e-6)
                ep_whiff_steps += int(attack != 0 and op_delta <= 1e-6)

                raw_feats = np.asarray(info.get("raw_goal_feats", []), dtype=np.float32).reshape(-1)
                feature_names = list(info.get("stage_feature_names", []))
                if "player_has_weapon" in feature_names:
                    idx = feature_names.index("player_has_weapon")
                    has_weapon = bool(idx < raw_feats.shape[0] and raw_feats[idx] > 0.5)
                    ep_weapon_pickup = bool(ep_weapon_pickup or (has_weapon and not prev_has_weapon))
                    prev_has_weapon = has_weapon

            base = self._extract_base_env(env)
            if base is not None:
                self_stocks = float(base.memory.self_stocks_left)
                op_stocks = float(base.memory.op_stocks_left)
            else:
                self_stocks = float("nan")
                op_stocks = float("nan")

            outcome = self._resolve_outcome(self_stocks, op_stocks, bool(truncated))

            rewards.append(ep_reward)
            lengths.append(ep_len)
            self_stocks_list.append(self_stocks)
            op_stocks_list.append(op_stocks)
            op_dmg_totals.append(ep_op_dmg)
            self_dmg_totals.append(ep_self_dmg)
            goal_error_means.append(float(ep_goal_error_sum / max(1, ep_len)))
            goal_success_means.append(float(ep_goal_success_sum / max(1, ep_len)))
            episode_successes.append(1.0 if ep_had_success else 0.0)
            hit_rates.append(float(ep_hit_steps / max(1, ep_len)))
            whiff_rates.append(float(ep_whiff_steps / max(1, ep_attack_steps)))
            idle_rates.append(float(ep_idle_steps / max(1, ep_len)))
            attack_precisions.append(float(ep_hit_steps / max(1, ep_attack_steps)))
            weapon_pickups.append(1.0 if ep_weapon_pickup else 0.0)
            outcomes.append(outcome)

        rewards_np = np.asarray(rewards, dtype=np.float32)
        lengths_np = np.asarray(lengths, dtype=np.float32)
        self_stocks_np = np.asarray(self_stocks_list, dtype=np.float32)
        op_stocks_np = np.asarray(op_stocks_list, dtype=np.float32)
        op_dmg_np = np.asarray(op_dmg_totals, dtype=np.float32)
        self_dmg_np = np.asarray(self_dmg_totals, dtype=np.float32)
        goal_err_np = np.asarray(goal_error_means, dtype=np.float32)
        goal_succ_np = np.asarray(goal_success_means, dtype=np.float32)
        ep_success_np = np.asarray(episode_successes, dtype=np.float32)
        hit_np = np.asarray(hit_rates, dtype=np.float32)
        whiff_np = np.asarray(whiff_rates, dtype=np.float32)
        idle_np = np.asarray(idle_rates, dtype=np.float32)
        precision_np = np.asarray(attack_precisions, dtype=np.float32)
        weapon_np = np.asarray(weapon_pickups, dtype=np.float32)

        wins = int(sum(1 for x in outcomes if x == "WIN"))
        losses = int(sum(1 for x in outcomes if x == "LOSS"))
        draws = int(sum(1 for x in outcomes if x == "DRAW"))
        truncs = int(sum(1 for x in outcomes if x == "TRUNC"))
        win_rate = float(wins / max(1, len(outcomes)))
        damage_trade_np = op_dmg_np - self_dmg_np

        return {
            "phase": str(phase),
            "episodes": int(len(outcomes)),
            "mean_reward": float(rewards_np.mean()),
            "std_reward": float(rewards_np.std()),
            "mean_steps": float(lengths_np.mean()),
            "mean_goal_error": float(goal_err_np.mean()),
            "mean_goal_success": float(goal_succ_np.mean()),
            "episode_success_rate": float(ep_success_np.mean()),
            "hit_rate": float(hit_np.mean()),
            "whiff_rate": float(whiff_np.mean()),
            "idle_rate": float(idle_np.mean()),
            "attack_precision": float(precision_np.mean()),
            "weapon_pickup_rate": float(weapon_np.mean()),
            "mean_damage_trade": float(damage_trade_np.mean()),
            "mean_op_damage": float(op_dmg_np.mean()),
            "mean_self_damage": float(self_dmg_np.mean()),
            "mean_self_stocks": float(self_stocks_np.mean()),
            "mean_op_stocks": float(op_stocks_np.mean()),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "truncs": truncs,
            "win_rate": win_rate,
        }

    def _log_eval_summary(self, summary: dict[str, float | int]) -> None:
        phase = str(summary.get("phase", "phase"))
        prefix = f"eval/{phase}"
        self.logger.record(f"{prefix}/mean_reward", float(summary["mean_reward"]))
        self.logger.record(f"{prefix}/std_reward", float(summary["std_reward"]))
        self.logger.record(f"{prefix}/mean_ep_length", float(summary["mean_steps"]))
        self.logger.record(f"{prefix}/win_rate", float(summary["win_rate"]))
        self.logger.record(f"{prefix}/episode_success_rate", float(summary.get("episode_success_rate", 0.0)))
        self.logger.record(f"{prefix}/skill_score", float(summary.get("skill_score", 0.0)))
        self.logger.record(f"{prefix}/amnesia", float(summary.get("amnesia", 0.0)))
        self.logger.record(f"{prefix}/mean_op_damage", float(summary["mean_op_damage"]))
        self.logger.record(f"{prefix}/mean_self_damage", float(summary["mean_self_damage"]))

    def _on_step(self) -> bool:
        while self.num_timesteps >= self._next_eval_step:
            self._eval_index += 1
            current_scores: dict[str, float] = {}

            for phase, env in self._eval_envs.items():
                summary = self._run_eval(phase, env)
                score = skill_score_for_phase(phase, summary)
                best_ref = max(float(self._best_scores.get(phase, 0.0)), score)
                retention, amnesia = retention_and_amnesia(score, best_ref)
                gate_pass = float(amnesia <= self.amnesia_threshold)
                summary["skill_score"] = float(score)
                summary["best_skill_score"] = float(best_ref)
                summary["retention"] = float(retention)
                summary["amnesia"] = float(amnesia)
                summary["amnesia_gate_pass"] = gate_pass
                current_scores[phase] = float(score)

                self._log_eval_summary(summary)

                mean_reward = float(summary["mean_reward"])
                improved = bool(phase == self.current_phase and mean_reward > self.best_mean_reward)
                if improved:
                    self.best_mean_reward = mean_reward
                    if self.best_model_path is not None:
                        self.model.save(str(self.best_model_path))

                if self._eval_writer is not None:
                    self._eval_writer.writerow(
                        {
                            "eval_index": self._eval_index,
                            "train_steps": int(self.num_timesteps),
                            "phase": phase,
                            "episodes": int(summary["episodes"]),
                            "mean_reward": float(summary["mean_reward"]),
                            "std_reward": float(summary["std_reward"]),
                            "mean_steps": float(summary["mean_steps"]),
                            "mean_goal_error": float(summary["mean_goal_error"]),
                            "mean_goal_success": float(summary["mean_goal_success"]),
                            "episode_success_rate": float(summary["episode_success_rate"]),
                            "hit_rate": float(summary["hit_rate"]),
                            "whiff_rate": float(summary["whiff_rate"]),
                            "idle_rate": float(summary["idle_rate"]),
                            "attack_precision": float(summary["attack_precision"]),
                            "weapon_pickup_rate": float(summary["weapon_pickup_rate"]),
                            "mean_damage_trade": float(summary["mean_damage_trade"]),
                            "skill_score": float(summary["skill_score"]),
                            "best_skill_score": float(summary["best_skill_score"]),
                            "retention": float(summary["retention"]),
                            "amnesia": float(summary["amnesia"]),
                            "amnesia_gate_pass": int(gate_pass > 0.5),
                            "mean_op_damage": float(summary["mean_op_damage"]),
                            "mean_self_damage": float(summary["mean_self_damage"]),
                            "mean_self_stocks": float(summary["mean_self_stocks"]),
                            "mean_op_stocks": float(summary["mean_op_stocks"]),
                            "wins": int(summary["wins"]),
                            "losses": int(summary["losses"]),
                            "draws": int(summary["draws"]),
                            "truncs": int(summary["truncs"]),
                            "win_rate": float(summary["win_rate"]),
                        }
                    )

                gate = "PASS" if gate_pass > 0.5 else "AMNESIA"
                print(
                    f"[Eval @{self.num_timesteps} {phase}] "
                    f"score={float(score):.3f} retain={retention:.3f} amnesia={amnesia:.3f} {gate} | "
                    f"succ={float(summary['episode_success_rate']):.3f} | "
                    f"win={float(summary['win_rate']):.3f} | "
                    f"reward={float(summary['mean_reward']):+.3f}+/-{float(summary['std_reward']):.3f} | "
                    f"dmg(self/op)={float(summary['mean_self_damage']):.3f}/{float(summary['mean_op_damage']):.3f}"
                    + (" | best" if improved else "")
                )

            self._best_scores = update_best_scores(self._best_scores, current_scores)
            if self.retention_path is not None:
                save_best_scores(self.retention_path, self._best_scores)
            if self._eval_fh is not None:
                self._eval_fh.flush()

            self._next_eval_step += self.eval_freq_steps

        return True

    def _on_training_end(self) -> None:
        for env in self._eval_envs.values():
            try:
                env.close()
            except Exception:
                pass
        if self._eval_fh is not None:
            self._eval_fh.close()


def default_env_config(max_episode_steps: int, terminate_on_stock_out: bool = False) -> EnvConfig:
    return EnvConfig(
        terminate_on_stock_out=terminate_on_stock_out,
        max_episode_steps=max_episode_steps,
        yolo_infer_every_n_steps=3,
        action_repeat_steps=2,
        action_repeat_min_steps=2,
        action_repeat_max_steps=2,
        tap_latch_steps=1,
    )


def parse_train_args(default_name: str, default_steps: int) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Train {default_name}")
    p.add_argument("--timesteps", type=int, default=default_steps)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--max-episode-steps", type=int, default=1200)
    p.add_argument("--save-dir", type=str, default="train/models")
    p.add_argument("--model-name", type=str, default=default_name)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--plot-every", type=int, default=0)
    p.add_argument("--log-csv", action="store_true", help="Write step/episode CSV logs")
    p.add_argument("--diag-report-every", type=int, default=0, help="Diagnostic print period in steps (0 disables)")
    p.add_argument("--moving-avg", type=int, default=300)
    p.add_argument("--eval-every-steps", type=int, default=0, help="Run periodic evaluation every N timesteps (0 disables)")
    p.add_argument("--eval-episodes", type=int, default=3, help="Episodes per periodic evaluation checkpoint")
    p.add_argument("--eval-stochastic", action="store_true", help="Use stochastic policy during periodic evaluation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delay", type=float, default=3.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.15)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--replay-ratio", type=float, default=0.30, help="Fraction of PPO minibatch sampled from replay")
    p.add_argument("--replay-capacity", type=int, default=262_144, help="Replay storage capacity in transitions")
    p.add_argument("--replay-warmup-updates", type=int, default=1, help="Number of PPO updates before replay sampling")
    p.add_argument("--strict-replay-mix", dest="strict_replay_mix", action="store_true", help="Enforce configured replay ratio when possible")
    p.add_argument("--no-strict-replay-mix", dest="strict_replay_mix", action="store_false")
    p.add_argument("--normalize-advantage-per-source", dest="normalize_advantage_per_source", action="store_true", help="Normalize on-policy and replay advantages separately")
    p.add_argument("--no-normalize-advantage-per-source", dest="normalize_advantage_per_source", action="store_false")
    p.add_argument("--anchor-kl-coef", type=float, default=0.02, help="KL anchor loss coefficient")
    p.add_argument("--anchor-update-interval", type=int, default=8, help="Anchor snapshot refresh period in PPO updates")
    p.add_argument("--anchor-pool-size", type=int, default=4, help="Number of anchor policy snapshots to keep")
    p.add_argument("--bc-loss-coef", type=float, default=0.05, help="Behavior cloning loss coefficient")
    p.add_argument("--bc-batch-size", type=int, default=128, help="BC mini-batch size")
    p.add_argument("--bc-demos-path", type=str, default=None, help="One or more NPZ demo paths separated by semicolon/comma")
    p.add_argument("--eval-phases", type=str, default="", help="'all' or comma/semicolon phase list for retention eval")
    p.add_argument("--eval-include-previous", action="store_true", help="Evaluate all curriculum phases up to the current phase")
    p.add_argument("--amnesia-threshold", type=float, default=0.15, help="Fail retention gate above this amnesia value")
    p.add_argument("--retention-scores-path", type=str, default="", help="Shared retention best-score JSON path")
    p.add_argument("--pcgrad", dest="pcgrad", action="store_true", help="Enable PCGrad between PPO and auxiliary losses")
    p.add_argument("--no-pcgrad", dest="pcgrad", action="store_false")
    p.set_defaults(strict_replay_mix=True, normalize_advantage_per_source=True, pcgrad=True)
    return p.parse_args()


def train_stage_model(
    args: argparse.Namespace,
    make_env: Callable[[], gym.Env],
    stage_spec: Optional[StageSpec] = None,
    make_eval_env: Optional[Callable[[], gym.Env]] = None,
    make_eval_env_for_phase: Optional[Callable[[str], gym.Env]] = None,
) -> None:
    """Train a stage LLC policy with PPO."""
    from feature_extractor.film_extractor import StageGoalFiLMExtractor

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    final_model = save_dir / f"{args.model_name}.zip"

    def _make_algo_env() -> gym.Env:
        return make_env()

    def _make_algo_eval_env() -> gym.Env:
        if make_eval_env is None:
            raise RuntimeError("make_eval_env was requested but not provided")
        return make_eval_env()

    def _make_algo_eval_env_for_phase(phase: str) -> gym.Env:
        if make_eval_env_for_phase is not None:
            return make_eval_env_for_phase(phase)
        return _make_algo_eval_env()

    base_vec = VecMonitor(DummyVecEnv([_make_algo_env]))
    vecnorm_path = save_dir / f"{args.model_name}.vecnormalize.pkl"

    if args.resume and vecnorm_path.exists():
        try:
            vec_env = VecNormalize.load(str(vecnorm_path), base_vec)
        except Exception as exc:
            raise RuntimeError(
                f"[{args.model_name}] Could not load VecNormalize stats from {vecnorm_path}. "
                "This usually means observation dimensions changed (for example after goal schema updates). "
                "Start a fresh run (no --resume) or use a matching checkpoint/stats pair."
            ) from exc
    else:
        vec_env = VecNormalize(base_vec, norm_obs=False, norm_reward=False, clip_obs=10.0)

    model = _build_ppo(args, vec_env, stage_spec, StageGoalFiLMExtractor)

    print(f"[{args.model_name}] Training PPO for {args.timesteps:,} timesteps")
    print(f"[{args.model_name}] Starting in {args.delay:.0f}s - switch to Brawlhalla")
    time.sleep(args.delay)

    her_spec = stage_spec

    dashboard_cb = StageDashboardCallback(
        save_dir=save_dir,
        model_name=args.model_name,
        stage_spec=her_spec,
        plot_every_episodes=max(1, int(args.plot_every)),
        moving_avg_window=args.moving_avg,
        enable_csv=bool(getattr(args, "log_csv", False)),
        enable_plot=int(getattr(args, "plot_every", 0)) > 0,
    )
    callbacks_list: list[BaseCallback] = [dashboard_cb]
    diag_every = int(getattr(args, "diag_report_every", 0))
    if diag_every > 0:
        callbacks_list.append(DiagnosticCallback(report_every=diag_every))
    eval_every = int(getattr(args, "eval_every_steps", 0))
    if eval_every > 0:
        if make_eval_env is None:
            print(f"[{args.model_name}] eval_every_steps={eval_every} requested, but no eval env builder was provided.")
        else:
            eval_cb = PeriodicEvalCallback(
                make_eval_env=_make_algo_eval_env,
                eval_freq_steps=eval_every,
                eval_episodes=int(getattr(args, "eval_episodes", 3)),
                deterministic=not bool(getattr(args, "eval_stochastic", False)),
                seed=int(getattr(args, "seed", 42)),
                save_dir=save_dir,
                model_name=args.model_name,
                current_phase=str(getattr(args, "phase", "")),
                eval_phases=parse_phase_list(
                    getattr(args, "eval_phases", ""),
                    str(getattr(args, "phase", "")),
                    include_previous=bool(getattr(args, "eval_include_previous", False)),
                ),
                make_eval_env_for_phase=_make_algo_eval_env_for_phase,
                amnesia_threshold=float(getattr(args, "amnesia_threshold", 0.15)),
                retention_scores_path=(
                    Path(str(getattr(args, "retention_scores_path", "")).strip())
                    if str(getattr(args, "retention_scores_path", "")).strip()
                    else None
                ),
            )
            callbacks_list.append(eval_cb)
            print(
                f"[{args.model_name}] Periodic eval enabled: every {eval_every} steps, "
                f"{int(getattr(args, 'eval_episodes', 3))} episodes"
            )
    callbacks = CallbackList(callbacks_list)

    interrupted = False
    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=callbacks)
    except KeyboardInterrupt:
        interrupted = True
        model.save(str(final_model))
        print(f"[{args.model_name}] Interrupted checkpoint saved: {final_model}")
    finally:
        try:
            dashboard_cb._on_training_end()
        except Exception:
            pass

    model.save(str(final_model))
    print(f"[{args.model_name}] Saved model to {final_model}")
    if interrupted:
        print(f"[{args.model_name}] Final model also saved after interruption.")


def _build_ppo(args, vec_env, stage_spec, FiLMClass):
    """Build or resume a PPO model."""
    replay_ratio = float(np.clip(getattr(args, "replay_ratio", 0.30), 0.0, 0.95))
    replay_capacity = int(max(1, getattr(args, "replay_capacity", 262_144)))
    replay_warmup_updates = int(max(0, getattr(args, "replay_warmup_updates", 1)))
    strict_replay_mix = bool(getattr(args, "strict_replay_mix", True))
    normalize_advantage_per_source = bool(getattr(args, "normalize_advantage_per_source", True))
    anchor_kl_coef = float(max(0.0, getattr(args, "anchor_kl_coef", 0.02)))
    anchor_update_interval = int(max(1, getattr(args, "anchor_update_interval", 8)))
    anchor_pool_size = int(max(1, getattr(args, "anchor_pool_size", 4)))
    bc_loss_coef = float(max(0.0, getattr(args, "bc_loss_coef", 0.05)))
    bc_batch_size = int(max(1, getattr(args, "bc_batch_size", 128)))
    bc_demos_path = getattr(args, "bc_demos_path", None)
    enable_pcgrad = bool(getattr(args, "pcgrad", True))

    if not bc_demos_path:
        candidates: list[Path] = []
        save_root = Path(getattr(args, "save_dir", "train/models"))
        phase_name = getattr(args, "phase", None)
        if str(phase_name or "").strip().lower() == "all_skills_llc":
            archive_paths = [
                save_root / f"{phase}_demos.npz"
                for phase in previous_phases("all_skills_llc", include_current=False)
            ]
            archive_paths = [path for path in archive_paths if path.exists()]
            if archive_paths:
                bc_demos_path = ";".join(str(path) for path in archive_paths)
        if not bc_demos_path:
            if stage_spec is not None:
                candidates.append(save_root / f"{stage_spec.name}_demos.npz")
            if phase_name:
                candidates.append(save_root / f"{phase_name}_demos.npz")
            model_name = str(getattr(args, "model_name", ""))
            if model_name.startswith("llc_"):
                candidates.append(save_root / f"{model_name[4:]}_demos.npz")
            for candidate in candidates:
                if candidate.exists():
                    bc_demos_path = str(candidate)
                    break

    if args.resume:
        print(f"[{args.model_name}] Resuming PPO from {args.resume}")
        try:
            return AnchoredReplayPPO.load(
                args.resume,
                env=vec_env,
                learning_rate=args.learning_rate,
                clip_range=args.clip_range,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                ent_coef=args.ent_coef,
                vf_coef=args.vf_coef,
                max_grad_norm=args.max_grad_norm,
                seed=args.seed,
                device=args.device,
                replay_ratio=replay_ratio,
                replay_capacity=replay_capacity,
                replay_warmup_updates=replay_warmup_updates,
                strict_replay_mix=strict_replay_mix,
                normalize_advantage_per_source=normalize_advantage_per_source,
                anchor_kl_coef=anchor_kl_coef,
                anchor_update_interval=anchor_update_interval,
                anchor_pool_size=anchor_pool_size,
                bc_loss_coef=bc_loss_coef,
                bc_batch_size=bc_batch_size,
                bc_demos_path=bc_demos_path,
                enable_pcgrad=enable_pcgrad,
            )
        except Exception as exc:
            raise RuntimeError(
                f"[{args.model_name}] Could not resume PPO checkpoint {args.resume}. "
                "Checkpoint architecture likely differs from current stage goal schema/observation shape. "
                "Use a compatible checkpoint or restart training without --resume."
            ) from exc

    if stage_spec is not None:
        goal_feature_names = GOAL_STATE_SPEC_NAMES
        if stage_spec.feature_names is not None:
            goal_feature_names = list(stage_spec.feature_names)
        policy_kwargs = dict(
            features_extractor_class=FiLMClass,
            features_extractor_kwargs=dict(
                goal_feature_names=goal_feature_names,
                features_dim=256,
            ),
            net_arch=dict(pi=[128], vf=[128]),
        )
    else:
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    return AnchoredReplayPPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=args.device,
        replay_ratio=replay_ratio,
        replay_capacity=replay_capacity,
        replay_warmup_updates=replay_warmup_updates,
        strict_replay_mix=strict_replay_mix,
        normalize_advantage_per_source=normalize_advantage_per_source,
        anchor_kl_coef=anchor_kl_coef,
        anchor_update_interval=anchor_update_interval,
        anchor_pool_size=anchor_pool_size,
        bc_loss_coef=bc_loss_coef,
        bc_batch_size=bc_batch_size,
        bc_demos_path=bc_demos_path,
        enable_pcgrad=enable_pcgrad,
    )

def make_base_env(max_episode_steps: int, terminate_on_stock_out: bool = False) -> BrawlDeepEnv:
    return BrawlDeepEnv(config=default_env_config(max_episode_steps=max_episode_steps, terminate_on_stock_out=terminate_on_stock_out))
