"""Low-Level Controller (LLC) training wrapper with continuous goals.

This replaces the old discrete-subgoal/WPN reward design.
The LLC receives a 6D continuous goal vector concatenated to the base state.
Goals are sampled with a structured distribution and held for 8-16 frames.
"""

from __future__ import annotations

from typing import Optional, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from feature_extractor.memory.state_spec import StateSpec
from hierarchical.goals import GOAL_DIM, GoalSampler, clip_goal

class GoalSamplerLike(Protocol):
    def sample(self, obs: np.ndarray) -> np.ndarray:
        ...


class LLCEnv(gym.Wrapper):
    """Goal-conditioned LLC wrapper.

    Observation:
        [base_state, goal]

    Reward:
        -L1 goal tracking error + small combat auxiliary terms.
    """

    def __init__(
        self,
        env: gym.Env,
        goal_sampler: Optional[GoalSamplerLike] = None,
        min_goal_duration: int = 75,
        max_goal_duration: int = 100,
        goal_error_scale: float = 1.0,
        goal_progress_scale: float = 1.0,
        goal_progress_clip_min: Optional[float] = None,
        goal_progress_clip_max: Optional[float] = None,
        reward_clip: float = 3.0,
        damage_reward_scale: float = 0.02,
        damage_penalty_scale: float = 0.02,
        hitstun_bonus_scale: float = 0.01,
        stock_event_scale: float = 0.05,
        goal_success_threshold: float = 0.2,
        goal_success_bonus: float = 0.2,
        adaptive_success_bonus: bool = True,
        success_bonus_multiplier: float = 3.0,
        success_bonus_min: float = 0.2,
        success_bonus_max: float = 2.0,
        success_enter_threshold: float = 0.18,
        success_exit_threshold: float = 0.24,
        success_cooldown_steps: int = 4,
        segment_end_success_bonus: float = 0.25,
        segment_end_fail_penalty: float = -0.05,
        segment_magnitude_ema_alpha: float = 0.05,
        goal_error_mode: str = "full",
        mask_opponent_features: bool = False,
        resample_goal_on_timer: bool = True,
        terminate_on_goal_success: bool = False,
        terminate_on_death: bool = False,
        min_alive_steps_before_goal_termination: int = 0,
        success_terminal_bonus: float = 0.0,
        death_terminal_penalty: float = -0.2,
        suppress_reward_when_player_missing: bool = True,
    ):
        super().__init__(env)
        self.goal_sampler = goal_sampler or GoalSampler()
        self.min_goal_duration = int(min_goal_duration)
        self.max_goal_duration = int(max_goal_duration)
        self.goal_error_scale = float(goal_error_scale)
        self.goal_progress_scale = float(goal_progress_scale)
        self.goal_progress_clip_min = None if goal_progress_clip_min is None else float(goal_progress_clip_min)
        self.goal_progress_clip_max = None if goal_progress_clip_max is None else float(goal_progress_clip_max)
        self.reward_clip = float(reward_clip)
        self.damage_reward_scale = float(damage_reward_scale)
        self.damage_penalty_scale = float(damage_penalty_scale)
        self.hitstun_bonus_scale = float(hitstun_bonus_scale)
        self.stock_event_scale = float(stock_event_scale)
        self.goal_success_threshold = float(goal_success_threshold)
        self.goal_success_bonus = float(goal_success_bonus)
        self.adaptive_success_bonus = bool(adaptive_success_bonus)
        self.success_bonus_multiplier = float(success_bonus_multiplier)
        self.success_bonus_min = float(success_bonus_min)
        self.success_bonus_max = float(success_bonus_max)
        self.success_enter_threshold = float(success_enter_threshold)
        self.success_exit_threshold = float(success_exit_threshold)
        self.success_cooldown_steps = int(success_cooldown_steps)
        self.segment_end_success_bonus = float(segment_end_success_bonus)
        self.segment_end_fail_penalty = float(segment_end_fail_penalty)
        self.segment_magnitude_ema_alpha = float(segment_magnitude_ema_alpha)
        self.goal_error_mode = str(goal_error_mode)
        self.mask_opponent_features = bool(mask_opponent_features)
        self.resample_goal_on_timer = bool(resample_goal_on_timer)
        self.terminate_on_goal_success = bool(terminate_on_goal_success)
        self.terminate_on_death = bool(terminate_on_death)
        self.min_alive_steps_before_goal_termination = max(0, int(min_alive_steps_before_goal_termination))
        self.success_terminal_bonus = float(success_terminal_bonus)
        self.death_terminal_penalty = float(death_terminal_penalty)
        self.suppress_reward_when_player_missing = bool(suppress_reward_when_player_missing)

        obs_shape = env.observation_space.shape
        assert obs_shape is not None, "Base env must define observation shape"
        self._base_dim = int(obs_shape[0])
        self._aug_dim = self._base_dim + GOAL_DIM
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._aug_dim,),
            dtype=np.float32,
        )

        self._aug_obs_buffer = np.zeros(self._aug_dim, dtype=np.float32)
        self._goal = np.zeros(GOAL_DIM, dtype=np.float32)
        self._goal_lane = "unknown"
        self._goal_steps_left = 0
        self._goal_uid = 0
        self._prev_goal_error: float | None = None
        self._success_state: bool = False
        self._success_cooldown_left: int = 0
        self._segment_had_success: bool = False
        self._segment_reward_sum: float = 0.0
        self._segment_step_count: int = 0
        self._segment_abs_mag_ema: float = 0.5
        self._alive_steps_since_reset: int = 0

        self._opponent_mask_indices: list[int] = []
        if self.mask_opponent_features:
            names = [
                "opponent_x", "opponent_y", "opponent_vx", "opponent_vy",
                "opponent_grounded", "opponent_damage_pct", "opponent_exists",
                "opponent_jumps_norm", "opponent_on_edge", "opponent_is_offstage",
                "rel_dx", "rel_dy", "rel_distance", "in_strike_range", "facing_opponent",
                "opponent_hitstun", "opponent_facing_dir", "opponent_time_since_hit",
                "opponent_dodge_cooldown_norm", "rel_vx", "rel_vy", "frame_advantage_estimate",
            ]
            self._opponent_mask_indices = [StateSpec.index(n) for n in names if n in StateSpec.names()]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._prepare_obs(obs)
        self._success_state = False
        self._success_cooldown_left = 0
        self._segment_had_success = False
        self._segment_reward_sum = 0.0
        self._segment_step_count = 0
        self._alive_steps_since_reset = 0
        self._sample_new_goal(obs)
        info["goal"] = self._goal.copy()
        info["goal_lane"] = self._goal_lane
        info["goal_error_mode"] = self.goal_error_mode
        info["goal_uid"] = int(self._goal_uid)
        info["goal_new_sampled"] = True
        return self._augment(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        obs = self._prepare_obs(obs)

        op_delta = float(info.get("op_delta_damage", 0.0))
        self_delta = float(info.get("self_delta_damage", 0.0))
        op_stock_lost = float(info.get("op_stock_lost_step", 0.0))
        self_stock_lost = float(info.get("self_stock_lost_step", 0.0))
        player_exists_info = float(info.get("player_exists", 1.0))
        player_respawn_timer = float(info.get("player_respawn_timer", 0.0))
        player_exists = (player_exists_info > 0.5) and (player_respawn_timer <= 0.0)
        if player_exists:
            self._alive_steps_since_reset += 1

        sampled_new_goal = False
        if self.resample_goal_on_timer:
            if self._goal_steps_left <= 0:
                self._sample_new_goal(obs)
                sampled_new_goal = True
                self._prev_goal_error = None
                self._success_state = False
                self._success_cooldown_left = 0
                self._segment_had_success = False
                self._segment_reward_sum = 0.0
                self._segment_step_count = 0
            else:
                self._goal_steps_left -= 1
            segment_end_now = self._goal_steps_left == 0
        else:
            segment_end_now = False

        goal_error = self.goal_error(obs, self._goal, mode=self.goal_error_mode)
        norm_goal_error = goal_error / float(self.goal_error_dim(self.goal_error_mode))
        goal_progress = 0.0 if self._prev_goal_error is None else (self._prev_goal_error - goal_error)
        progress_term = self.goal_progress_scale * goal_progress
        if (self.goal_progress_clip_min is not None) or (self.goal_progress_clip_max is not None):
            clip_min = -np.inf if self.goal_progress_clip_min is None else self.goal_progress_clip_min
            clip_max = np.inf if self.goal_progress_clip_max is None else self.goal_progress_clip_max
            progress_term = float(min(max(progress_term, clip_min), clip_max))
        # Progress term helps centering returns around 0; penalty still discourages drift.
        goal_reward = progress_term - self.goal_error_scale * norm_goal_error

        if self._success_cooldown_left > 0:
            self._success_cooldown_left -= 1

        entered_success = False
        reward_suppressed = self.suppress_reward_when_player_missing and (not player_exists)
        if (not reward_suppressed) and (not self._success_state) and (norm_goal_error < self.success_enter_threshold) and (self._success_cooldown_left <= 0):
            self._success_state = True
            entered_success = True
            self._segment_had_success = True
            self._success_cooldown_left = self.success_cooldown_steps
        elif (not reward_suppressed) and self._success_state and (norm_goal_error > self.success_exit_threshold):
            self._success_state = False

        if self.adaptive_success_bonus:
            adaptive_bonus = min(
                max(self.success_bonus_multiplier * self._segment_abs_mag_ema, self.success_bonus_min),
                self.success_bonus_max,
            )
        else:
            adaptive_bonus = self.goal_success_bonus

        success_bonus = float(adaptive_bonus if entered_success else 0.0)
        if reward_suppressed:
            goal_reward = 0.0
            success_bonus = 0.0
        goal_reward += success_bonus

        segment_terminal_bonus = 0.0
        if segment_end_now:
            if self._segment_had_success:
                segment_terminal_bonus = self.segment_end_success_bonus
            else:
                segment_terminal_bonus = self.segment_end_fail_penalty
            goal_reward += segment_terminal_bonus

        if reward_suppressed:
            self._prev_goal_error = None
        else:
            self._prev_goal_error = goal_error

        opponent_hitstun = StateSpec.get(obs, "opponent_hitstun")
        player_hitstun = StateSpec.get(obs, "player_hitstun")

        # KO/death event signal requested by user: +1.0 for kill, -1.0 for death.
        # Scaled down so this remains a minor LLC shaping term.
        stock_event_signal = (1.0 * op_stock_lost) + (-1.0 * self_stock_lost)
        stock_event_term = self.stock_event_scale * stock_event_signal

        if reward_suppressed:
            combat_aux = 0.0
        elif self.goal_error_mode == "locomotion":
            combat_aux = 0.0
        else:
            combat_aux = (
                self.damage_reward_scale * op_delta
                - self.damage_penalty_scale * self_delta
                + self.hitstun_bonus_scale * (opponent_hitstun - player_hitstun)
                + stock_event_term
            )

        can_terminate_on_goal = self._alive_steps_since_reset >= self.min_alive_steps_before_goal_termination
        goal_reached_event = bool(entered_success and self.terminate_on_goal_success and can_terminate_on_goal)
        death_event = bool((self_stock_lost > 0.0) and self.terminate_on_death)
        terminal_event_reward = 0.0
        if goal_reached_event:
            terminal_event_reward += self.success_terminal_bonus
        if death_event:
            terminal_event_reward += self.death_terminal_penalty

        unclipped_reward = goal_reward + combat_aux + terminal_event_reward
        reward = float(min(max(unclipped_reward, -self.reward_clip), self.reward_clip))
        terminated = bool(terminated or goal_reached_event or death_event)

        self._segment_reward_sum += reward
        self._segment_step_count += 1
        if segment_end_now and self._segment_step_count > 0:
            segment_mag = abs(self._segment_reward_sum) / float(self._segment_step_count)
            alpha = float(min(max(self.segment_magnitude_ema_alpha, 0.001), 1.0))
            self._segment_abs_mag_ema = (1.0 - alpha) * self._segment_abs_mag_ema + alpha * segment_mag

        info["goal"] = self._goal.copy()
        info["goal_lane"] = self._goal_lane
        info["goal_error_mode"] = self.goal_error_mode
        info["goal_uid"] = int(self._goal_uid)
        info["goal_new_sampled"] = bool(sampled_new_goal)
        info["goal_progress"] = float(goal_progress)
        info["goal_progress_term"] = float(progress_term)
        info["goal_error"] = float(goal_error)
        info["goal_error_norm"] = float(norm_goal_error)
        info["goal_success"] = float(1.0 if self._success_state else 0.0)
        info["goal_success_event"] = float(1.0 if entered_success else 0.0)
        info["goal_success_bonus"] = float(success_bonus)
        info["goal_success_bonus_adaptive"] = float(adaptive_bonus)
        info["segment_terminal_bonus"] = float(segment_terminal_bonus)
        info["segment_end_now"] = float(1.0 if segment_end_now else 0.0)
        info["segment_had_success"] = float(1.0 if self._segment_had_success else 0.0)
        info["segment_abs_mag_ema"] = float(self._segment_abs_mag_ema)
        info["goal_reward"] = float(goal_reward)
        info["reward_suppressed_player_missing"] = float(1.0 if reward_suppressed else 0.0)
        info["player_exists"] = float(1.0 if player_exists else 0.0)
        info["goal_reached_event"] = float(1.0 if goal_reached_event else 0.0)
        info["death_event"] = float(1.0 if death_event else 0.0)
        info["terminal_event_reward"] = float(terminal_event_reward)
        info["alive_steps_since_reset"] = int(self._alive_steps_since_reset)
        info["can_terminate_on_goal"] = float(1.0 if can_terminate_on_goal else 0.0)
        info["stock_event_signal"] = float(stock_event_signal)
        info["stock_event_term"] = float(stock_event_term)
        info["combat_aux_reward"] = float(combat_aux)
        info["llc_reward"] = reward
        info["goal_steps_left"] = int(self._goal_steps_left)

        return self._augment(obs), reward, terminated, truncated, info

    def set_goal(self, goal: np.ndarray, duration: Optional[int] = None) -> None:
        """Set an externally provided goal (used by HSP)."""
        self._goal = clip_goal(np.asarray(goal, dtype=np.float32))
        if duration is None:
            if self.resample_goal_on_timer:
                self._goal_steps_left = np.random.randint(self.min_goal_duration, self.max_goal_duration + 1)
            else:
                self._goal_steps_left = -1
        else:
            self._goal_steps_left = max(1, int(duration))
        self._prev_goal_error = None

    def _sample_new_goal(self, obs: np.ndarray) -> None:
        self._goal = self.goal_sampler.sample(obs)
        self._goal_lane = str(getattr(self.goal_sampler, "last_lane", "unknown"))
        if self.resample_goal_on_timer:
            self._goal_steps_left = np.random.randint(self.min_goal_duration, self.max_goal_duration + 1)
        else:
            self._goal_steps_left = -1
        self._goal_uid += 1

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        np.copyto(self._aug_obs_buffer[: self._base_dim], obs)
        np.copyto(self._aug_obs_buffer[self._base_dim :], self._goal)
        return self._aug_obs_buffer

    def _prepare_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self.mask_opponent_features:
            return obs
        masked = np.array(obs, copy=True)
        masked[self._opponent_mask_indices] = 0.0
        return masked

    @staticmethod
    def goal_error(obs: np.ndarray, goal: np.ndarray, mode: str = "full") -> float:
        if mode == "xy":
            player_x = StateSpec.get(obs, "player_x")
            player_y = StateSpec.get(obs, "player_y")
            return float(
                abs(player_x - goal[0])
                + abs(player_y - goal[1])
            )

        if mode == "locomotion":
            signed_dx_center = StateSpec.get(obs, "signed_dx_to_stage_center")
            dist_center = StateSpec.get(obs, "dist_to_stage_center")
            player_vx = StateSpec.get(obs, "player_vx")
            player_vy = StateSpec.get(obs, "player_vy")
            dx_to_ledge = StateSpec.get(obs, "signed_dx_to_ledge")
            dy_to_ledge = StateSpec.get(obs, "dy_to_ledge")

            return float(
                abs(signed_dx_center - goal[0])
                + abs(dist_center - goal[1])
                + abs(player_vx - goal[2])
                + abs(player_vy - goal[3])
                + abs(dx_to_ledge - goal[4])
                + abs(dy_to_ledge - goal[5])
            )

        rel_dx = StateSpec.get(obs, "rel_dx")
        rel_dy = StateSpec.get(obs, "rel_dy")
        player_vx = StateSpec.get(obs, "player_vx")
        player_vy = StateSpec.get(obs, "player_vy")
        dx_to_ledge = StateSpec.get(obs, "signed_dx_to_ledge")
        dy_to_ledge = StateSpec.get(obs, "dy_to_ledge")

        return float(
            abs(rel_dx - goal[0])
            + abs(rel_dy - goal[1])
            + abs(player_vx - goal[2])
            + abs(player_vy - goal[3])
            + abs(dx_to_ledge - goal[4])
            + abs(dy_to_ledge - goal[5])
        )

    @staticmethod
    def goal_error_dim(mode: str = "full") -> int:
        if mode == "xy":
            return 2
        return GOAL_DIM
