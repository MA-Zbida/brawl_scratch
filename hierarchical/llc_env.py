"""Low-Level Controller (LLC) wrapper for structured goal-conditioned training.

Observation layout:
    [base_state, goal_target(k), goal_mask(k)]

Reward:
    progress = error(s_t, g) - error(s_{t+1}, g)
    reward = clip(progress_scale * progress, -0.1, 0.3)
    + success_bonus when current_error < threshold
"""

from __future__ import annotations

from typing import Optional, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hierarchical.goals import (
    GOAL_DIM,
    GOAL_TARGET_DIM,
    GoalSampler,
    clip_goal,
    extract_goal_features,
    split_goal,
)


class GoalSamplerLike(Protocol):
    def sample(self, obs: np.ndarray) -> np.ndarray:
        ...


class LLCEnv(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        goal_sampler: Optional[GoalSamplerLike] = None,
        min_goal_duration: int = 16,
        max_goal_duration: int = 32,
        progress_scale: float = 1.0,
        progress_clip_min: float = -0.1,
        progress_clip_max: float = 0.3,
        success_threshold: float = 0.12,
        success_bonus: float = 0.25,
        reward_clip: float = 1.0,
        terminate_on_goal_success: bool = False,
        resample_goal_on_timer: bool = True,
    ):
        super().__init__(env)
        self.goal_sampler = goal_sampler or GoalSampler()

        self.min_goal_duration = int(min_goal_duration)
        self.max_goal_duration = int(max_goal_duration)
        self.progress_scale = float(progress_scale)
        self.progress_clip_min = float(progress_clip_min)
        self.progress_clip_max = float(progress_clip_max)
        self.success_threshold = float(success_threshold)
        self.success_bonus = float(success_bonus)
        self.reward_clip = float(reward_clip)
        self.terminate_on_goal_success = bool(terminate_on_goal_success)
        self.resample_goal_on_timer = bool(resample_goal_on_timer)

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

        self._aug_obs_buffer = np.zeros((self._aug_dim,), dtype=np.float32)
        self._goal = np.zeros((GOAL_DIM,), dtype=np.float32)
        self._goal_lane = "unknown"
        self._goal_steps_left = 0
        self._goal_uid = 0
        self._prev_goal_error: float | None = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = np.asarray(obs, dtype=np.float32)
        self._sample_new_goal(obs)
        self._prev_goal_error = None

        info["goal"] = self._goal.copy()
        info["goal_lane"] = self._goal_lane
        info["goal_uid"] = int(self._goal_uid)
        info["goal_new_sampled"] = True
        return self._augment(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        obs = np.asarray(obs, dtype=np.float32)

        sampled_new_goal = False
        if self.resample_goal_on_timer:
            if self._goal_steps_left <= 0:
                self._sample_new_goal(obs)
                sampled_new_goal = True
                self._prev_goal_error = None
            else:
                self._goal_steps_left -= 1

        prev_error = self._prev_goal_error
        curr_error = self.goal_error(obs, self._goal)
        progress = 0.0 if prev_error is None else (prev_error - curr_error)

        reward = self.progress_scale * progress
        reward = float(np.clip(reward, self.progress_clip_min, self.progress_clip_max))

        success = bool(curr_error < self.success_threshold)
        if success:
            reward += self.success_bonus

        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))
        self._prev_goal_error = curr_error

        if self.terminate_on_goal_success and success:
            terminated = True

        info["goal"] = self._goal.copy()
        info["goal_lane"] = self._goal_lane
        info["goal_uid"] = int(self._goal_uid)
        info["goal_new_sampled"] = bool(sampled_new_goal)
        info["goal_error"] = float(curr_error)
        info["goal_progress"] = float(progress)
        info["goal_success"] = float(1.0 if success else 0.0)
        info["goal_steps_left"] = int(self._goal_steps_left)
        info["llc_reward"] = float(reward)

        return self._augment(obs), reward, terminated, truncated, info

    def set_goal(self, goal: np.ndarray, duration: Optional[int] = None) -> None:
        self._goal = clip_goal(np.asarray(goal, dtype=np.float32))
        if duration is None:
            if self.resample_goal_on_timer:
                self._goal_steps_left = int(np.random.randint(self.min_goal_duration, self.max_goal_duration + 1))
            else:
                self._goal_steps_left = -1
        else:
            self._goal_steps_left = max(1, int(duration))
        self._prev_goal_error = None
        self._goal_uid += 1

    def _sample_new_goal(self, obs: np.ndarray) -> None:
        self._goal = self.goal_sampler.sample(obs)
        self._goal_lane = str(getattr(self.goal_sampler, "last_lane", "unknown"))
        if self.resample_goal_on_timer:
            self._goal_steps_left = int(np.random.randint(self.min_goal_duration, self.max_goal_duration + 1))
        else:
            self._goal_steps_left = -1
        self._goal_uid += 1

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        np.copyto(self._aug_obs_buffer[: self._base_dim], obs)
        np.copyto(self._aug_obs_buffer[self._base_dim :], self._goal)
        return self._aug_obs_buffer

    @staticmethod
    def goal_error(obs: np.ndarray, goal: np.ndarray) -> float:
        target, mask = split_goal(clip_goal(goal))
        feats = extract_goal_features(obs)
        masked_abs = np.abs(feats - target) * mask
        return float(np.sum(masked_abs))

    @staticmethod
    def goal_error_dim() -> int:
        return GOAL_TARGET_DIM
