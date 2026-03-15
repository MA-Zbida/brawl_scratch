"""High-Level Strategic Policy (HSP) environment with continuous goal actions.

HSP outputs a 6D continuous goal. The environment executes a frozen LLC for
`macro_steps` primitive steps under that goal.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from feature_extractor.memory.state_spec import StateSpec
from hierarchical.goals import GOAL_DIM, GOAL_HIGH, GOAL_LOW, clip_goal
from hierarchical.llc_env import LLCEnv


class HSPEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        base_env: gym.Env,
        llc_model,
        macro_steps: int = 12,
        deterministic_llc: bool = True,
        goal_progress_scale: float = 1.0,
        combat_scale: float = 0.03,
        stock_scale: float = 0.5,
    ):
        super().__init__()
        self.base_env = base_env
        self.llc = llc_model
        self.macro_steps = int(macro_steps)
        self.deterministic_llc = bool(deterministic_llc)
        self.goal_progress_scale = float(goal_progress_scale)
        self.combat_scale = float(combat_scale)
        self.stock_scale = float(stock_scale)

        self.observation_space = base_env.observation_space
        self.action_space = spaces.Box(low=GOAL_LOW, high=GOAL_HIGH, shape=(GOAL_DIM,), dtype=np.float32)

        self._obs = np.zeros(base_env.observation_space.shape or (0,), dtype=np.float32)
        self._llc_state = None
        self._episode_starts: Optional[np.ndarray] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._obs = obs
        self._llc_state = None
        self._episode_starts = np.ones(1, dtype=bool)
        return obs, info

    def step(self, goal_action: np.ndarray):
        goal = clip_goal(np.asarray(goal_action, dtype=np.float32))

        total_goal_progress = 0.0
        total_combat = 0.0
        total_stock = 0.0
        steps_executed = 0
        terminated = False
        truncated = False
        info = {}

        prev_error = LLCEnv.goal_error(self._obs, goal)

        for _ in range(self.macro_steps):
            llc_obs = np.concatenate([self._obs, goal], dtype=np.float32)
            llc_action, self._llc_state = self.llc.predict(
                llc_obs,
                state=self._llc_state,
                episode_start=self._episode_starts,
                deterministic=self.deterministic_llc,
            )
            self._episode_starts = np.zeros(1, dtype=bool)

            obs, _, terminated, truncated, step_info = self.base_env.step(llc_action)

            curr_error = LLCEnv.goal_error(obs, goal)
            total_goal_progress += (prev_error - curr_error)
            prev_error = curr_error

            op_delta = float(step_info.get("op_delta_damage", 0.0))
            self_delta = float(step_info.get("self_delta_damage", 0.0))
            total_combat += (op_delta - self_delta)

            op_stock_lost = float(step_info.get("op_stock_lost_step", 0.0))
            self_stock_lost = float(step_info.get("self_stock_lost_step", 0.0))
            total_stock += (op_stock_lost - self_stock_lost)

            self._obs = obs
            info = step_info
            steps_executed += 1

            if terminated or truncated:
                break

        reward = (
            self.goal_progress_scale * total_goal_progress
            + self.combat_scale * total_combat
            + self.stock_scale * total_stock
        )

        info["hsp_goal"] = goal.copy()
        info["hsp_macro_steps"] = int(steps_executed)
        info["hsp_goal_progress"] = float(total_goal_progress)
        info["hsp_combat_delta"] = float(total_combat)
        info["hsp_stock_delta"] = float(total_stock)
        info["hsp_reward"] = float(reward)
        info["frame_advantage_estimate"] = StateSpec.get(self._obs, "frame_advantage_estimate")

        return self._obs, float(reward), terminated, truncated, info

    def close(self):
        self.base_env.close()

    @property
    def unwrapped(self):
        return self.base_env.unwrapped
