from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
pytest.importorskip("ultralytics")

import gymnasium as gym

from action_space import ACTION_DIM, Action
from feature_extractor.memory.state_spec import StateSpec
from train.curriculum_config import build_phase_spec
from train.curriculum_goals import GOAL_INDEX, default_goal_target
from train.llc_stage_common import StageGoalEnv


class _StaticStageEnv(gym.Env):
    metadata = {}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(StateSpec.dim(),),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(ACTION_DIM)

    @staticmethod
    def _obs() -> np.ndarray:
        obs = np.zeros((StateSpec.dim(),), dtype=np.float32)
        obs[StateSpec.index("player_x")] = 0.45
        obs[StateSpec.index("player_y")] = 0.58
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs(), {"player_exists": 1.0, "canon_mirrored": 0.0}

    def step(self, action):
        return self._obs(), 0.0, False, False, {
            "player_exists": 1.0,
            "canon_mirrored": 0.0,
        }


def _movement_target(x: float, y: float = 0.58) -> np.ndarray:
    target = default_goal_target()
    target[GOAL_INDEX["player_x"]] = x
    target[GOAL_INDEX["player_y"]] = y
    return target


def test_reset_resamples_goal_that_is_already_satisfied() -> None:
    samples = iter((_movement_target(0.45), _movement_target(0.65)))
    spec = replace(
        build_phase_spec("movement_fluency", terminate_on_death=False),
        target_sampler=lambda _obs: next(samples),
        reset_goal_resample_attempts=3,
    )
    env = StageGoalEnv(_StaticStageEnv(), spec)

    _, info = env.reset(seed=7)

    assert info["goal_reset_resamples"] == 1
    assert info["goal_reset_rejected"] == 0.0
    assert info["goal_active"] == 1.0
    assert env.goal_error(env._obs_buf, info) >= spec.success_threshold


def test_exhausted_reset_goal_is_inactive_and_cannot_score_trivial_success() -> None:
    spec = replace(
        build_phase_spec("movement_fluency", terminate_on_death=False),
        target_sampler=lambda _obs: _movement_target(0.45),
        reset_goal_resample_attempts=2,
    )
    env = StageGoalEnv(_StaticStageEnv(), spec)

    _, reset_info = env.reset(seed=7)
    _, _, terminated, truncated, step_info = env.step(int(Action.NOOP))

    assert reset_info["goal_reset_resamples"] == 2
    assert reset_info["goal_reset_rejected"] == 1.0
    assert reset_info["goal_active"] == 0.0
    assert not terminated
    assert not truncated
    assert step_info["goal_success"] == 0.0
    assert step_info["goal_active"] == 0.0
