from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
pytest.importorskip("ultralytics")

import gymnasium as gym

from feature_extractor.memory.state_spec import StateSpec
from train.curriculum_config import build_phase_spec
from train.llc_stage_common import StageGoalEnv


class DummyStageEnv(gym.Env):
    metadata = {}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(StateSpec.dim(),),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiDiscrete([4, 2, 2, 4])
        self.steps = 0

    def _obs(self) -> np.ndarray:
        obs = np.zeros((StateSpec.dim(),), dtype=np.float32)
        obs[StateSpec.index("player_x")] = 0.45
        obs[StateSpec.index("player_y")] = 0.58
        obs[StateSpec.index("signed_dx_to_ledge")] = 0.0
        obs[StateSpec.index("dy_to_ledge")] = 0.0
        obs[StateSpec.index("player_has_weapon")] = 0.0
        obs[StateSpec.index("weapon_dx")] = 0.1
        obs[StateSpec.index("weapon_dy")] = 0.0
        obs[StateSpec.index("rel_distance")] = 0.2
        obs[StateSpec.index("rel_dy")] = 0.0
        obs[StateSpec.index("in_strike_range")] = 1.0
        obs[StateSpec.index("frame_advantage_estimate")] = 0.2
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self._obs(), {"player_exists": 1.0}

    def step(self, action):
        self.steps += 1
        info = {
            "op_delta_damage": 1.0 if int(np.asarray(action).reshape(-1)[3]) else 0.0,
            "self_delta_damage": 0.25,
            "player_exists": 1.0,
            "player_respawn_timer": 0.0,
        }
        return self._obs(), 0.0, False, self.steps >= 3, info


def test_stage_goal_env_emits_dynamic_goal_diagnostics() -> None:
    spec = build_phase_spec("all_skills_llc", terminate_on_death=False)
    env = StageGoalEnv(DummyStageEnv(), spec)

    obs, info = env.reset(seed=7)
    assert obs.shape[0] == StateSpec.dim() + (2 * len(spec.feature_names or []))
    assert "goal_mask" in info

    _, _, _, _, info = env.step(np.array([3, 0, 0, 1], dtype=np.int64))
    assert "active_goal_feature_errors" in info
    assert "raw_goal_feats" in info
    assert "stage_action" in info
    assert info["stage_action_attack"] == 1
    assert np.asarray(info["goal_mask"]).shape == np.asarray(info["goal_target"]).shape

