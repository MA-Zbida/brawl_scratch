from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
pytest.importorskip("ultralytics")

import gymnasium as gym

from action_space import ACTION_DIM, Action, components as action_components


def _is_attack(action) -> bool:
    comp = action_components(int(np.asarray(action).reshape(-1)[0]))
    return bool(comp.light or comp.heavy)

from feature_extractor.memory.state_spec import StateSpec
from train.curriculum_config import build_phase_spec
from train.llc_stage_common import StageGoalEnv


class _DummyPlayer:
    def __init__(self) -> None:
        self.weapon_state = 1.0


class _DummyMemory:
    def __init__(self) -> None:
        self.player = _DummyPlayer()


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
        self.action_space = gym.spaces.Discrete(ACTION_DIM)
        self.steps = 0
        self.memory = _DummyMemory()

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
        obs[StateSpec.index("opponent_damage_pct")] = 0.2
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self._obs(), {"player_exists": 1.0}

    def step(self, action):
        self.steps += 1
        info = {
            # An attack landed iff the executed action carries a light/heavy component.
            "op_delta_damage": 1.0 if _is_attack(action) else 0.0,
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

    # Action 0 is NOOP; use a real attack so the attack diagnostic is exercised.
    _, _, _, _, info = env.step(int(Action.LIGHT_TOWARD))
    assert "active_goal_feature_errors" in info
    assert "raw_goal_feats" in info
    assert "stage_action" in info
    assert info["stage_action"] == int(Action.LIGHT_TOWARD)
    assert info["stage_action_attack"] == 1
    assert info["stage_action_hdir"] == 1
    assert np.asarray(info["goal_mask"]).shape == np.asarray(info["goal_target"]).shape


def test_stage_goal_env_reset_preserves_base_weapon_state() -> None:
    spec = build_phase_spec("weapon_acquisition", terminate_on_death=False)
    base = DummyStageEnv()
    base.memory.player.weapon_state = 1.0
    env = StageGoalEnv(base, spec)

    env.reset(seed=7)

    assert base.memory.player.weapon_state == 1.0
