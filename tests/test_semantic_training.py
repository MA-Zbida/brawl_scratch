"""New action masks must survive collection, PPO optimization and reload."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from brawl.control.semantic_actions import ACTIONS, BY_NAME, ControlContext
from brawl.control.semantic_executor import SemanticExecutor
from brawl.control.input import NullInputController
from brawl.pipelines.semantic57.contract import OBSERVATION_DIM, BASE_DIM, ACTION_DIM, metadata, validate_metadata
from brawl.pipelines.semantic57.policy import build_model, load_model, save_model


class MaskEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBSERVATION_DIM,), np.float32)
        self.action_space = gym.spaces.Discrete(ACTION_DIM)
        self.index = 0
        self.chosen = []

    def obs(self):
        obs = np.zeros(OBSERVATION_DIM, np.float32)
        obs[-ACTION_DIM + self.index % ACTION_DIM] = 1
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return self.obs(), {}

    def step(self, action):
        assert int(action) == self.index % ACTION_DIM
        self.chosen.append(int(action))
        self.index += 1
        return self.obs(), .1, False, self.index % 8 == 0, {}


def test_sampling_evaluation_and_prediction_use_identical_masks():
    env = MaskEnv()
    model = build_model(env, n_steps=8, batch_size=4, n_epochs=1)
    obs = np.zeros((2, OBSERVATION_DIM), np.float32)
    obs[0, -ACTION_DIM + 3] = 1
    obs[1, -ACTION_DIM + 45] = 1
    tensor = torch.as_tensor(obs)
    actions, values, logprob = model.policy(tensor)
    assert actions.tolist() == [3, 45]
    _, evaluated, entropy = model.policy.evaluate_actions(tensor, actions)
    torch.testing.assert_close(logprob, evaluated)
    assert torch.isfinite(entropy).all()
    assert model.predict(obs, deterministic=True)[0].tolist() == [3, 45]
    with pytest.raises(ValueError, match="mask"):
        model.policy(torch.zeros_like(tensor))


def test_ppo_uses_stored_masks_across_two_rollouts_and_reload(tmp_path):
    env = MaskEnv()
    model = build_model(env, n_steps=8, batch_size=4, n_epochs=2)
    model.learn(16)
    assert len(env.chosen) == 16
    path = tmp_path / "model.zip"
    save_model(model, path)
    loaded = load_model(path, env)
    loaded.learn(8, reset_num_timesteps=False)
    assert loaded.num_timesteps == 24
    assert str(loaded.device) == "cpu"


def test_contract_rejects_legacy_metadata_and_changed_semantics():
    validate_metadata(metadata())
    with pytest.raises(ValueError):
        validate_metadata({"action_dim": 27, "observation_dim": 146})
    altered = metadata()
    altered["semantic_contract"]["actions"][0]["name"] = "CHANGED"
    with pytest.raises(ValueError):
        validate_metadata(altered)


def test_executor_busy_mask_and_state_are_observable():
    executor = SemanticExecutor(NullInputController())
    ctx = ControlContext(grounded=False)
    executor.submit("AIR_DODGE_TOWARD", ctx, now=0)
    assert executor.available_actions(ctx, now=.01) == (True,) + (False,) * (len(ACTIONS) - 1)
    state = executor.observation_state(now=.01, mirrored=False)
    assert len(state) == OBSERVATION_DIM - BASE_DIM - ACTION_DIM
    assert state[BY_NAME["AIR_DODGE_TOWARD"].id] == 1
    executor.advance(ctx, now=.03)
    executor.advance(ctx, now=.06)
    assert not executor.available_actions(ctx, now=.07)[BY_NAME["RECOVERY_TOWARD"].id]
    assert executor.available_actions(ctx, now=.3)[BY_NAME["RECOVERY_TOWARD"].id]
