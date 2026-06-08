from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")

import gymnasium as gym
import torch as th

from algo.anchored_replay_ppo import AnchoredReplayPPO


class TinyMultiDiscreteEnv(gym.Env):
    metadata = {}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(77,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([4, 2, 2, 4])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((77,), dtype=np.float32), {}

    def step(self, action):
        return np.zeros((77,), dtype=np.float32), 0.0, False, True, {}


def test_anchor_snapshot_pool_is_bounded_and_sampleable() -> None:
    model = AnchoredReplayPPO(
        "MlpPolicy",
        TinyMultiDiscreteEnv(),
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        anchor_kl_coef=0.1,
        anchor_pool_size=2,
        bc_loss_coef=0.0,
        verbose=0,
    )

    model._refresh_anchor_policy()
    model._refresh_anchor_policy()
    model._refresh_anchor_policy()

    assert len(model._anchor_pool) == 2
    loss = model._compute_anchor_kl_loss(th.zeros((2, 77), dtype=th.float32))
    assert th.isfinite(loss)

