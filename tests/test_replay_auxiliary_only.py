from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")

import gymnasium as gym
import torch as th

from brawl.control.action_space import ACTION_DIM
from brawl.algo.anchored_replay_ppo import AnchoredReplayPPO


class _TinyEnv(gym.Env):
    metadata = {}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(ACTION_DIM)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((4,), dtype=np.float32), {}

    def step(self, action):
        return np.zeros((4,), dtype=np.float32), 0.0, False, True, {}


def _rollout_batch(size: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        observations=th.zeros((size, 4), dtype=th.float32),
        actions=th.zeros((size, 1), dtype=th.int64),
        old_values=th.zeros((size,), dtype=th.float32),
        old_log_prob=th.zeros((size,), dtype=th.float32),
        advantages=th.arange(size, dtype=th.float32),
        returns=th.zeros((size,), dtype=th.float32),
    )


def test_replay_rows_are_auxiliary_only_and_never_enter_ppo_batch() -> None:
    model = AnchoredReplayPPO(
        "MlpPolicy",
        _TinyEnv(),
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        replay_ratio=0.5,
        replay_warmup_updates=0,
        anchor_kl_coef=0.1,
        bc_loss_coef=0.0,
        device="cpu",
        verbose=0,
    )
    replay = model._replay_store
    replay.observations = th.full((4, 4), 99.0, dtype=th.float32)
    replay.actions = th.full((4, 1), ACTION_DIM - 1, dtype=th.int64)
    replay.old_values = th.zeros((4,), dtype=th.float32)
    replay.old_log_prob = th.zeros((4,), dtype=th.float32)
    replay.advantages = th.ones((4,), dtype=th.float32)
    replay.returns = th.zeros((4,), dtype=th.float32)
    replay.priorities = th.ones((4,), dtype=th.float32)
    model._train_calls = 1

    ppo_batch, auxiliary_observations, replay_fraction = model._prepare_training_sources(
        _rollout_batch(),
    )

    assert ppo_batch.observations.shape == (2, 4)
    assert th.all(ppo_batch.observations == 0.0)
    assert auxiliary_observations.shape == (4, 4)
    assert th.any(th.all(auxiliary_observations == 99.0, dim=1))
    assert replay_fraction == pytest.approx(0.5)


def test_replay_ratio_never_removes_every_on_policy_row() -> None:
    model = AnchoredReplayPPO(
        "MlpPolicy",
        _TinyEnv(),
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        replay_ratio=0.95,
        replay_warmup_updates=0,
        anchor_kl_coef=0.1,
        bc_loss_coef=0.0,
        device="cpu",
        verbose=0,
    )
    replay = model._replay_store
    replay.observations = th.ones((4, 4), dtype=th.float32)
    replay.actions = th.ones((4, 1), dtype=th.int64)
    replay.old_values = th.zeros((4,), dtype=th.float32)
    replay.old_log_prob = th.zeros((4,), dtype=th.float32)
    replay.advantages = th.ones((4,), dtype=th.float32)
    replay.returns = th.zeros((4,), dtype=th.float32)
    replay.priorities = th.ones((4,), dtype=th.float32)
    model._train_calls = 1

    ppo_batch, _, _ = model._prepare_training_sources(_rollout_batch())

    assert ppo_batch.observations.shape[0] >= 1
