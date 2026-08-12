from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")

import gymnasium as gym
import torch as th

from action_space import ACTION_DIM
from algo.anchored_replay_ppo import AnchoredReplayPPO

_OBS_DIM = 77  # arbitrary; the pool mechanics do not depend on the real StateSpec width


class TinyDiscreteEnv(gym.Env):
    """Minimal env matching the live action space so the policy head has the right shape."""

    metadata = {}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(_OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(ACTION_DIM)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((_OBS_DIM,), dtype=np.float32), {}

    def step(self, action):
        return np.zeros((_OBS_DIM,), dtype=np.float32), 0.0, False, True, {}


def _make_model(**kwargs) -> AnchoredReplayPPO:
    # device="cpu" is deliberate: SB3 recommends it for MlpPolicy, and leaving it
    # on "auto" put the snapshot weights on CUDA while the test fed CPU tensors,
    # which fails only on a machine that has a GPU.
    return AnchoredReplayPPO(
        "MlpPolicy",
        TinyDiscreteEnv(),
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        anchor_kl_coef=0.1,
        anchor_pool_size=2,
        bc_loss_coef=0.0,
        device="cpu",
        verbose=0,
        **kwargs,
    )


def test_anchor_snapshot_pool_is_bounded_and_sampleable() -> None:
    model = _make_model()

    model._refresh_anchor_policy()
    model._refresh_anchor_policy()
    model._refresh_anchor_policy()

    assert len(model._anchor_pool) == 2
    loss = model._compute_anchor_kl_loss(th.zeros((2, _OBS_DIM), dtype=th.float32))
    assert th.isfinite(loss)


@pytest.mark.skipif(not th.cuda.is_available(), reason="needs a second device to mismatch")
def test_anchor_kl_loss_survives_observations_from_another_device() -> None:
    """The KL term must not crash when caller and policy disagree on device.

    In training both come from `self.device`, but the anchor pool is the one place
    that evaluates a *copied* policy, so a mismatch here stays silent until it throws
    mid-optimisation -- long after a run has started. The returned scalar must come
    back on the caller's device so the surrounding loss arithmetic still works.
    """
    model = _make_model()  # policy on CPU
    model._refresh_anchor_policy()

    observations = th.zeros((2, _OBS_DIM), dtype=th.float32, device="cuda")
    loss = model._compute_anchor_kl_loss(observations)

    assert th.isfinite(loss)
    assert loss.device.type == "cuda"
