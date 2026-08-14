from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

import torch as th
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from action_space import ACTION_DIM
from algo.anchored_replay_ppo import AnchoredReplayPPO, _BehaviorCloneStore
from train.llc_stage_common import _require_complete_bc_anchor


def _write_demo(path, phase: str, offset: float) -> None:
    obs = np.full((8, 77), offset, dtype=np.float32)
    actions = np.tile(np.array([[1, 0, 0, 3]], dtype=np.int64), (8, 1))
    np.savez(path, obs=obs, actions=actions, phase=np.array([phase]))


def test_bc_store_accepts_multiple_npz_archives(tmp_path) -> None:
    first = tmp_path / "recovery.npz"
    second = tmp_path / "movement.npz"
    _write_demo(first, "recovery_mastery", 0.1)
    _write_demo(second, "movement_fluency", 0.9)

    store = _BehaviorCloneStore(
        f"{first};{second}",
        spaces.MultiDiscrete([4, 2, 2, 4]),
        expected_obs_dim=77,
    )

    assert store.enabled
    assert store.num_datasets == 2
    assert store.size == 16

    obs, actions = store.sample(6, th.device("cpu"))
    assert obs.shape == (6, 77)
    assert actions.shape == (6, 4)
    assert th.any(obs < 0.5)
    assert th.any(obs > 0.5)


def test_bc_store_rejects_mismatched_observation_dims(tmp_path) -> None:
    bad = tmp_path / "bad.npz"
    np.savez(
        bad,
        obs=np.zeros((4, 12), dtype=np.float32),
        actions=np.zeros((4, 4), dtype=np.int64),
    )

    store = _BehaviorCloneStore(
        str(bad),
        spaces.MultiDiscrete([4, 2, 2, 4]),
        expected_obs_dim=77,
    )

    assert not store.enabled
    assert "obs dim mismatch" in store.disable_reason


class _TinyDiscreteEnv(gym.Env):
    metadata = {}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_DIM)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((12,), dtype=np.float32), {}

    def step(self, action):
        return np.zeros((12,), dtype=np.float32), 0.0, False, True, {}


@pytest.mark.parametrize("checkpoint_class", [PPO, AnchoredReplayPPO])
def test_loading_checkpoint_rebuilds_bc_store(tmp_path, checkpoint_class) -> None:
    """BC anchoring must survive both plain-BC and resumed-LLC checkpoints."""
    demo = tmp_path / "movement.npz"
    np.savez(
        demo,
        obs=np.zeros((8, 12), dtype=np.float32),
        actions_discrete=np.zeros((8,), dtype=np.int64),
        phase=np.asarray(["movement_fluency"]),
    )
    checkpoint = tmp_path / "checkpoint"
    common = {
        "n_steps": 8,
        "batch_size": 4,
        "n_epochs": 1,
        "device": "cpu",
        "verbose": 0,
    }
    if checkpoint_class is AnchoredReplayPPO:
        common["bc_loss_coef"] = 0.0
    checkpoint_class("MlpPolicy", _TinyDiscreteEnv(), **common).save(checkpoint)

    loaded = AnchoredReplayPPO.load(
        checkpoint,
        env=_TinyDiscreteEnv(),
        device="cpu",
        bc_loss_coef=0.05,
        bc_demos_path=str(demo),
    )

    assert loaded._bc_store is not None
    assert loaded._bc_store.enabled
    assert loaded._bc_store.num_datasets == 1
    assert loaded._bc_store.size == 8


def test_training_rejects_a_partial_bc_anchor_chain() -> None:
    store = SimpleNamespace(
        enabled=True,
        paths=["movement.npz", "weapon.npz"],
        num_datasets=1,
        size=8,
        disable_reason="some BC datasets skipped: dataset not found: weapon.npz",
    )
    model = SimpleNamespace(_bc_store=store)

    with pytest.raises(RuntimeError, match="incomplete BC anchor"):
        _require_complete_bc_anchor(
            model,
            bc_loss_coef=0.05,
            bc_demos_path="movement.npz;weapon.npz",
            model_name="pilot",
        )


def test_training_reports_a_complete_bc_anchor_chain(capsys) -> None:
    store = SimpleNamespace(
        enabled=True,
        paths=["movement.npz", "weapon.npz"],
        num_datasets=2,
        size=42,
        disable_reason="",
    )
    model = SimpleNamespace(_bc_store=store)

    _require_complete_bc_anchor(
        model,
        bc_loss_coef=0.05,
        bc_demos_path="movement.npz;weapon.npz",
        model_name="pilot",
    )

    assert "BC anchor enabled: datasets=2 samples=42" in capsys.readouterr().out
