from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

import torch as th
from gymnasium import spaces

from algo.anchored_replay_ppo import _BehaviorCloneStore


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

