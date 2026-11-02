from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from brawl.pipelines.llc.curriculum_config import build_phase_spec
from brawl.pipelines.llc.train_config import make_config
import brawl.pipelines.llc.train_curriculum as train_curriculum
from brawl.pipelines.llc.train_curriculum import _maybe_wrap_world_replay
from brawl.world_model import WorldReplayRecorder


class _FakeEnv:
    closed = False

    def close(self) -> None:
        self.closed = True


class _GymFakeEnv(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    action_space = gym.spaces.Discrete(27)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        return (
            np.ones(3, dtype=np.float32),
            1.0,
            False,
            True,
            {"effective_action": int(action)},
        )


def test_world_replay_is_zero_cost_and_off_by_default(monkeypatch) -> None:
    env = _FakeEnv()
    cfg = make_config("movement_fluency")
    monkeypatch.setattr(
        train_curriculum,
        "WorldReplayWriter",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("writer constructed")),
    )

    wrapped = _maybe_wrap_world_replay(env, cfg, build_phase_spec(cfg.phase))

    assert wrapped is env


def test_world_replay_is_created_only_when_cli_path_is_supplied(tmp_path) -> None:
    env = _FakeEnv()
    cfg = replace(
        make_config("spacing_neutral"),
        world_replay=str(tmp_path),
        world_replay_shard_size=17,
        opponent_setup="moving_non_attacking",
    )

    wrapped = _maybe_wrap_world_replay(env, cfg, build_phase_spec(cfg.phase))

    assert isinstance(wrapped, WorldReplayRecorder)
    assert wrapped.writer.shard_size == 17
    assert wrapped.writer.phase == "spacing_neutral"
    assert wrapped.writer.extra_metadata["source"] == "train_curriculum"
    assert wrapped.writer.extra_metadata["opponent_setup"] == "moving_non_attacking"
    assert wrapped.writer.dir.parent == tmp_path
    wrapped.close()


def test_opt_in_recorder_is_accepted_and_flushed_by_sb3_vec_env(tmp_path) -> None:
    cfg = replace(
        make_config("movement_fluency"),
        world_replay=str(tmp_path),
        world_replay_shard_size=17,
    )
    spec = build_phase_spec(cfg.phase)
    holder = {}

    def make_env():
        wrapped = _maybe_wrap_world_replay(_GymFakeEnv(), cfg, spec)
        holder["writer"] = wrapped.writer
        return wrapped

    vec_env = DummyVecEnv([make_env])
    vec_env.reset()
    vec_env.step(np.asarray([1], dtype=np.int64))
    vec_env.close()

    writer = holder["writer"]
    assert (writer.dir / "manifest.json").exists()
    assert list(writer.dir.glob("shard_*.npz")), "the final partial shard was not flushed"
