from __future__ import annotations

from dataclasses import replace

from train.curriculum_config import build_phase_spec
from train.train_config import make_config
from train.train_curriculum import _maybe_wrap_world_replay
from world_model import WorldReplayRecorder


class _FakeEnv:
    closed = False

    def close(self) -> None:
        self.closed = True


def test_world_replay_is_zero_cost_and_off_by_default(tmp_path) -> None:
    env = _FakeEnv()
    cfg = make_config("movement_fluency")

    wrapped = _maybe_wrap_world_replay(env, cfg, build_phase_spec(cfg.phase))

    assert wrapped is env
    assert not tmp_path.exists()


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
