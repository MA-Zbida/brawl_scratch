"""Tests for the persistent transition log.

The properties worth guarding are the ones that silently corrupt a world model
rather than crash: a dropped mirror flag, a requested action recorded where an
executed one belongs, and next-state adjacency broken at episode boundaries.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pytest

from world_model.replay_log import (
    TERMINAL_ACTION,
    WorldReplayRecorder,
    WorldReplayWriter,
    iter_transitions,
    read_session,
)


class _StubEnv:
    """Minimal env: returns a counter as the observation, ends after `length` steps."""

    def __init__(self, obs_dim: int = 6, length: int = 4, *, executed_action: int | None = None,
                 mirrored: bool = False) -> None:
        self.obs_dim = obs_dim
        self.length = length
        self.executed_action = executed_action
        self.mirrored = mirrored
        self.t = 0
        self.closed = False
        self.reset_count = 0

    def _obs(self) -> np.ndarray:
        return np.full(self.obs_dim, float(self.t), dtype=np.float32)

    def reset(self, **_: Any):
        self.t = 0
        self.reset_count += 1
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        terminated = self.t >= self.length
        info = {
            "canon_mirrored": 1.0 if self.mirrored else 0.0,
            "op_delta_damage": 2.5,
            "frame_skip": 3,
        }
        if self.executed_action is not None:
            info["effective_action"] = self.executed_action
        return self._obs(), 1.0, terminated, False, info

    def close(self) -> None:
        self.closed = True

    def action_masks(self):
        return np.ones(27, dtype=bool)

    @property
    def unwrapped(self):
        return self


def _run(tmp_path, env: _StubEnv, episodes: int = 1, shard_size: int = 4096):
    writer = WorldReplayWriter(tmp_path, session_id="s", shard_size=shard_size, phase="test")
    recorder = WorldReplayRecorder(env, writer)
    for _ in range(episodes):
        recorder.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = recorder.step(1)
            done = terminated or truncated
    recorder.close()
    return read_session(writer.dir)


def test_records_executed_action_not_requested(tmp_path) -> None:
    """The env sanitises actions; dynamics must be conditioned on what it applied."""
    env = _StubEnv(length=3, executed_action=9)
    data = _run(tmp_path, env)

    steps = data["action"] != TERMINAL_ACTION
    assert np.all(data["action"][steps] == 9), "executed action was not taken from info"
    assert np.all(data["action_requested"][steps] == 1), "requested action was lost"


def test_falls_back_to_requested_when_env_reports_no_executed_action(tmp_path) -> None:
    env = _StubEnv(length=3, executed_action=None)
    data = _run(tmp_path, env)
    steps = data["action"] != TERMINAL_ACTION
    assert np.all(data["action"][steps] == 1)


def test_mirror_flag_survives_the_round_trip(tmp_path) -> None:
    """A canonicalised obs is ambiguous without it -- losing it corrupts everything."""
    data = _run(tmp_path, _StubEnv(length=3, mirrored=True))
    assert np.all(data["mirrored"])

    data = _run(tmp_path / "b", _StubEnv(length=3, mirrored=False))
    assert not np.any(data["mirrored"])


def test_terminal_row_supplies_next_obs_without_duplicating_observations(tmp_path) -> None:
    env = _StubEnv(obs_dim=4, length=3)
    data = _run(tmp_path, env)

    # 3 transitions + 1 terminal state row.
    assert len(data["action"]) == 4
    assert int(data["action"][-1]) == TERMINAL_ACTION
    assert np.all(data["action"][:-1] != TERMINAL_ACTION)

    # obs rows are o_0, o_1, o_2, o_3 -- adjacency gives next_obs for free.
    assert [float(row[0]) for row in data["obs"]] == [0.0, 1.0, 2.0, 3.0]


def test_iter_transitions_pairs_states_and_skips_terminal_rows(tmp_path) -> None:
    data_dir = WorldReplayWriter(tmp_path, session_id="s", phase="test").dir
    env = _StubEnv(obs_dim=4, length=3)
    _run(tmp_path, env, episodes=2)

    pairs = list(iter_transitions(data_dir))
    assert len(pairs) == 6, "two episodes of three transitions each"
    for obs, action, _reward, next_obs, _done in pairs:
        assert action != TERMINAL_ACTION
        assert float(next_obs[0]) == float(obs[0]) + 1.0, "next_obs must be the following state"


def test_transitions_never_cross_an_episode_boundary(tmp_path) -> None:
    env = _StubEnv(obs_dim=4, length=2)
    _run(tmp_path, env, episodes=3)
    data = read_session(tmp_path / "s")

    for obs, _a, _r, next_obs, _d in iter_transitions(tmp_path / "s"):
        # Observations count up within an episode and reset to 0 at the start of
        # the next; a pair that decreases would mean two episodes were joined.
        assert float(next_obs[0]) > float(obs[0])
    assert int(data["episode_id"].max()) == 2


def test_shards_roll_over_and_reassemble_in_order(tmp_path) -> None:
    env = _StubEnv(obs_dim=3, length=10)
    data = _run(tmp_path, env, episodes=4, shard_size=8)

    shards = sorted((tmp_path / "s").glob("shard_*.npz"))
    assert len(shards) > 1, "shard_size=8 over 44 rows should produce several shards"
    assert len(data["action"]) == 4 * 11  # 10 transitions + 1 terminal row each
    assert not list((tmp_path / "s").glob("*.tmp")), "no partial writes left behind"


def test_manifest_records_schema_and_counts(tmp_path) -> None:
    env = _StubEnv(obs_dim=5, length=3)
    _run(tmp_path, env, episodes=2)

    manifest = json.loads((tmp_path / "s" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["obs_dim"] == 5
    assert manifest["transitions"] == 6
    assert manifest["terminal_rows"] == 2
    assert manifest["episodes"] == 2
    assert manifest["phase"] == "test"
    assert manifest["terminal_action"] == TERMINAL_ACTION


def test_dt_wall_spans_the_observation_interval_not_the_gap_between_steps(tmp_path) -> None:
    """dt_wall must cover the step itself plus caller time, so it is never below dt_step.

    The first live run reported dt_wall ~0 ms against dt_step ~21 ms because the
    interval was measured up to the *start* of the step rather than its end.
    """
    class _SlowEnv(_StubEnv):
        def step(self, action):
            time.sleep(0.01)
            return super().step(action)

    data = _run(tmp_path, _SlowEnv(length=4))
    steps = data["action"] != TERMINAL_ACTION
    assert np.all(data["dt_step"][steps] >= 0.009), "dt_step should capture time inside step()"
    assert np.all(data["dt_wall"][steps] >= data["dt_step"][steps]), (
        "dt_wall must span at least the step it covers"
    )


def test_observations_are_copied_from_a_reused_env_buffer(tmp_path) -> None:
    """The real env returns its internal buffer; storing a view repeats one frame.

    `BrawlDeepEnv._get_obs()` fills and returns `self._obs_buffer`, so every call
    hands back the same array object. A recorder that keeps `asarray(obs)` stores
    N views of one buffer, and the shard ends up containing the final frame repeated
    -- a file that is well-formed, whose scalar columns are all correct, and whose
    observations are entirely fabricated. A live session produced exactly that:
    0 of 168 columns varied across 999 frames.

    The other stub returns a fresh array per step, which is why it never caught this.
    """
    class _BufferReusingEnv(_StubEnv):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._buffer = np.zeros(self.obs_dim, dtype=np.float32)

        def _obs(self) -> np.ndarray:
            self._buffer[:] = float(self.t)      # same object, new contents
            return self._buffer

    data = _run(tmp_path, _BufferReusingEnv(obs_dim=4, length=3))
    assert [float(row[0]) for row in data["obs"]] == [0.0, 1.0, 2.0, 3.0], (
        "observations were aliased to the env's buffer instead of copied"
    )

    # The writer must defend itself too, not rely on the recorder having copied
    # first -- append() is public and takes whatever the caller hands it.
    writer = WorldReplayWriter(tmp_path / "direct", session_id="s", shard_size=4096)
    buffer = np.zeros(4, dtype=np.float32)
    common = dict(
        action_requested=1, reward=0.0, terminated=False, truncated=False,
        mirrored=False, dt_step=0.01, dt_wall=0.02, episode_id=0,
    )
    for value in (1.0, 2.0, 3.0):
        buffer[:] = value                        # mutate in place, as the env does
        writer.append(obs=buffer, action=1, step_in_episode=int(value), **common)
    writer.close()

    rows = read_session(writer.dir)["obs"]
    assert [float(row[0]) for row in rows] == [1.0, 2.0, 3.0], (
        "append() stored views of the caller's buffer; the shard holds one repeated frame"
    )


def test_reward_ingredients_are_captured_from_info(tmp_path) -> None:
    data = _run(tmp_path, _StubEnv(length=3))
    steps = data["action"] != TERMINAL_ACTION
    assert np.allclose(data["op_delta_damage"][steps], 2.5)
    assert np.all(data["frame_skip"][steps] == 3)


def test_observation_dim_change_is_rejected(tmp_path) -> None:
    """One session, one observation spec -- a silent change would poison the data."""
    writer = WorldReplayWriter(tmp_path, session_id="s")
    common = dict(
        action=1, action_requested=1, reward=0.0, terminated=False, truncated=False,
        mirrored=False, dt_step=0.01, dt_wall=0.02, episode_id=0, step_in_episode=0,
    )
    writer.append(obs=np.zeros(8, dtype=np.float32), **common)
    with pytest.raises(ValueError, match="observation dim changed"):
        writer.append(obs=np.zeros(9, dtype=np.float32), **common)


def test_wrapper_forwards_attributes_and_closes_the_env(tmp_path) -> None:
    env = _StubEnv(length=2)
    writer = WorldReplayWriter(tmp_path, session_id="s")
    recorder = WorldReplayRecorder(env, writer)

    assert recorder.action_masks().shape == (27,)
    assert recorder.unwrapped is env

    recorder.reset()
    recorder.step(1)
    recorder.close()
    assert env.closed, "wrapping must not swallow close()"
