"""Persistent transition log — the permanent record of every real interaction.

Real interaction is this project's scarce resource. At 30 Hz a full curriculum is
~21 hours of gameplay, and every hour run before this log exists is an hour that
cannot be recovered. So the log is written unconditionally alongside whatever else
a session is doing: demo collection, BC, PPO, evaluation. It costs one wrapper and
some disk, and it is the difference between starting a world model with a dataset
and starting it at zero.

Design constraints that shaped this:

* **Zero hot-path cost.** This is a `gymnasium.Wrapper`, not an edit to `env.step`.
  Nothing changes when it is not attached.
* **Executed, not requested, actions.** The env sanitises and masks actions before
  injection, so the action the policy asked for is not always the action the game
  received. Dynamics must be conditioned on what actually happened; both are stored.
* **The mirror flag is mandatory.** Observations are canonicalised, so an obs row is
  ambiguous without knowing whether it was reflected. Losing it silently corrupts
  every downstream model.
* **Two clocks.** `dt_step` is time spent inside `env.step`; `dt_wall` is the
  interval between successive observations, which additionally covers policy
  inference. Dynamics needs the latter -- it is the real time between o_t and o_t+1.
* **Crash tolerance.** Shards are flushed periodically and renamed into place
  atomically, so a hard kill loses at most one buffer rather than the session.

Storage layout::

    <root>/<session_id>/
        manifest.json        schema, dims, git commit, counts
        shard_00000.npz
        shard_00001.npz
        ...

Transitions are stored as o_t rows in sequence, so ``next_obs`` for row *i* is row
*i+1*. To keep that true at episode ends without duplicating every observation, the
terminal observation is appended as its own row with ``action == TERMINAL_ACTION``
(-1). Those rows are states, not transitions: read them as next-states, never as
training targets. This costs one extra row per episode instead of doubling the file.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

#: Sentinel action for a terminal-observation row (a state, not a transition).
TERMINAL_ACTION: int = -1

#: 1 -- `dt_wall` held only the gap *between* steps, excluding the step itself, so it
#:      read ~0 ms against a ~21 ms `dt_step`. Recoverable: the true observation
#:      interval for a v1 session is `dt_wall + dt_step`. Use `observation_interval()`.
#: 2 -- `dt_wall` spans the full interval between successive observations.
SCHEMA_VERSION: int = 2

#: Per-transition scalar fields pulled from `info`, with their dtype and the
#: default used when a wrapper in the stack does not provide them.
_INFO_SCALARS: tuple[tuple[str, str, str, float], ...] = (
    # (column, info key, dtype, default)
    ("op_delta_damage", "op_delta_damage", "float32", 0.0),
    ("self_delta_damage", "self_delta_damage", "float32", 0.0),
    ("op_stock_lost", "op_stock_lost_step", "float32", 0.0),
    ("self_stock_lost", "self_stock_lost_step", "float32", 0.0),
    ("goal_error", "goal_error", "float32", 0.0),
    ("goal_success", "goal_success", "float32", 0.0),
    ("frame_skip", "frame_skip", "int16", 1.0),
    # Match state as held at observation-assembly time. Paired with the deltas
    # above these show whether reward and observation agree: a delta firing while
    # health is unchanged means the reward saw a transient the policy never did.
    ("op_health", "op_health", "float32", -1.0),
    ("self_health", "self_health", "float32", -1.0),
    ("op_stocks_left", "op_stocks_left", "float32", -1.0),
    ("self_stocks_left", "self_stocks_left", "float32", -1.0),
)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


class WorldReplayWriter:
    """Append-only transition store, flushed to disk in shards."""

    def __init__(
        self,
        root: str | Path,
        *,
        session_id: str = "",
        shard_size: int = 4096,
        phase: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if int(shard_size) <= 0:
            raise ValueError(f"shard_size must be positive, got {shard_size}")
        self.shard_size = int(shard_size)
        self.session_id = session_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.dir = Path(root) / self.session_id
        self.dir.mkdir(parents=True, exist_ok=True)

        self.phase = str(phase)
        self.extra_metadata = dict(metadata or {})
        self.transitions_written = 0
        self.terminal_rows_written = 0
        self.episodes = 0
        self._shard_index = 0
        self._obs_dim = -1
        self._closed = False
        self._buf: list[dict[str, Any]] = []

    # ── writing ────────────────────────────────────────────────────────────

    def append(
        self,
        *,
        obs: np.ndarray,
        action: int,
        action_requested: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        mirrored: bool,
        dt_step: float,
        dt_wall: float,
        episode_id: int,
        step_in_episode: int,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        """Record one transition (or, with `action=TERMINAL_ACTION`, one final state)."""
        if self._closed:
            raise RuntimeError("cannot append to a closed WorldReplayWriter")

        # Copy, never view. `_get_obs()` hands back the env's reusable observation
        # buffer, so `asarray(...).reshape(-1)` aliases it -- every buffered row would
        # point at the same memory and the shard would contain one frame repeated N
        # times. That corrupts silently: the file is well-formed and every scalar
        # column (taken from `info` by value) still looks right.
        flat = np.array(obs, dtype=np.float32, copy=True).reshape(-1)
        if self._obs_dim < 0:
            self._obs_dim = int(flat.shape[0])
        elif int(flat.shape[0]) != self._obs_dim:
            raise ValueError(
                f"observation dim changed mid-session: {flat.shape[0]} != {self._obs_dim}. "
                "A single session must come from one observation spec."
            )

        row: dict[str, Any] = {
            "obs": flat,
            "action": int(action),
            "action_requested": int(action_requested),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "mirrored": bool(mirrored),
            "dt_step": float(dt_step),
            "dt_wall": float(dt_wall),
            "episode_id": int(episode_id),
            "step_in_episode": int(step_in_episode),
        }
        source = info or {}
        for column, key, _dtype, default in _INFO_SCALARS:
            row[column] = float(source.get(key, default))

        self._buf.append(row)
        if int(action) == TERMINAL_ACTION:
            self.terminal_rows_written += 1
        else:
            self.transitions_written += 1

        if len(self._buf) >= self.shard_size:
            self.flush()

    def flush(self) -> Path | None:
        """Write the buffered rows as one shard. Returns the path, or None if empty."""
        if not self._buf:
            return None

        payload: dict[str, np.ndarray] = {
            "obs": np.stack([r["obs"] for r in self._buf]).astype(np.float32),
            "action": np.asarray([r["action"] for r in self._buf], dtype=np.int16),
            "action_requested": np.asarray([r["action_requested"] for r in self._buf], dtype=np.int16),
            "reward": np.asarray([r["reward"] for r in self._buf], dtype=np.float32),
            "terminated": np.asarray([r["terminated"] for r in self._buf], dtype=bool),
            "truncated": np.asarray([r["truncated"] for r in self._buf], dtype=bool),
            "mirrored": np.asarray([r["mirrored"] for r in self._buf], dtype=bool),
            "dt_step": np.asarray([r["dt_step"] for r in self._buf], dtype=np.float32),
            "dt_wall": np.asarray([r["dt_wall"] for r in self._buf], dtype=np.float32),
            "episode_id": np.asarray([r["episode_id"] for r in self._buf], dtype=np.int32),
            "step_in_episode": np.asarray([r["step_in_episode"] for r in self._buf], dtype=np.int32),
        }
        for column, _key, dtype, _default in _INFO_SCALARS:
            payload[column] = np.asarray([r[column] for r in self._buf], dtype=dtype)

        path = self.dir / f"shard_{self._shard_index:05d}.npz"
        tmp = path.with_suffix(".npz.tmp")
        # Write then rename: a kill mid-write leaves a .tmp behind rather than a
        # truncated shard that reads as valid until it doesn't. The handle is
        # passed rather than the path because savez_compressed appends `.npz` to
        # any filename lacking it, which would defeat the rename.
        with open(tmp, "wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(tmp, path)

        self._shard_index += 1
        self._buf.clear()
        self._write_manifest()
        return path

    def _write_manifest(self) -> None:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "phase": self.phase,
            "obs_dim": self._obs_dim,
            "shards": self._shard_index,
            "transitions": self.transitions_written,
            "terminal_rows": self.terminal_rows_written,
            "episodes": self.episodes,
            "terminal_action": TERMINAL_ACTION,
            "git_commit": _git_commit(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **self.extra_metadata,
        }
        tmp = self.dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp, self.dir / "manifest.json")

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._closed = True
        self._write_manifest()


# ── reading ────────────────────────────────────────────────────────────────


def read_session(session_dir: str | Path) -> dict[str, np.ndarray]:
    """Concatenate every shard of one session in order."""
    directory = Path(session_dir)
    shards = sorted(directory.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards in {directory}")

    parts: list[dict[str, np.ndarray]] = []
    for shard in shards:
        with np.load(shard) as data:
            parts.append({key: np.asarray(data[key]) for key in data.files})

    keys = parts[0].keys()
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}


def observation_interval(session_dir: str | Path) -> np.ndarray:
    """Seconds between successive observations, corrected for the session's schema.

    Always prefer this to reading `dt_wall` directly. Schema 1 recorded only the gap
    between steps, so its true interval is `dt_wall + dt_step`; schema 2 records the
    interval outright. Dynamics conditioned on the raw v1 column would see ~0 s
    between states that are ~21 ms apart.
    """
    directory = Path(session_dir)
    data = read_session(directory)
    try:
        version = int(json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["schema_version"])
    except Exception:
        version = SCHEMA_VERSION

    if version < 2:
        return np.asarray(data["dt_wall"], dtype=np.float64) + np.asarray(data["dt_step"], dtype=np.float64)
    return np.asarray(data["dt_wall"], dtype=np.float64)


def iter_transitions(session_dir: str | Path) -> Iterator[tuple[np.ndarray, int, float, np.ndarray, bool]]:
    """Yield `(obs, action, reward, next_obs, done)` with terminal rows resolved.

    Terminal rows supply `next_obs` for the preceding transition and are never
    yielded as transitions themselves.
    """
    data = read_session(session_dir)
    obs = data["obs"]
    action = data["action"]
    reward = data["reward"]
    done = data["terminated"] | data["truncated"]
    episode = data["episode_id"]

    for i in range(len(action) - 1):
        if int(action[i]) == TERMINAL_ACTION:
            continue
        if int(episode[i + 1]) != int(episode[i]):
            # A missing terminal row means the session was killed mid-episode.
            continue
        yield obs[i], int(action[i]), float(reward[i]), obs[i + 1], bool(done[i])


# ── the wrapper ────────────────────────────────────────────────────────────


class WorldReplayRecorder:
    """Gymnasium wrapper that logs every transition to a `WorldReplayWriter`.

    Deliberately not a subclass of `gymnasium.Wrapper` so that this module stays
    importable without gymnasium (the tests exercise it against a stub env).
    Attribute access falls through to the wrapped env, so `action_masks()` and the
    rest of the stack keep working.
    """

    def __init__(self, env: Any, writer: WorldReplayWriter) -> None:
        self.env = env
        self.writer = writer
        self._episode_id = -1
        self._step_in_episode = 0
        self._last_obs: np.ndarray | None = None
        self._last_return_time: float | None = None

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails, so it cannot shadow our own state.
        return getattr(self.env, name)

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def reset(self, **kwargs: Any) -> Any:
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        self._episode_id += 1
        self._step_in_episode = 0
        self._last_obs = np.array(obs, dtype=np.float32, copy=True).reshape(-1)
        self._last_return_time = time.perf_counter()
        self.writer.episodes = self._episode_id + 1
        return result

    def step(self, action: Any) -> Any:
        requested = int(np.asarray(action, dtype=np.int64).reshape(-1)[0])
        started = time.perf_counter()

        obs, reward, terminated, truncated, info = self.env.step(action)

        finished = time.perf_counter()
        # The interval between o_t and o_t+1 runs from when the previous observation
        # was handed back to when this one is: it covers this step *plus* the caller's
        # time in between. Measuring to `started` instead would exclude the step and
        # report ~0, which is what the first live run produced.
        dt_wall = finished - self._last_return_time if self._last_return_time is not None else finished - started
        executed = int(info.get("effective_action", requested)) if isinstance(info, Mapping) else requested
        mirrored = bool(float(info.get("canon_mirrored", 0.0)) > 0.5) if isinstance(info, Mapping) else False

        if self._last_obs is not None:
            self.writer.append(
                obs=self._last_obs,
                action=executed,
                action_requested=requested,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                mirrored=mirrored,
                dt_step=finished - started,
                dt_wall=dt_wall,
                episode_id=self._episode_id,
                step_in_episode=self._step_in_episode,
                info=info if isinstance(info, Mapping) else {},
            )

        self._last_obs = np.array(obs, dtype=np.float32, copy=True).reshape(-1)
        self._step_in_episode += 1
        self._last_return_time = time.perf_counter()

        if terminated or truncated:
            # The final observation is a state with no action taken from it. It is
            # recorded so the last real transition has a next_obs.
            self.writer.append(
                obs=self._last_obs,
                action=TERMINAL_ACTION,
                action_requested=TERMINAL_ACTION,
                reward=0.0,
                terminated=bool(terminated),
                truncated=bool(truncated),
                mirrored=mirrored,
                dt_step=0.0,
                dt_wall=0.0,
                episode_id=self._episode_id,
                step_in_episode=self._step_in_episode,
                info={},
            )
            self._last_obs = None

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        try:
            self.env.close()
        finally:
            self.writer.close()
