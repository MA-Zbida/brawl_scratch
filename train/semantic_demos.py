"""Versioned goal-free demos, with the sampling mask embedded in every row."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from train.semantic_contract import ACTION_DIM, OBSERVATION_DIM, metadata, validate_metadata


@dataclass(frozen=True)
class DemoBatch:
    obs: np.ndarray
    actions: np.ndarray
    episodes: np.ndarray
    families: np.ndarray
    drills: np.ndarray | None = None


def validate_rows(obs: np.ndarray, actions: np.ndarray) -> None:
    if obs.ndim != 2 or obs.shape[1] != OBSERVATION_DIM or actions.shape != (len(obs),):
        raise ValueError("Semantic demo shape mismatch; legacy phase demos cannot be relabelled")
    if not len(obs) or not np.isfinite(obs).all() or not np.issubdtype(actions.dtype, np.integer):
        raise ValueError("Empty/nonfinite observations or noninteger actions")
    mask = obs[:, -ACTION_DIM:]
    if not np.isin(mask, [0, 1]).all() or not (mask.sum(axis=1) > 0).all():
        raise ValueError("Invalid sampling masks")
    if (actions < 0).any() or (actions >= ACTION_DIM).any():
        raise ValueError("Action outside semantic registry")
    if not mask[np.arange(len(obs)), actions].all():
        raise ValueError("Demo contains a masked action")


def save_demos(path: Path, batch: DemoBatch, summary: dict) -> None:
    validate_rows(batch.obs, batch.actions)
    if batch.episodes.shape != batch.actions.shape or batch.families.shape != batch.actions.shape:
        raise ValueError("Episode/family arrays must align with actions")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, obs=batch.obs, actions=batch.actions, episodes=batch.episodes,
                            families=batch.families, contract_json=np.asarray(json.dumps(metadata())),
                            summary_json=np.asarray(json.dumps(summary)))


def load_demos(path: Path) -> DemoBatch:
    with np.load(path, allow_pickle=False) as data:
        if "contract_json" not in data:
            raise ValueError("Missing semantic demo contract; refusing legacy archive")
        validate_metadata(json.loads(str(data["contract_json"].item())))
        batch = DemoBatch(*(data[key].copy() for key in ("obs", "actions", "episodes", "families")))
    validate_rows(batch.obs, batch.actions)
    if batch.episodes.shape != batch.actions.shape or batch.families.shape != batch.actions.shape:
        raise ValueError("Corrupt demo episode/family arrays")
    return batch


def split_episodes(batch: DemoBatch, *, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.unique(batch.episodes)
    if len(episodes) < 2:
        raise ValueError("BC validation requires at least two independent collection episodes")
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    n_validation = min(len(episodes) - 1, max(1, int(np.ceil(len(episodes) * .2))))
    validation = np.isin(batch.episodes, episodes[:n_validation])
    return np.flatnonzero(~validation), np.flatnonzero(validation)
