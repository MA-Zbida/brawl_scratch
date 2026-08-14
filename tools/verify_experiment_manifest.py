#!/usr/bin/env python
"""Fail closed when frozen experiment artifacts no longer match their manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_scalar(data: Any, key: str, default: str = "") -> str:
    if key not in data.files:
        return default
    values = np.asarray(data[key]).reshape(-1)
    return str(values[0]) if values.size else default


def _resolve(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path


def _verify_file(record: dict[str, Any], repo_root: Path) -> tuple[Path, list[str]]:
    raw_path = str(record.get("path", "")).strip()
    path = _resolve(repo_root, raw_path)
    failures: list[str] = []
    if not raw_path:
        return path, ["artifact has no path"]
    if not path.is_file():
        return path, [f"missing artifact: {raw_path}"]

    expected_size = int(record.get("size_bytes", -1))
    if expected_size >= 0 and path.stat().st_size != expected_size:
        failures.append(
            f"size mismatch for {raw_path}: expected {expected_size}, "
            f"got {path.stat().st_size}"
        )
    expected_hash = str(record.get("sha256", "")).lower()
    actual_hash = _sha256(path)
    if not expected_hash or actual_hash != expected_hash:
        failures.append(
            f"sha256 mismatch for {raw_path}: expected {expected_hash or '<missing>'}, "
            f"got {actual_hash}"
        )
    return path, failures


def verify_manifest(
    manifest_path: str | Path,
    *,
    repo_root: str | Path = ".",
) -> list[str]:
    """Return every manifest violation; an empty list means promotion may proceed."""
    manifest_path = Path(manifest_path)
    repo_root = Path(repo_root)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"could not read manifest {manifest_path}: {exc}"]

    failures: list[str] = []
    contract = dict(manifest.get("data_contract", {}))
    obs_dim = int(contract.get("observation_dim", -1))
    goal_dim = int(contract.get("goal_dim", -1))
    action_dim = int(contract.get("action_dim", -1))
    action_encoding = str(contract.get("action_encoding", ""))
    demo_records = list(manifest.get("demo_archives", []))
    sample_total = 0
    episode_total = 0

    for record in demo_records:
        path, file_failures = _verify_file(record, repo_root)
        failures.extend(file_failures)
        if file_failures or not path.is_file():
            continue
        raw_path = str(record.get("path", ""))
        try:
            with np.load(path, allow_pickle=False) as data:
                obs = np.asarray(data["obs"])
                actions_key = (
                    "actions_discrete"
                    if "actions_discrete" in data.files
                    else "actions"
                )
                actions = np.asarray(data[actions_key])
                dones = np.asarray(data["dones"], dtype=bool).reshape(-1)
                target = np.asarray(data["goal_target"])
                mask = np.asarray(data["goal_mask"])

                samples = int(obs.shape[0]) if obs.ndim == 2 else -1
                episodes = int(dones.sum())
                expected_samples = int(record.get("samples", -1))
                expected_episodes = int(record.get("episodes", -1))
                expected_phase = str(record.get("phase", ""))
                phase = _metadata_scalar(data, "phase")
                encoding = _metadata_scalar(data, "action_encoding")

                if obs.ndim != 2 or int(obs.shape[1]) != obs_dim:
                    failures.append(
                        f"observation contract mismatch for {raw_path}: got {obs.shape}, "
                        f"expected [N, {obs_dim}]"
                    )
                if samples != expected_samples:
                    failures.append(
                        f"sample count mismatch for {raw_path}: expected "
                        f"{expected_samples}, got {samples}"
                    )
                if episodes != expected_episodes:
                    failures.append(
                        f"episode count mismatch for {raw_path}: expected "
                        f"{expected_episodes}, got {episodes}"
                    )
                if phase != expected_phase:
                    failures.append(
                        f"phase mismatch for {raw_path}: expected {expected_phase}, got {phase}"
                    )
                if encoding != action_encoding:
                    failures.append(
                        f"action encoding mismatch for {raw_path}: expected "
                        f"{action_encoding}, got {encoding}"
                    )
                if actions.ndim != 1 or actions.shape[0] != samples:
                    failures.append(
                        f"action shape mismatch for {raw_path}: got {actions.shape}, "
                        f"expected ({samples},)"
                    )
                elif actions.size and (
                    int(actions.min()) < 0 or int(actions.max()) >= action_dim
                ):
                    failures.append(
                        f"action range mismatch for {raw_path}: expected [0, {action_dim})"
                    )
                if target.shape != (samples, goal_dim) or mask.shape != (
                    samples,
                    goal_dim,
                ):
                    failures.append(
                        f"goal shape mismatch for {raw_path}: target={target.shape}, "
                        f"mask={mask.shape}, expected ({samples}, {goal_dim})"
                    )
                sample_total += max(0, samples)
                episode_total += episodes
        except (KeyError, OSError, ValueError) as exc:
            failures.append(f"could not validate demo archive {raw_path}: {exc}")

    aggregate = dict(manifest.get("aggregate", {}))
    if sample_total != int(aggregate.get("samples", -1)):
        failures.append(
            f"aggregate sample mismatch: manifest={aggregate.get('samples')}, "
            f"archives={sample_total}"
        )
    if episode_total != int(aggregate.get("episodes", -1)):
        failures.append(
            f"aggregate episode mismatch: manifest={aggregate.get('episodes')}, "
            f"archives={episode_total}"
        )

    for record in manifest.get("artifacts", []):
        _path, file_failures = _verify_file(record, repo_root)
        failures.extend(file_failures)
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="experiments/easy_bot_v0/manifest.json",
    )
    parser.add_argument("--repo-root", default=".")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures = verify_manifest(args.manifest, repo_root=args.repo_root)
    if failures:
        print("EXPERIMENT MANIFEST: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 2
    print("EXPERIMENT MANIFEST: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
