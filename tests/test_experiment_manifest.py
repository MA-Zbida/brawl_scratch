"""The Easy-bot baseline must train from the exact reviewed artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tools.verify_experiment_manifest import verify_manifest
from train.phase_registry import PHASE_ORDER


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_demo(path: Path) -> None:
    np.savez_compressed(
        path,
        obs=np.zeros((3, 4), dtype=np.float32),
        actions=np.zeros((3,), dtype=np.int64),
        actions_discrete=np.zeros((3,), dtype=np.int64),
        dones=np.asarray([False, False, True]),
        goal_target=np.ones((3, 2), dtype=np.float32),
        goal_mask=np.ones((3, 2), dtype=np.float32),
        phase=np.asarray(["movement_fluency"]),
        action_encoding=np.asarray(["discrete27"]),
    )


def _write_manifest(path: Path, demo: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "test",
                "data_contract": {
                    "observation_dim": 4,
                    "action_encoding": "discrete27",
                    "action_dim": 27,
                    "goal_dim": 2,
                },
                "demo_archives": [
                    {
                        "phase": "movement_fluency",
                        "path": demo.name,
                        "sha256": _sha256(demo),
                        "size_bytes": demo.stat().st_size,
                        "samples": 3,
                        "episodes": 1,
                    }
                ],
                "aggregate": {"samples": 3, "episodes": 1},
            }
        ),
        encoding="utf-8",
    )


def test_manifest_verifier_accepts_exact_demo_contract(tmp_path) -> None:
    demo = tmp_path / "movement.npz"
    manifest = tmp_path / "manifest.json"
    _write_demo(demo)
    _write_manifest(manifest, demo)

    assert verify_manifest(manifest, repo_root=tmp_path) == []


def test_manifest_verifier_fails_closed_after_archive_changes(tmp_path) -> None:
    demo = tmp_path / "movement.npz"
    manifest = tmp_path / "manifest.json"
    _write_demo(demo)
    _write_manifest(manifest, demo)
    demo.write_bytes(demo.read_bytes() + b"changed")

    failures = verify_manifest(manifest, repo_root=tmp_path)

    assert any("sha256 mismatch" in failure for failure in failures)


def test_easy_bot_config_freezes_the_deferred_pilot_contract() -> None:
    config_path = Path("experiments/easy_bot_v0/config.json")
    manifest_path = Path("experiments/easy_bot_v0/manifest.json")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert tuple(config["phase_order"]) == PHASE_ORDER
    assert config["model_dir"] == "train/models/easy_bot_v0"
    assert config["perception"]["weapon_classifier_enabled"] is False
    assert config["perception"]["yolo_infer_every_n_steps"] == 1
    assert config["ppo_pilot"]["timesteps_per_phase"] == 4096
    assert config["ppo_pilot"]["n_steps"] == 512
    assert manifest["aggregate"] == {
        "samples": 7938,
        "episodes": 200,
        "size_bytes": 986780,
    }
