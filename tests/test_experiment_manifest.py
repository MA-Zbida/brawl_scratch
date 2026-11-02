"""The Easy-bot baseline must train from the exact reviewed artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.llc.verify_experiment_manifest import verify_manifest
from brawl.pipelines.llc.phase_registry import PHASE_ORDER


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

    assert config["schema_version"] == 2
    assert config["model_dir"] == "train/models/easy_bot_v0"
    # The current checkpoint learned the export pipeline rather than live crops:
    # it reports mean P(armed)=0.873 on unarmed live samples. Keep the live chain
    # on action inference until the classifier is retrained on its serving domain.
    perception = config["perception"]
    assert perception["weapon_classifier_enabled"] is False
    assert perception["weapon_classifier_checkpoint"] == "train/models/weapon_classifier_r18.pt"
    assert perception["weapon_classifier_cv_accuracy"] >= 0.90
    # 4-frame cadence: ~1.7 ms/frame amortised (8% of a 21 ms step) versus 21% if
    # it ran every frame, against a variable that moved 3 times in 1537 live steps.
    assert perception["weapon_classifier_interval"] == 4
    assert perception["weapon_classifier_min_confidence"] == 0.7
    assert perception["yolo_infer_every_n_steps"] == 1
    pipeline = config["pipeline"]
    assert pipeline["collect_phases"] == [
        "weapon_acquisition",
        "spacing_neutral",
        "combat_execution",
    ]
    assert "recovery_mastery" not in pipeline["collect_phases"]
    assert pipeline["recovery"]["mode"] == "bc_only_existing_archive"
    assert pipeline["stage_order"][0] == "recovery_bc"
    assert all(stage != "recovery_collect" for stage in pipeline["stage_order"])
    skill = config["skill_pretraining"]
    final = config["final_agent"]
    assert tuple(skill["phase_order"]) == PHASE_ORDER
    assert skill["initialization"]["device"] == "cpu"
    assert skill["ppo_pilot"]["timesteps_per_phase"] == 4096
    assert skill["ppo_pilot"]["n_steps"] == 512
    assert skill["ppo_pilot"]["device"] == "cpu"
    assert final["goal_conditioning"] is False
    assert final["objective"] == "match"
    assert final["device"] == "cpu"
    assert final["observation_dim"] == 146
    assert manifest["final_agent_contract"] == {
        "observation_dim": 146,
        "action_encoding": "discrete27",
        "action_dim": 27,
        "goal_conditioning": False,
    }
    protocol = Path("experiments/easy_bot_v0/protocol.md").read_text(encoding="utf-8")
    assert protocol.count("--device cuda") == 1  # TensorRT/CUDA preflight only.
    assert protocol.count("--device cpu") == 12
    assert manifest["aggregate"] == {
        "samples": 7938,
        "episodes": 200,
        "size_bytes": 986780,
    }
