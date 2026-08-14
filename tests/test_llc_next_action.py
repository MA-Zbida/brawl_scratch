from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from action_space import ACTION_DIM, Action
from feature_extractor.memory.state_spec import StateSpec

from tools.llc_next_action import build_advice


def _args(tmp_path, **overrides) -> argparse.Namespace:
    data = {
        "phase": "auto",
        "models_dir": str(tmp_path / "models"),
        "outputs_dir": str(tmp_path / "outputs"),
        "device": "$device",
        "best_scores": str(tmp_path / "models" / "llc_retention_best.json"),
        "python": "python",
        "bc_epochs": 20,
        "eval_episodes": 5,
        "all_skills_eval_episodes": 10,
        "timesteps": 0,
        "min_samples": 100,
        "allow_demo_warnings": False,
        "require_plots": True,
        "require_report": True,
        "require_manual_approval": True,
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def _write_demo(models_dir, phase: str, *, idle: bool = False) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    n = 120
    obs = np.zeros((n, StateSpec.observation_dim((2, 4, 8)) + 22), dtype=np.float32)
    actions = np.zeros((n,), dtype=np.int64)
    if idle:
        actions[:] = int(Action.NOOP)
    else:
        actions[:] = np.arange(n) % ACTION_DIM
    dones = np.zeros((n,), dtype=bool)
    dones[-1] = True
    goal_mask = np.ones((n, 11), dtype=np.float32)
    np.savez_compressed(
        models_dir / f"{phase}_demos.npz",
        obs=obs,
        actions=actions,
        actions_discrete=actions,
        dones=dones,
        goal_mask=goal_mask,
        phase=np.asarray([phase]),
        episodes_collected=np.asarray([1], dtype=np.int64),
    )


def _write_eval(models_dir, phase: str, rows) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "phase",
        "skill_score",
        "best_skill_score",
        "retention",
        "amnesia",
        "idle_rate",
        "whiff_rate",
        "mean_damage_trade",
    ]
    with (models_dir / f"llc_{phase}_retention_eval.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot_set(models_dir, phase: str) -> None:
    suffixes = (
        "retention_amnesia",
        "goal_family_errors",
        "goal_feature_traces",
        "goal_phase_spaces",
        "episode_health",
        "combat_precision",
    )
    for suffix in suffixes:
        (models_dir / f"llc_{phase}_{suffix}.png").write_bytes(b"fake")


def _write_report(outputs_dir, phase: str) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / f"llc_{phase}_run_report.md").write_text("report", encoding="utf-8")


def _write_observation(outputs_dir, phase: str, *, approved: bool = True) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase,
        "observed_at": "2026-06-08T00:00:00",
        "approved": approved,
        "checks": {
            "movement_collapse_visible": "no",
            "recovery_reliable": "yes",
            "weapon_intentional": "na",
            "spacing_safe": "na",
            "combat_clean": "na",
        },
        "bad_visual_flags": [],
        "notes": "looks good",
    }
    (outputs_dir / f"llc_{phase}_manual_observation.json").write_text(json.dumps(payload), encoding="utf-8")


def test_advisor_starts_with_perception_when_no_demos_exist(tmp_path) -> None:
    advice = build_advice(_args(tmp_path))

    assert advice.next_kind == "perception"
    assert "debug_observation_overlay.py" in advice.command
    assert "pip install -r requirements-llc.txt" in "\n".join(advice.details)
    assert "llc_preflight.py --device cuda" in "\n".join(advice.details)
    assert "collect_bc_locomotion_demos --phase movement_fluency" in "\n".join(advice.details)


def test_advisor_sends_valid_demo_to_bc_pretrain(tmp_path) -> None:
    models_dir = tmp_path / "models"
    _write_demo(models_dir, "movement_fluency")

    advice = build_advice(_args(tmp_path, phase="movement_fluency"))

    assert advice.next_kind == "bc_pretrain"
    assert "train.pretrain_bc_locomotion" in advice.command
    assert "--phase movement_fluency" in advice.command


def test_advisor_stops_on_demo_warning_by_default(tmp_path) -> None:
    models_dir = tmp_path / "models"
    _write_demo(models_dir, "movement_fluency", idle=True)

    advice = build_advice(_args(tmp_path, phase="movement_fluency"))

    assert advice.next_kind == "validate_demos"
    assert advice.exit_code == 2
    assert any("high idle rate" in item for item in advice.details)


def test_advisor_stops_on_failed_gate(tmp_path) -> None:
    models_dir = tmp_path / "models"
    _write_demo(models_dir, "movement_fluency")
    (models_dir / "llc_movement_fluency_bc_init.zip").write_bytes(b"fake")
    (models_dir / "llc_movement_fluency.zip").write_bytes(b"fake")
    _write_eval(
        models_dir,
        "movement_fluency",
        [
            {
                "phase": "movement_fluency",
                "skill_score": 0.40,
                "best_skill_score": 0.40,
                "retention": 1.0,
                "amnesia": 0.0,
                "idle_rate": 0.10,
                "whiff_rate": 0.0,
                "mean_damage_trade": 0.0,
            }
        ],
    )

    advice = build_advice(_args(tmp_path, phase="movement_fluency"))

    assert advice.next_kind == "gate_failed"
    assert advice.exit_code == 2
    assert any("current score" in item for item in advice.details)


def test_advisor_includes_monitor_hint_for_ppo_training(tmp_path) -> None:
    models_dir = tmp_path / "models"
    _write_demo(models_dir, "movement_fluency")
    (models_dir / "llc_movement_fluency_bc_init.zip").write_bytes(b"fake")

    advice = build_advice(_args(tmp_path, phase="movement_fluency"))

    assert advice.next_kind == "ppo_train"
    assert "train.train_curriculum" in advice.command
    assert any("llc_live_monitor.py --phase movement_fluency" in item for item in advice.details)


def test_advisor_requests_plots_then_report_after_gate_passes(tmp_path) -> None:
    models_dir = tmp_path / "models"
    _write_demo(models_dir, "movement_fluency")
    (models_dir / "llc_movement_fluency_bc_init.zip").write_bytes(b"fake")
    (models_dir / "llc_movement_fluency.zip").write_bytes(b"fake")
    _write_eval(
        models_dir,
        "movement_fluency",
        [
            {
                "phase": "movement_fluency",
                "skill_score": 0.80,
                "best_skill_score": 0.80,
                "retention": 1.0,
                "amnesia": 0.0,
                "idle_rate": 0.10,
                "whiff_rate": 0.0,
                "mean_damage_trade": 0.0,
            }
        ],
    )

    advice = build_advice(_args(tmp_path, phase="movement_fluency"))
    assert advice.next_kind == "plot"
    assert "plot_llc_diagnostics.py" in advice.command

    _write_plot_set(models_dir, "movement_fluency")
    advice = build_advice(_args(tmp_path, phase="movement_fluency"))
    assert advice.next_kind == "report"
    assert "summarize_llc_run.py" in advice.command

    _write_report(tmp_path / "outputs", "movement_fluency")
    advice = build_advice(_args(tmp_path, phase="movement_fluency"))
    assert advice.next_kind == "manual_observation"
    assert "record_llc_observation.py" in advice.command


def test_advisor_blocks_failed_manual_observation(tmp_path) -> None:
    models_dir = tmp_path / "models"
    outputs_dir = tmp_path / "outputs"
    _write_demo(models_dir, "movement_fluency")
    (models_dir / "llc_movement_fluency_bc_init.zip").write_bytes(b"fake")
    (models_dir / "llc_movement_fluency.zip").write_bytes(b"fake")
    _write_eval(
        models_dir,
        "movement_fluency",
        [
            {
                "phase": "movement_fluency",
                "skill_score": 0.80,
                "best_skill_score": 0.80,
                "retention": 1.0,
                "amnesia": 0.0,
                "idle_rate": 0.10,
                "whiff_rate": 0.0,
                "mean_damage_trade": 0.0,
            }
        ],
    )
    _write_plot_set(models_dir, "movement_fluency")
    _write_report(outputs_dir, "movement_fluency")
    _write_observation(outputs_dir, "movement_fluency", approved=False)

    advice = build_advice(_args(tmp_path, phase="movement_fluency"))

    assert advice.next_kind == "manual_observation"
    assert advice.exit_code == 2
    assert any("do not advance" in item for item in advice.details)
