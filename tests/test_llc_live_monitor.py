from __future__ import annotations

import argparse
import csv

from tools.llc_live_monitor import build_snapshot, summarize_episodes, summarize_steps


def _args(tmp_path, **overrides) -> argparse.Namespace:
    models_dir = tmp_path / "models"
    data = {
        "phase": "combat_execution",
        "models_dir": str(models_dir),
        "steps_csv": "",
        "episodes_csv": "",
        "eval_csv": "",
        "interval": 10.0,
        "once": True,
        "tail_steps": 500,
        "tail_episodes": 20,
        "max_idle_rate": 0.45,
        "max_whiff_rate": 0.80,
        "min_action_entropy": 0.15,
        "min_combat_damage_trade": 0.0,
        "min_retention": 0.85,
        "amnesia_threshold": 0.15,
        "fail_on_alert": False,
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def _write_csv(path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_step_and_episode_summaries_average_recent_rows() -> None:
    step_rows = [
        {"reward": "1.0", "goal_error": "0.2", "goal_success": "0", "idle": "1", "hit": "0", "whiff": "1", "damage_trade": "-0.1", "goal_type": "combat", "step": "1"},
        {"reward": "3.0", "goal_error": "0.1", "goal_success": "1", "idle": "0", "hit": "1", "whiff": "0", "damage_trade": "0.2", "goal_type": "combat", "step": "2"},
    ]
    ep_rows = [
        {"return": "5", "episode_success": "1", "success_ratio": "1", "mean_goal_error": "0.1", "time_to_success": "3", "action_entropy": "0.8", "idle_rate": "0.1", "whiff_rate": "0.2", "attack_precision": "0.8", "damage_trade": "1.0", "episode": "1"},
        {"return": "7", "episode_success": "0", "success_ratio": "0.5", "mean_goal_error": "0.3", "time_to_success": "0", "action_entropy": "0.6", "idle_rate": "0.2", "whiff_rate": "0.4", "attack_precision": "0.6", "damage_trade": "-0.5", "episode": "2"},
    ]

    step = summarize_steps(step_rows, 2)
    episode = summarize_episodes(ep_rows, 2)

    assert step["latest_step"] == 2
    assert step["reward"] == 2.0
    assert step["top_goals"] == "combat:2"
    assert episode["latest_episode"] == 2
    assert episode["success"] == 0.5
    assert episode["damage_trade"] == 0.25


def test_monitor_reports_waiting_when_csvs_are_missing(tmp_path) -> None:
    snapshot, alerts = build_snapshot(_args(tmp_path, phase="movement_fluency"))

    assert alerts == []
    assert "Step CSV missing or empty" in snapshot
    assert "no hard collapse signal" in snapshot


def test_monitor_alerts_on_episode_collapse_metrics(tmp_path) -> None:
    models_dir = tmp_path / "models"
    _write_csv(
        models_dir / "llc_combat_execution_episodes.csv",
        [
            "episode",
            "return",
            "episode_success",
            "success_ratio",
            "mean_goal_error",
            "time_to_success",
            "action_entropy",
            "idle_rate",
            "whiff_rate",
            "attack_precision",
            "damage_trade",
        ],
        [
            {
                "episode": 1,
                "return": -1,
                "episode_success": 0,
                "success_ratio": 0,
                "mean_goal_error": 0.5,
                "time_to_success": 0,
                "action_entropy": 0.05,
                "idle_rate": 0.70,
                "whiff_rate": 0.90,
                "attack_precision": 0.10,
                "damage_trade": -1.0,
            }
        ],
    )

    snapshot, alerts = build_snapshot(_args(tmp_path))

    assert any("Idle collapse risk" in item for item in alerts)
    assert any("Action collapse risk" in item for item in alerts)
    assert any("Attack spam risk" in item for item in alerts)
    assert any("Combat trade risk" in item for item in alerts)
    assert "STOP SIGNALS" in snapshot


def test_monitor_alerts_on_eval_gate_failure(tmp_path) -> None:
    models_dir = tmp_path / "models"
    _write_csv(
        models_dir / "llc_movement_fluency_eval.csv",
        [
            "phase",
            "skill_score",
            "best_skill_score",
            "retention",
            "amnesia",
            "idle_rate",
            "whiff_rate",
            "mean_damage_trade",
        ],
        [
            {
                "phase": "recovery_mastery",
                "skill_score": 0.60,
                "best_skill_score": 0.80,
                "retention": 0.75,
                "amnesia": 0.25,
                "idle_rate": 0.10,
                "whiff_rate": 0.0,
                "mean_damage_trade": 0.0,
            },
            {
                "phase": "movement_fluency",
                "skill_score": 0.70,
                "best_skill_score": 0.70,
                "retention": 1.0,
                "amnesia": 0.0,
                "idle_rate": 0.10,
                "whiff_rate": 0.0,
                "mean_damage_trade": 0.0,
            },
        ],
    )

    snapshot, alerts = build_snapshot(_args(tmp_path, phase="movement_fluency"))

    assert any("recovery_mastery" in item and "amnesia" in item for item in alerts)
    assert "Eval latest" in snapshot
    assert "recovery_mastery" in snapshot
