from __future__ import annotations

import argparse
import csv

from tools.summarize_llc_run import build_report


def _write_eval(path, rows) -> None:
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
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _args(tmp_path, phase: str, eval_csv, **overrides) -> argparse.Namespace:
    data = {
        "phase": phase,
        "eval_csv": str(eval_csv),
        "phases": "",
        "models_dir": str(tmp_path),
        "prefix": f"llc_{phase}",
        "include_demo_check": False,
        "amnesia_threshold": 0.15,
        "min_retention": 0.85,
        "min_current_score": None,
        "max_idle_rate": 0.45,
        "max_combat_whiff_rate": 0.80,
        "min_combat_damage_trade": 0.0,
        "no_combat_trade_gate": False,
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def test_run_report_marks_advance_and_plot_evidence(tmp_path) -> None:
    eval_csv = tmp_path / "eval.csv"
    _write_eval(
        eval_csv,
        [
            {
                "phase": "recovery_mastery",
                "skill_score": 0.8,
                "best_skill_score": 0.8,
                "retention": 1.0,
                "amnesia": 0.0,
                "idle_rate": 0.1,
                "whiff_rate": 0.0,
                "mean_damage_trade": 0.0,
            }
        ],
    )
    (tmp_path / "llc_recovery_mastery_retention_amnesia.png").write_bytes(b"fake")

    report = build_report(_args(tmp_path, "recovery_mastery", eval_csv, phases="recovery_mastery"))

    assert "Gate: **ADVANCE**" in report
    assert "`recovery_mastery`" in report
    assert "Current Min" in report
    assert "| `retention_amnesia` | yes |" in report
    assert "record_llc_observation.py --phase recovery_mastery" in report
    assert "Movement collapse visible?" in report


def test_run_report_includes_failure_recommendations(tmp_path) -> None:
    eval_csv = tmp_path / "eval.csv"
    _write_eval(
        eval_csv,
        [
            {
                "phase": "movement_fluency",
                "skill_score": 0.6,
                "best_skill_score": 0.8,
                "retention": 0.75,
                "amnesia": 0.25,
                "idle_rate": 0.1,
                "whiff_rate": 0.0,
                "mean_damage_trade": 0.0,
            },
            {
                "phase": "weapon_acquisition",
                "skill_score": 0.7,
                "best_skill_score": 0.7,
                "retention": 1.0,
                "amnesia": 0.0,
                "idle_rate": 0.1,
                "whiff_rate": 0.0,
                "mean_damage_trade": 0.0,
            },
        ],
    )

    report = build_report(_args(tmp_path, "weapon_acquisition", eval_csv))

    assert "Gate: **STOP**" in report
    assert "Skill collapse detected" in report
    assert "movement_fluency: amnesia" in report
