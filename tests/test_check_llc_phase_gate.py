from __future__ import annotations

import argparse
import csv

from tools.check_llc_phase_gate import evaluate_gate, recommend_actions


def _write_eval_csv(path, rows) -> None:
    fieldnames = [
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
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _args(path, phase: str, phases: str = "") -> argparse.Namespace:
    return argparse.Namespace(
        eval_csv=str(path),
        phase=phase,
        phases=phases,
        amnesia_threshold=0.15,
        min_retention=0.85,
        min_current_score=None,
        max_idle_rate=0.45,
        max_combat_whiff_rate=0.80,
        min_combat_damage_trade=0.0,
        no_combat_trade_gate=False,
    )


def test_phase_gate_passes_when_retention_and_collapse_metrics_pass(tmp_path) -> None:
    csv_path = tmp_path / "eval.csv"
    _write_eval_csv(
        csv_path,
        [
            {
                "phase": "recovery_mastery",
                "skill_score": 0.78,
                "best_skill_score": 0.80,
                "retention": 0.975,
                "amnesia": 0.025,
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
                "idle_rate": 0.12,
                "whiff_rate": 0.0,
                "mean_damage_trade": 0.0,
            },
        ],
    )

    passed, failures, table = evaluate_gate(_args(csv_path, "movement_fluency"))

    assert passed
    assert failures == []
    assert [row["status"] for row in table] == ["PASS", "PASS"]


def test_phase_gate_fails_on_previous_phase_amnesia(tmp_path) -> None:
    csv_path = tmp_path / "eval.csv"
    _write_eval_csv(
        csv_path,
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
                "idle_rate": 0.12,
                "whiff_rate": 0.0,
                "mean_damage_trade": 0.0,
            },
        ],
    )

    passed, failures, _ = evaluate_gate(_args(csv_path, "movement_fluency"))

    assert not passed
    assert any("recovery_mastery" in failure and "amnesia" in failure for failure in failures)

    _, _, table = evaluate_gate(_args(csv_path, "movement_fluency"))
    actions = recommend_actions(failures, table, current_phase="movement_fluency")
    assert any("Skill collapse detected" in action for action in actions)
    assert any("--bc-demos-path" in action for action in actions)


def test_phase_gate_fails_on_combat_damage_trade(tmp_path) -> None:
    csv_path = tmp_path / "eval.csv"
    _write_eval_csv(
        csv_path,
        [
            {
                "phase": "combat_execution",
                "skill_score": 0.50,
                "best_skill_score": 0.50,
                "retention": 1.0,
                "amnesia": 0.0,
                "idle_rate": 0.10,
                "whiff_rate": 0.40,
                "mean_damage_trade": -0.05,
            },
        ],
    )

    passed, failures, _ = evaluate_gate(_args(csv_path, "combat_execution", "combat_execution"))

    assert not passed
    assert any("damage trade" in failure for failure in failures)

    _, _, table = evaluate_gate(_args(csv_path, "combat_execution", "combat_execution"))
    actions = recommend_actions(failures, table, current_phase="combat_execution")
    assert any("Negative damage trade" in action for action in actions)


def test_phase_gate_fails_on_weak_current_phase_score(tmp_path) -> None:
    csv_path = tmp_path / "eval.csv"
    _write_eval_csv(
        csv_path,
        [
            {
                "phase": "weapon_acquisition",
                "skill_score": 0.40,
                "best_skill_score": 0.40,
                "retention": 1.0,
                "amnesia": 0.0,
                "idle_rate": 0.10,
                "whiff_rate": 0.0,
                "mean_damage_trade": 0.0,
            },
        ],
    )

    passed, failures, table = evaluate_gate(_args(csv_path, "weapon_acquisition", "weapon_acquisition"))

    assert not passed
    assert table[0]["current_threshold"] == 0.60
    assert any("current score" in failure for failure in failures)


def test_phase_gate_reads_utf8_sig_csv_headers(tmp_path) -> None:
    csv_path = tmp_path / "eval_sig.csv"
    csv_path.write_text(
        "phase,skill_score,best_skill_score,retention,amnesia,idle_rate,whiff_rate,mean_damage_trade\n"
        "recovery_mastery,0.8,0.8,1.0,0.0,0.1,0.0,0.0\n",
        encoding="utf-8-sig",
    )

    passed, failures, table = evaluate_gate(_args(csv_path, "recovery_mastery", "recovery_mastery"))

    assert passed
    assert failures == []
    assert table[0]["phase"] == "recovery_mastery"
