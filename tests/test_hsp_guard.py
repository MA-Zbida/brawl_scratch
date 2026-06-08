from __future__ import annotations

import argparse
import csv

from train.hsp_guard import hsp_readiness_failure


def _args(**overrides) -> argparse.Namespace:
    data = {
        "allow_legacy_hsp": False,
        "llc_retention_csv": "",
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


def _passing_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for phase in (
        "recovery_mastery",
        "movement_fluency",
        "weapon_acquisition",
        "spacing_neutral",
        "combat_execution",
        "all_skills_llc",
    ):
        rows.append(
            {
                "phase": phase,
                "skill_score": 0.8,
                "best_skill_score": 0.8,
                "retention": 1.0,
                "amnesia": 0.0,
                "idle_rate": 0.1,
                "whiff_rate": 0.1,
                "mean_damage_trade": 0.05,
            }
        )
    return rows


def test_hsp_guard_requires_retention_csv_by_default() -> None:
    failure = hsp_readiness_failure(_args())
    assert "HSP is deferred" in failure


def test_hsp_guard_allows_explicit_legacy_override() -> None:
    failure = hsp_readiness_failure(_args(allow_legacy_hsp=True))
    assert failure == ""


def test_hsp_guard_passes_when_all_skills_retention_gate_passes(tmp_path) -> None:
    csv_path = tmp_path / "eval.csv"
    _write_eval(csv_path, _passing_rows())

    failure = hsp_readiness_failure(_args(llc_retention_csv=str(csv_path)))

    assert failure == ""


def test_hsp_guard_blocks_failed_retention_gate(tmp_path) -> None:
    csv_path = tmp_path / "eval.csv"
    rows = _passing_rows()
    rows[0]["retention"] = 0.7
    rows[0]["amnesia"] = 0.3
    _write_eval(csv_path, rows)

    failure = hsp_readiness_failure(_args(llc_retention_csv=str(csv_path)))

    assert "has not passed" in failure
    assert "recovery_mastery" in failure
