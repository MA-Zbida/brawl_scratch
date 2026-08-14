"""Tests for goal-free final-agent match accounting."""

from __future__ import annotations

from pathlib import Path

from train.run_match import MatchResult


def test_match_harness_has_one_policy_and_no_goal_scheduler() -> None:
    source = Path("train/run_match.py").read_text(encoding="utf-8")
    assert "select_skill" not in source
    assert "StageGoalEnv" not in source
    assert "goal_target" not in source
    assert "--model" in source


def test_match_result_reports_trade_and_rates() -> None:
    result = MatchResult(match=1, steps=100, damage_dealt=120.0, damage_taken=45.0,
                         weapon_steps=60, offstage_steps=10)
    row = result.row()

    assert row["damage_trade"] == 75.0
    assert row["weapon_uptime"] == 0.6
    assert row["offstage_rate"] == 0.1
    assert row["environment_steps"] == 100


def test_match_result_row_never_divides_by_zero() -> None:
    """A match that dies on step 0 must still produce a scorable row."""
    row = MatchResult(match=1, steps=0).row()
    assert row["weapon_uptime"] == 0.0
    assert row["damage_trade"] == 0.0


def test_match_row_reports_interaction_time_and_outcome() -> None:
    result = MatchResult(match=1, steps=900, seconds=30.0, ended="won",
                         stocks_taken=1.0, damage_dealt=210.0, damage_taken=88.0)
    row = result.row()

    assert row["ended"] == "won"
    assert row["environment_steps"] == 900
    assert row["interaction_seconds"] == 30.0
    assert row["steps_per_second"] == 30.0
    assert row["damage_trade"] == 122.0
