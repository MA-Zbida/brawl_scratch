from __future__ import annotations

from train.retention import (
    PHASE_ORDER,
    parse_phase_list,
    phase_score_threshold,
    previous_phases,
    retention_and_amnesia,
    skill_score_for_phase,
    update_best_scores,
)


def test_curriculum_defers_recovery_until_after_combat() -> None:
    assert PHASE_ORDER == (
        "movement_fluency",
        "weapon_acquisition",
        "spacing_neutral",
        "combat_execution",
        "recovery_mastery",
        "all_skills_llc",
    )


def test_retention_and_amnesia_are_ratio_based() -> None:
    retention, amnesia = retention_and_amnesia(0.72, 0.90)
    assert round(retention, 3) == 0.800
    assert round(amnesia, 3) == 0.200


def test_best_scores_only_increase() -> None:
    updated = update_best_scores({"movement_fluency": 0.8}, {"movement_fluency": 0.7, "combat_execution": 0.4})
    assert updated["movement_fluency"] == 0.8
    assert updated["combat_execution"] == 0.4


def test_phase_list_can_expand_previous_phases() -> None:
    phases = parse_phase_list("", "weapon_acquisition", include_previous=True)
    assert phases == ["movement_fluency", "weapon_acquisition"]


def test_all_skills_previous_phases_can_exclude_current() -> None:
    phases = previous_phases("all_skills_llc", include_current=False)
    assert phases == [
        "movement_fluency",
        "weapon_acquisition",
        "spacing_neutral",
        "combat_execution",
        "recovery_mastery",
    ]


def test_phase_score_thresholds_have_override_support() -> None:
    assert phase_score_threshold("weapon_acquisition") == 0.60
    assert phase_score_threshold("weapon_acquisition", override=0.42) == 0.42


def test_combat_score_uses_damage_and_hit_signals() -> None:
    score = skill_score_for_phase(
        "combat_execution",
        {
            "episode_success_rate": 0.4,
            "hit_rate": 0.5,
            "mean_damage_trade": 0.10,
            "win_rate": 0.25,
        },
    )
    assert 0.0 < score < 1.0
