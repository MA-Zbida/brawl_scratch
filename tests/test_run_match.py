"""Tests for the scripted goal selector in the match harness.

The selector is the part of the milestone run that must be trustworthy: if it
picks the wrong skill, the video shows a bad agent and the cause is the harness,
not the policy. Its rules are pure functions of the observation, so they test
without a game.
"""

from __future__ import annotations

import numpy as np

from feature_extractor.memory.state_spec import StateSpec
from train.run_match import SKILLS, MatchResult, SelectorConfig, select_skill


def _obs(**features: float) -> np.ndarray:
    obs = np.zeros(StateSpec.observation_dim((2, 4, 8)) + 22, dtype=np.float32)
    for name, value in features.items():
        obs[StateSpec.index(name)] = value
    return obs


def test_offstage_outranks_everything() -> None:
    """Being offstage is the only state that ends the match outright."""
    obs = _obs(player_is_offstage=1.0, player_has_weapon=0.0, rel_distance=0.5)
    assert select_skill(obs, SelectorConfig()) == "recovery_mastery"


def test_unarmed_agent_seeks_a_weapon() -> None:
    obs = _obs(player_is_offstage=0.0, player_has_weapon=0.0, rel_distance=0.15)
    assert select_skill(obs, SelectorConfig()) == "weapon_acquisition"


def test_armed_and_at_fighting_range_commits_to_combat() -> None:
    obs = _obs(player_is_offstage=0.0, player_has_weapon=1.0, rel_distance=0.15)
    assert select_skill(obs, SelectorConfig()) == "combat_execution"


def test_too_close_and_too_far_both_route_to_spacing() -> None:
    cfg = SelectorConfig()
    close = _obs(player_has_weapon=1.0, rel_distance=cfg.too_close - 0.01)
    far = _obs(player_has_weapon=1.0, rel_distance=cfg.too_far + 0.01)
    assert select_skill(close, cfg) == "spacing_neutral"
    assert select_skill(far, cfg) == "spacing_neutral"


def test_every_returned_skill_is_one_the_harness_can_load() -> None:
    """A typo here would silently route every step to the fallback policy."""
    cfg = SelectorConfig()
    for offstage in (0.0, 1.0):
        for weapon in (0.0, 1.0):
            for distance in (0.0, 0.05, 0.15, 0.5, 1.0):
                obs = _obs(player_is_offstage=offstage, player_has_weapon=weapon,
                           rel_distance=distance)
                assert select_skill(obs, cfg) in SKILLS


def test_match_result_reports_trade_and_rates() -> None:
    result = MatchResult(match=1, steps=100, damage_dealt=120.0, damage_taken=45.0,
                         weapon_steps=60, offstage_steps=10)
    result.skill_steps["combat_execution"] = 70
    row = result.row()

    assert row["damage_trade"] == 75.0
    assert row["weapon_uptime"] == 0.6
    assert row["offstage_rate"] == 0.1
    assert row["pct_combat_execution"] == 0.7


def test_match_result_row_never_divides_by_zero() -> None:
    """A match that dies on step 0 must still produce a scorable row."""
    row = MatchResult(match=1, steps=0).row()
    assert row["weapon_uptime"] == 0.0
    assert row["damage_trade"] == 0.0


def test_match_row_reports_env_resets_and_outcome() -> None:
    """A match spans many env episodes; the count and outcome must be visible.

    The first live run produced 2-3 step matches because the loop broke on the
    training wrapper's `terminated`. A match now ends only on stocks or the step
    cap, and `env_episodes` records how many wrapper episodes it absorbed -- if
    that number is large, the wrapper is churning and the run is worth a look.
    """
    result = MatchResult(match=1, steps=900, env_episodes=14, ended="won",
                         stocks_taken=1.0, damage_dealt=210.0, damage_taken=88.0)
    row = result.row()

    assert row["ended"] == "won"
    assert row["env_episodes"] == 14
    assert row["damage_trade"] == 122.0
