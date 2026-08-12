"""Identity resolution under the 3-class detector schema.

The agent is the character carrying the blue self-indicator. These tests pin the
behaviour that matters: the overlap case that defeats nearest-neighbour
association, and the explicit failure mode when the indicator is not detected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extractor.memory.detection_schema import (
    CURRENT_CLASS_NAMES,
    LEGACY_CLASS_NAMES,
    detect_schema,
    match_indicator_to_character,
    resolve,
)


def det(class_name: str, x: float, y: float, w: float = 0.05, h: float = 0.09, conf: float = 0.9) -> dict:
    return {"class_name": class_name, "bbox": [x, y, w, h], "confidence": conf}


def test_class_order_matches_data_yaml():
    """Must stay in lockstep with data.yaml: nc=3, names in this exact order."""
    assert CURRENT_CLASS_NAMES == ("character", "indicator_self", "weapon")
    assert LEGACY_CLASS_NAMES[0] == "agent"


def test_schema_is_inferred_not_configured():
    assert detect_schema([det("character", 0.5, 0.6)]) == "current"
    assert detect_schema([det("agent", 0.5, 0.6)]) == "legacy"
    assert detect_schema([]) == "none"


def test_agent_is_the_character_under_the_indicator():
    left = det("character", 0.40, 0.60)
    right = det("character", 0.55, 0.60)
    indicator = det("indicator_self", 0.55, 0.53, w=0.02, h=0.02)

    result = resolve([left, right, indicator])

    assert result.agent is right
    assert result.opponent is left
    assert result.identity_source == "indicator"
    assert result.identity_is_observed


def test_overlapping_fighters_resolved_by_indicator_not_proximity():
    """The case that defeats nearest-to-last-position association.

    Both characters are nearly coincident and the agent's last known position is
    closer to the *opponent*. Only the indicator disambiguates correctly.
    """
    opponent = det("character", 0.500, 0.60)
    agent = det("character", 0.530, 0.60)
    indicator = det("indicator_self", 0.530, 0.53, w=0.02, h=0.02)

    result = resolve(
        [opponent, agent, indicator],
        last_agent_xy=(0.498, 0.60),   # misleadingly close to the opponent
        last_opponent_xy=(0.520, 0.60),
    )

    assert result.agent is agent, "indicator must win over positional proximity"
    assert result.opponent is opponent


def test_missing_indicator_falls_back_and_flags_identity_as_stale():
    a = det("character", 0.40, 0.60)
    b = det("character", 0.62, 0.60)

    result = resolve([a, b], last_agent_xy=(0.61, 0.60))

    assert result.agent is b
    assert result.identity_source == "carry_forward"
    assert not result.identity_is_observed, "stale identity must not read as observed"
    assert result.indicator_score == float("inf")


def test_no_indicator_and_no_history_yields_no_agent():
    """Better to report no agent than to guess silently."""
    result = resolve([det("character", 0.4, 0.6), det("character", 0.6, 0.6)])
    assert result.agent is None
    assert result.identity_source == "none"


def test_character_above_the_indicator_is_rejected():
    """The indicator floats above its own character, never below it."""
    above = det("character", 0.50, 0.40)
    indicator = det("indicator_self", 0.50, 0.53, w=0.02, h=0.02)

    matched, score = match_indicator_to_character([above], [indicator])
    assert matched is None
    assert score == float("inf")


def test_distant_indicator_does_not_match():
    far = det("character", 0.10, 0.60)
    indicator = det("indicator_self", 0.90, 0.53, w=0.02, h=0.02)

    matched, _ = match_indicator_to_character([far], [indicator])
    assert matched is None


def test_multiple_weapons_all_returned():
    """Ground weapons are many; all are reported, nearest-selection is Memory's job."""
    result = resolve([
        det("character", 0.5, 0.6),
        det("indicator_self", 0.5, 0.53, w=0.02, h=0.02),
        det("weapon", 0.2, 0.7),
        det("weapon", 0.8, 0.7),
        det("weapon", 0.35, 0.7),
    ])
    assert len(result.weapons) == 3


def test_multiple_opponents_are_all_returned():
    """Roster growth must not need new classes."""
    agent = det("character", 0.50, 0.60)
    result = resolve([
        agent,
        det("indicator_self", 0.50, 0.53, w=0.02, h=0.02),
        det("character", 0.20, 0.60),
        det("character", 0.80, 0.60),
        det("character", 0.65, 0.40),
    ])
    assert result.agent is agent
    assert len(result.opponents) == 3
    assert agent not in result.opponents


def test_legacy_schema_still_resolves():
    """The 5-class engine keeps working until the new weights are swapped in."""
    result = resolve([
        det("agent", 0.4, 0.6),
        det("op1", 0.7, 0.6),
        det("weapons", 0.2, 0.7),
    ])
    assert result.schema == "legacy"
    assert result.agent is not None
    assert result.opponent is not None
    assert result.opponent["class_name"] == "op1"
    assert len(result.weapons) == 1
    assert result.identity_is_observed


@pytest.mark.parametrize("indicator_x", [0.30, 0.50, 0.72])
def test_indicator_selects_correct_character_across_positions(indicator_x):
    characters = [det("character", x, 0.60) for x in (0.30, 0.50, 0.72)]
    indicator = det("indicator_self", indicator_x, 0.53, w=0.02, h=0.02)

    result = resolve(characters + [indicator])
    assert result.agent is not None
    assert result.agent["bbox"][0] == pytest.approx(indicator_x)
