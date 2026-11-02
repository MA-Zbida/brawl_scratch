"""Visual evidence must count classifier refreshes, not consumer reads."""
from __future__ import annotations

import pytest

from perception.weapon_evidence import WeaponEvidence, WeaponHoldGate


def reading(armed=True, *, fresh=True, valid=True, confidence=.99):
    return {
        "player_weapon_visual_armed": float(armed),
        "player_weapon_visual_fresh": float(fresh),
        "player_weapon_visual_valid": float(valid),
        "player_weapon_visual_confidence": confidence,
    }


def test_low_confidence_is_unknown_not_a_disarm():
    evidence = WeaponEvidence()
    assert evidence.update(reading()) is True
    assert evidence.update(reading(False, confidence=.6)) is True
    assert evidence.fresh_value is None


def test_confirmation_expires_without_fresh_evidence():
    evidence = WeaponEvidence(max_age_steps=4)
    evidence.update(reading())
    for _ in range(4):
        assert evidence.update(reading(fresh=False)) is True
    assert evidence.update(reading(fresh=False)) is None


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -.1, 1.1])
def test_invalid_confidence_cannot_confirm(confidence):
    assert WeaponEvidence().update(reading(confidence=confidence)) is None


def test_cached_negative_is_one_drop_vote_not_four():
    gate = WeaponHoldGate(required_steps=6, drop_grace=1)
    gate.update(True, reading())
    gate.update(True, reading())
    gate.update(False, reading(False))
    for _ in range(3):
        gate.update(False, reading(False, fresh=False))
    assert gate.negative_refreshes == 1
    assert not gate.dropped
    for _ in range(6):
        gate.update(True, reading())
    assert gate.success


def test_independent_confident_negatives_reject_real_loss():
    gate = WeaponHoldGate(required_steps=40, drop_grace=1)
    gate.update(True, reading())
    gate.update(True, reading())
    gate.update(False, reading(False))
    gate.update(False, reading(False))
    assert gate.dropped
    gate.update(True, reading())
    assert not gate.success  # A later re-pickup cannot rescue a failed hold.


def test_hold_completion_requires_fresh_positive_not_cached_threshold_crossing():
    gate = WeaponHoldGate(required_steps=4)
    gate.update(True, reading())
    gate.update(True, reading())
    for _ in range(3):
        gate.update(True, reading(fresh=False))
    assert not gate.success
    gate.update(True, reading())
    assert gate.success


def test_invalid_crops_never_count_as_drops_or_complete_a_hold():
    gate = WeaponHoldGate(required_steps=40)
    gate.update(True, reading())
    for _ in range(50):
        gate.update(True, reading(False, valid=False))
    assert not gate.dropped
    assert not gate.success
    assert gate.hold_steps == 0


def test_explicit_action_only_mode_keeps_nonvisual_contract():
    gate = WeaponHoldGate(required_steps=3, require_visual=False)
    for _ in range(3):
        gate.update(True, {})
    assert gate.success


@pytest.mark.parametrize("kwargs", [{"required_steps": 0}, {"drop_grace": -1}])
def test_invalid_hold_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        WeaponHoldGate(**kwargs)


def test_reset_forgets_previous_episode_evidence():
    evidence = WeaponEvidence()
    evidence.update(reading())
    evidence.reset()
    assert evidence.update(reading(fresh=False)) is None


def test_cached_armed_cannot_start_pickup_evidence():
    gate = WeaponHoldGate()
    gate.update(True, reading(fresh=False))
    assert not gate.pickup_seen


@pytest.mark.parametrize("kwargs", [{"min_confidence": 1.1}, {"max_age_steps": -1}])
def test_invalid_evidence_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        WeaponEvidence(**kwargs)
