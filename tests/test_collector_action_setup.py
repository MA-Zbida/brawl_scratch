"""Collector setup must work for every phase.

`main()` crashed on `spec.allowed_attack_actions`, a field removed when the action
space became a single Discrete(27). Nothing caught it: the entry-point tests run
`--help`, which exits long before that line, so the collector body had no coverage at
all despite being the next thing to run.

These exercise the phase-dependent setup directly, without a live environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from action_space import ACTION_DIM, Action, components
from train.collect_bc_locomotion_demos import resolve_allowed_actions
from train.curriculum_config import PHASES, build_phase_spec


@pytest.mark.parametrize("phase", PHASES)
def test_setup_resolves_for_every_phase(phase):
    """The regression: this raised AttributeError for combat_execution."""
    spec = build_phase_spec(phase, terminate_on_death=False)
    allowed = resolve_allowed_actions(spec)

    assert allowed, f"{phase} allows no actions at all"
    assert all(0 <= a < ACTION_DIM for a in allowed)
    assert int(Action.NOOP) in allowed, "doing nothing must always be available"


@pytest.mark.parametrize("phase", PHASES)
def test_restrictions_are_actually_applied(phase):
    spec = build_phase_spec(phase, terminate_on_death=False)
    allowed = resolve_allowed_actions(spec)

    if spec.allowed_actions is not None:
        return   # explicit list wins; nothing to derive

    for action in allowed:
        comp = components(action)
        if spec.disable_attack:
            assert not (comp.light or comp.heavy), f"{Action(action).name} attacks in {phase}"
        if spec.disable_dodge:
            assert not comp.dodge, f"{Action(action).name} dodges in {phase}"
        if spec.disable_jump:
            assert not comp.jump, f"{Action(action).name} jumps in {phase}"


def test_weapon_phase_keeps_pickup_and_drops_attacks():
    spec = build_phase_spec("weapon_acquisition", terminate_on_death=False)
    allowed = resolve_allowed_actions(spec)

    assert int(Action.PICKUP) in allowed
    assert all(not (components(a).light or components(a).heavy) for a in allowed)


def test_combat_phase_keeps_attacks():
    spec = build_phase_spec("combat_execution", terminate_on_death=False)
    allowed = resolve_allowed_actions(spec)

    assert any(components(a).light for a in allowed)
    assert any(components(a).heavy for a in allowed)


def test_movement_phase_keeps_ground_dash_actions():
    spec = build_phase_spec("movement_fluency", terminate_on_death=False)
    allowed = resolve_allowed_actions(spec)

    assert int(Action.DODGE_TOWARD) in allowed
    assert int(Action.DODGE_AWAY) in allowed


def test_recovery_phase_has_exactly_locomotion_dodge_and_two_recovery_heavies():
    spec = build_phase_spec("recovery_mastery", terminate_on_death=False)
    allowed = resolve_allowed_actions(spec)

    assert len(allowed) == 20
    assert int(Action.HEAVY_NEUTRAL) in allowed
    assert int(Action.HEAVY_TOWARD) in allowed
    assert int(Action.HEAVY_AWAY) not in allowed
    assert int(Action.HEAVY_DOWN) not in allowed
    assert int(Action.PICKUP) not in allowed
    assert all(not components(action).light for action in allowed)


def test_no_phase_references_the_removed_field():
    """Guards the whole class of half-finished rename."""
    for phase in PHASES:
        spec = build_phase_spec(phase, terminate_on_death=False)
        assert not hasattr(spec, "allowed_attack_actions"), (
            "StageSpec still exposes allowed_attack_actions; the rename to "
            "allowed_actions was left incomplete"
        )
