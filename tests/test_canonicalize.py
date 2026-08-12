"""Mirror canonicalisation.

The point of canonicalisation is that two situations which are reflections of
each other produce the *same* observation, so the policy learns one skill instead
of two. These tests pin that invariant, plus the two ways it can silently break:
the action not being flipped back, and the mirror decision chattering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extractor.memory.canonicalize import (
    DEFAULT_DEADBAND,
    STAGE_CENTER_X,
    GOAL_MIRROR_FEATURES,
    mirror_dynamic_block,
    mirror_goal_target,
    mirror_state_vector,
    should_mirror,
)
from action_space import ACTION_DIM, KEY_LEFT, KEY_RIGHT, Action, describe, to_keys
from feature_extractor.memory.state_spec import StateSpec


def _state(**values: float) -> np.ndarray:
    """All-zero is NOOP under the decomposed action encoding, so no priming needed."""
    vec = np.zeros((StateSpec.dim(),), dtype=np.float32)
    for name, value in values.items():
        vec[StateSpec.index(name)] = value
    return vec


# ── the core invariant ──────────────────────────────────────────────────────

def _reflect(x: float) -> float:
    return (2.0 * STAGE_CENTER_X) - x


def test_mirrored_pair_produces_identical_canonical_state():
    """A situation and its true reflection collapse to one canonical observation.

    The reflection axis is the stage centre, not 0.5 -- the calibrated platform is
    not centred on the screen.
    """
    right = _state(player_x=0.40, opponent_x=0.60, rel_dx=0.20, player_vx=0.1, rel_vx=-0.05)
    left = _state(
        player_x=_reflect(0.40), opponent_x=_reflect(0.60),
        rel_dx=-0.20, player_vx=-0.1, rel_vx=0.05,
    )

    np.testing.assert_allclose(mirror_state_vector(left), right, atol=1e-6)


def test_mirror_is_an_involution():
    original = _state(
        player_x=0.3, opponent_x=0.8, rel_dx=0.5, player_vx=-0.4,
        weapon_dx=0.2, signed_dx_to_ledge=-0.15, prev_hdir=1.0,
    )
    np.testing.assert_allclose(mirror_state_vector(mirror_state_vector(original)), original, atol=1e-6)


# ── what does and does not flip ─────────────────────────────────────────────

def test_absolute_positions_reflect_about_the_stage_centre():
    out = mirror_state_vector(_state(player_x=0.25, opponent_x=0.90))
    assert StateSpec.get(out, "player_x") == pytest.approx(_reflect(0.25))
    assert StateSpec.get(out, "opponent_x") == pytest.approx(_reflect(0.90))


def test_stage_centre_is_not_the_screen_centre():
    """Guards the reason the reflection axis is what it is."""
    assert STAGE_CENTER_X != pytest.approx(0.5, abs=1e-4)


def test_signed_horizontals_negate():
    out = mirror_state_vector(_state(rel_dx=0.3, weapon_dx=-0.2, signed_dx_to_ledge=0.1, rel_vx=0.4))
    assert StateSpec.get(out, "rel_dx") == pytest.approx(-0.3)
    assert StateSpec.get(out, "weapon_dx") == pytest.approx(0.2)
    assert StateSpec.get(out, "signed_dx_to_ledge") == pytest.approx(-0.1)
    assert StateSpec.get(out, "rel_vx") == pytest.approx(-0.4)


def test_vertical_and_scalar_features_are_untouched():
    out = mirror_state_vector(_state(
        player_y=0.7, player_vy=0.3, rel_dy=-0.2, rel_distance=0.44,
        player_damage_pct=0.6, player_h=0.09, player_w=0.05,
    ))
    assert StateSpec.get(out, "player_y") == pytest.approx(0.7)
    assert StateSpec.get(out, "player_vy") == pytest.approx(0.3)
    assert StateSpec.get(out, "rel_dy") == pytest.approx(-0.2)
    assert StateSpec.get(out, "rel_distance") == pytest.approx(0.44)
    assert StateSpec.get(out, "player_damage_pct") == pytest.approx(0.6)
    assert StateSpec.get(out, "player_w") == pytest.approx(0.05)


def test_previous_action_direction_flips_with_the_frame():
    """The recorded action must mirror with the state that it produced.

    Leaving prev_hdir unflipped would have a history slice claim the agent moved
    toward the opponent while the mirrored positions show it moving away.
    """
    out = mirror_state_vector(_state(prev_hdir=1.0))
    assert StateSpec.get(out, "prev_hdir") == pytest.approx(-1.0)


def test_non_directional_action_components_are_untouched():
    out = mirror_state_vector(_state(prev_jump=1.0, prev_light=1.0, prev_vdir=1.0))
    assert StateSpec.get(out, "prev_jump") == pytest.approx(1.0)
    assert StateSpec.get(out, "prev_light") == pytest.approx(1.0)
    assert StateSpec.get(out, "prev_vdir") == pytest.approx(1.0)


# ── the action half, now handled by action_space.to_keys ────────────────────

@pytest.mark.parametrize("action", [Action.MOVE_TOWARD, Action.LIGHT_TOWARD, Action.DODGE_DOWN_TOWARD])
def test_canonical_action_becomes_the_opposite_key_when_mirrored(action):
    """Canonical TOWARD is physical right normally, physical left when mirrored."""
    held_normal, _ = to_keys(int(action), mirrored=False)
    held_mirrored, _ = to_keys(int(action), mirrored=True)
    assert KEY_RIGHT in held_normal and KEY_LEFT not in held_normal
    assert KEY_LEFT in held_mirrored and KEY_RIGHT not in held_mirrored


def test_mirroring_an_action_changes_only_the_horizontal_key():
    for action in range(ACTION_DIM):
        held_n, tap_n = to_keys(action, mirrored=False)
        held_m, tap_m = to_keys(action, mirrored=True)
        assert tap_n == tap_m, f"{describe(action)} changed its taps under mirroring"
        assert held_n - {KEY_LEFT, KEY_RIGHT} == held_m - {KEY_LEFT, KEY_RIGHT}


# ── the decision, including hysteresis ──────────────────────────────────────

def test_mirrors_when_opponent_is_on_the_left():
    assert should_mirror(rel_dx=-0.5, opponent_exists=True, signed_dx_to_stage_center=0.0) is True


def test_does_not_mirror_when_opponent_is_on_the_right():
    assert should_mirror(rel_dx=0.5, opponent_exists=True, signed_dx_to_stage_center=0.0) is False


def test_decision_is_held_inside_the_deadband():
    """Without hysteresis the frame would chatter whenever rel_dx crosses zero."""
    inside = DEFAULT_DEADBAND / 2.0
    assert should_mirror(rel_dx=inside, opponent_exists=True, signed_dx_to_stage_center=0.0, previous=True) is True
    assert should_mirror(rel_dx=-inside, opponent_exists=True, signed_dx_to_stage_center=0.0, previous=False) is False


def test_falls_back_to_stage_side_without_an_opponent():
    """Recovery and movement still benefit: both ledges become the same ledge."""
    assert should_mirror(rel_dx=0.0, opponent_exists=False, signed_dx_to_stage_center=0.5) is False
    assert should_mirror(rel_dx=0.0, opponent_exists=False, signed_dx_to_stage_center=-0.5) is True


# ── goal space and history slices use the same convention ───────────────────

def test_goal_target_mirrors_horizontal_dimensions_only():
    names = ["signed_dx_to_ledge", "dy_to_ledge", "player_x", "player_y", "rel_distance"]
    target = np.array([0.2, 0.3, 0.25, 0.8, 0.4], dtype=np.float32)

    out = mirror_goal_target(target, names)

    assert out[0] == pytest.approx(0.8)   # signed_dx_to_ledge -> 1 - v
    assert out[1] == pytest.approx(0.3)   # dy_to_ledge unchanged
    assert out[2] == pytest.approx(_reflect(0.25))  # player_x -> reflected
    assert out[3] == pytest.approx(0.8)   # player_y unchanged
    assert out[4] == pytest.approx(0.4)   # rel_distance unchanged


def test_goal_mirror_set_is_a_subset_of_state_features():
    for name in GOAL_MIRROR_FEATURES:
        assert name in StateSpec.names()


def test_dynamic_block_mirror_matches_full_state_mirror():
    """History slices must use the same convention as the current frame."""
    full = _state(player_x=0.3, opponent_x=0.7, rel_dx=0.4, player_vx=0.2, rel_vx=-0.1)
    dynamic = full[: StateSpec.dynamic_dim()].copy()

    np.testing.assert_allclose(
        mirror_dynamic_block(dynamic),
        mirror_state_vector(full)[: StateSpec.dynamic_dim()],
        atol=1e-6,
    )
