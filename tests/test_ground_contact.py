"""Foot-position convention and ground contact.

Regression fixtures taken from real overlay captures, where the agent was standing on
the platform but read as airborne while the opponent beside it read as grounded.

Cause: the two fighters used different foot conventions. The agent's box centre was
shifted by a full ``height`` at detection time; the opponent's by ``height / 2`` at
ground-check time. The agent's foot therefore sat about half a body too low.

It broke two things at once, and only one was visible:

* ``player_grounded`` stayed 0 while standing on the platform, so every
  jump/edge/offstage feature derived from it was wrong;
* ``player_y`` was biased past grounded movement targets by more than the success
  radius, making those goals unreachable -- silently, since a goal that is never
  reached just looks like a policy that has not learned yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extractor.memory.structured_memory import Memory


def det(class_name: str, cx: float, cy: float, w: float, h: float, conf: float = 0.9) -> dict:
    return {"class_name": class_name, "bbox": [cx, cy, w, h], "confidence": conf}


def observe(player_center_y, player_h, opponent_center_y, opponent_h,
            player_x=0.5145, opponent_x=0.6420):
    """Feed one frame; centres are raw detector boxes, as the model emits them."""
    memory = Memory()
    memory.update_from_detections(
        [
            det("character", player_x, player_center_y, 0.0315, player_h),
            det("character", opponent_x, opponent_center_y, 0.0380, opponent_h),
            det("indicator_self", player_x, player_center_y - 0.06, 0.02, 0.02),
        ],
        dt=1.0 / 40.0,
    )
    return memory


# ── the captured regression ─────────────────────────────────────────────────

def test_both_fighters_grounded_in_captured_frame():
    """Overlay capture: both stood on the platform; only the opponent was detected as such."""
    memory = observe(
        player_center_y=0.6102 - 0.0722, player_h=0.0704,     # as reported, un-shifted
        opponent_center_y=0.5412, opponent_h=0.0620,
    )

    assert memory.opponent.grounded, "opponent was already correct"
    assert memory.player.grounded, (
        "agent stands on the platform in this frame; it previously read as airborne "
        "because its foot offset was a full height instead of half"
    )


def test_grounded_movement_target_is_reachable():
    """The same bias put grounded goals outside the success radius."""
    memory = observe(
        player_center_y=0.6102 - 0.0722, player_h=0.0704,
        opponent_center_y=0.5412, opponent_h=0.0620,
    )

    # curriculum_config samples grounded targets at platform y_min + 0.015
    target_y = memory.platform.y_min + 0.015
    error = abs(memory.player.y - target_y)

    assert error < 0.04, (
        f"grounded target off by {error:.4f}; the movement success threshold is 0.04, "
        "so a correctly grounded agent could never complete the goal"
    )


# ── the invariant that prevents a recurrence ────────────────────────────────

def test_both_fighters_use_the_same_foot_convention():
    """Identical boxes at the same height must produce identical foot positions."""
    memory = observe(
        player_center_y=0.50, player_h=0.09,
        opponent_center_y=0.50, opponent_h=0.09,
        player_x=0.40, opponent_x=0.60,
    )
    assert memory.player.y == pytest.approx(memory.opponent.y, abs=1e-6)


def test_foot_position_tracks_measured_box_height():
    """A taller box means a lower foot from the same centre; a constant cannot track that."""
    short = observe(player_center_y=0.50, player_h=0.06, opponent_center_y=0.50, opponent_h=0.06)
    tall = observe(player_center_y=0.50, player_h=0.14, opponent_center_y=0.50, opponent_h=0.14)

    assert tall.player.y > short.player.y
    assert tall.player.y - short.player.y == pytest.approx((0.14 - 0.06) / 2.0, abs=1e-6)


def test_relative_vertical_offset_is_foot_to_foot():
    """rel_dy must not carry a constant bias from mismatched conventions."""
    memory = observe(
        player_center_y=0.50, player_h=0.10,
        opponent_center_y=0.50, opponent_h=0.10,
        player_x=0.40, opponent_x=0.60,
    )
    memory.to_vector()
    assert memory.rel_dy == pytest.approx(0.0, abs=1e-6)


def test_airborne_fighter_is_not_grounded():
    """The fix must not simply mark everything grounded."""
    memory = observe(
        player_center_y=0.30, player_h=0.09,      # well above the platform
        opponent_center_y=0.5412, opponent_h=0.0620,
    )
    assert not memory.player.grounded
    assert memory.opponent.grounded
