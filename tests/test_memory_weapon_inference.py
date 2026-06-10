from __future__ import annotations

import pytest

from feature_extractor.memory.state_spec import StateSpec
from feature_extractor.memory.structured_memory import Memory


def _weapon(x: float, y: float) -> dict:
    return {
        "class_name": "weapons",
        "bbox": [x, y, 0.04, 0.04],
        "confidence": 0.9,
    }


def test_closest_visible_weapon_drives_weapon_features() -> None:
    memory = Memory()
    memory.player.x = 0.5
    memory.player.y = 0.5

    memory.update_from_detections([
        _weapon(0.8, 0.5),
        _weapon(0.53, 0.49),
    ])
    obs = memory.to_vector()

    assert memory.weapon.exists
    assert memory.weapon.x == pytest.approx(0.53)
    assert memory.weapon.y == pytest.approx(0.49)
    assert memory.closest_weapon_distance == pytest.approx(((0.03**2) + (0.01**2)) ** 0.5)
    assert StateSpec.get(obs, "weapon_dx") == pytest.approx(0.03)
    assert StateSpec.get(obs, "weapon_dy") == pytest.approx(-0.01)
    assert StateSpec.get(obs, "weapon_on_ground") == 1.0


def test_pickup_requires_num5_and_distance_under_threshold() -> None:
    memory = Memory()
    memory.player.x = 0.5
    memory.player.y = 0.5
    memory.update_from_detections([_weapon(0.54, 0.5)])

    memory.update_player_weapon_from_action(action_pick_throw=False, dist_to_weapon=0.04)
    assert memory.player.weapon_state == 0.0
    assert not memory.weapon_pickup_action_this_frame

    memory.update_player_weapon_from_action(action_pick_throw=True, dist_to_weapon=0.04)
    obs = memory.to_vector()

    assert memory.player.weapon_state == 1.0
    assert memory.weapon_pickup_action_this_frame
    assert StateSpec.get(obs, "player_has_weapon") == 1.0


def test_pickup_fails_when_closest_weapon_is_too_far() -> None:
    memory = Memory()
    memory.player.x = 0.5
    memory.player.y = 0.5
    memory.update_from_detections([_weapon(0.56, 0.5)])

    memory.update_player_weapon_from_action(action_pick_throw=True, dist_to_weapon=0.06)
    obs = memory.to_vector()

    assert memory.player.weapon_state == 0.0
    assert not memory.weapon_pickup_action_this_frame
    assert StateSpec.get(obs, "player_has_weapon") == 0.0


def test_num5_drops_when_player_already_has_weapon() -> None:
    memory = Memory()
    memory.player.weapon_state = 1.0

    memory.update_player_weapon_from_action(action_pick_throw=True, dist_to_weapon=0.0)

    assert memory.player.weapon_state == 0.0
    assert memory.weapon_drop_action_this_frame
    assert not memory.weapon_pickup_action_this_frame
