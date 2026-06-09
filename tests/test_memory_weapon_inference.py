from __future__ import annotations

from feature_extractor.memory.state_spec import StateSpec
from feature_extractor.memory.structured_memory import Memory


def _weapon(x: float, y: float) -> dict:
    return {
        "class_name": "weapons",
        "bbox": [x, y, 0.04, 0.04],
        "confidence": 0.9,
    }


def test_visual_weapon_disappearance_near_player_sets_player_has_weapon() -> None:
    memory = Memory()
    memory.player.x = 0.5
    memory.player.y = 0.5

    memory.update_from_detections([_weapon(0.53, 0.5)])
    assert memory.player.weapon_state == 0.0
    assert memory.weapon_visible_this_frame

    memory.update_from_detections([])
    assert memory.player.weapon_state == 0.0

    memory.update_from_detections([])
    obs = memory.to_vector()

    assert memory.weapon_pickup_inferred_this_frame
    assert memory.player.weapon_state == 1.0
    assert StateSpec.get(obs, "player_has_weapon") == 1.0
    assert StateSpec.get(obs, "weapon_on_ground") == 0.0


def test_visual_pickup_infers_candidate_disappearance_even_if_other_weapon_remains() -> None:
    memory = Memory()
    memory.player.x = 0.5
    memory.player.y = 0.5

    memory.update_from_detections([_weapon(0.53, 0.5), _weapon(0.82, 0.5)])
    memory.update_from_detections([_weapon(0.82, 0.5)])
    memory.update_from_detections([_weapon(0.82, 0.5)])
    obs = memory.to_vector()

    assert memory.player.weapon_state == 1.0
    assert StateSpec.get(obs, "player_has_weapon") == 1.0
    assert StateSpec.get(obs, "weapon_on_ground") == 1.0


def test_visual_weapon_disappearance_far_from_player_does_not_set_has_weapon() -> None:
    memory = Memory()
    memory.player.x = 0.5
    memory.player.y = 0.5

    memory.update_from_detections([_weapon(0.53, 0.5)])
    memory.player.x = 0.1
    memory.player.y = 0.5

    for _ in range(3):
        memory.update_from_detections([])

    obs = memory.to_vector()
    assert memory.player.weapon_state == 0.0
    assert StateSpec.get(obs, "player_has_weapon") == 0.0
