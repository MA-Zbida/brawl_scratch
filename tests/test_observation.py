"""brawl.sense.perception and brawl.observation: a frame becomes a Sense, a Sense becomes one vector."""
from __future__ import annotations

import numpy as np
import pytest

from brawl import actions, observation, stage
from brawl.observation import FEATURES, Observation
from brawl.sense.perception import Perception
from brawl.sense.truth import Box
from test_hud import draw


def value(obs, name):
    return obs[FEATURES.index(name)]


class Detector:
    """Returns the boxes it is given, in the frame's own pixel coordinates."""

    def __init__(self):
        self.boxes: list[Box] = []

    def __call__(self, image):
        return self.boxes


def box(cls, x, y, score=0.9):
    return Box(cls, x - 30, y - 40, x + 30, y + 40, score, point=(x, y))


def frame(mordex=(255, 255, 255), wu=(255, 255, 255), badge=None, scale=1.0):
    return draw(("mordex", mordex, badge), ("wu_shang", wu, None), scale)


def test_a_half_size_frame_gives_positions_on_the_full_screen_and_damage_from_the_hud():
    detector = Detector()
    sense = Perception(detector)
    detector.boxes = [box("wu_shang_spear", 450, 291), box("mordex_unarmed", 600, 291)]
    s = sense(frame(scale=0.5), 0)
    assert s.fighters["wu_shang"].box.position == (900, 582) and s.readable
    s = sense(frame(mordex=(255, 255, 0), scale=0.5), 3)
    assert s.events.damage_dealt == pytest.approx(50, abs=1) and s.damage["mordex"] == pytest.approx(50, abs=1)


def test_the_vector_names_every_entry_and_places_the_fighters():
    detector = Detector()
    sense, obs = Perception(detector), Observation()
    detector.boxes = [box("wu_shang_spear", 900, 582), box("mordex_gauntlets", 500, 700),
                      Box("weapon_spawn", 1000, 540, 1040, 620, 0.9)]
    sense(frame(), 0)
    v = obs.build(sense(frame(), 3))  # the second frame confirms the spawn marker
    assert v.shape == (observation.SIZE,) == (len(FEATURES),) and v.dtype == np.float32
    assert value(v, "agent_grounded") == 1 and value(v, "agent_weapon_1") == 1
    assert value(v, "agent_x") == pytest.approx(-55 / 960)
    assert value(v, "opponent_offstage") == 1 and value(v, "opponent_below") == 1 and value(v, "opponent_weapon_2") == 1
    assert value(v, "opponent_dx") == pytest.approx(-400 / 960) and value(v, "spawn_present") == 1
    assert value(v, "spawn_dx") == pytest.approx(120 / 960) and value(v, "thrown_present") == 0


def test_a_ko_removes_the_fighter_but_keeps_where_it_was_and_actions_are_remembered():
    detector = Detector()
    sense, obs = Perception(detector), Observation()
    detector.boxes = [box("wu_shang_unarmed", 900, 582), box("mordex_unarmed", 300, 582)]
    obs.build(sense(frame(mordex=(255, 153, 0)), 300))
    detector.boxes = [box("wu_shang_unarmed", 900, 582)]
    s = sense(frame(), 303)  # Mordex's ring is white again: a lost stock
    assert s.events.ko_scored == 1 and "mordex" not in s.fighters
    obs.acted((2, 3))
    v = obs.build(s)
    assert value(v, "opponent_present") == 0
    assert value(v, "opponent_x") == pytest.approx((300 - stage.CENTRE[0]) / 960)
    assert value(v, "action_4_dir_right") == 1 and value(v, "action_4_btn_heavy") == 1 and value(v, "action_3_dir_none") == 1


def test_timers_count_decisions_since_presses_ground_hits_and_respawns():
    detector = Detector()
    sense, obs = Perception(detector), Observation()
    jump, dodge = actions.BUTTON_NAMES.index("jump"), actions.BUTTON_NAMES.index("dodge")
    air, ground = 400, 582
    script = [  # Wu Shang's y, Mordex's ring, the action chosen after the observation
        (ground, (255, 255, 255), (0, jump)), (air, (255, 255, 255), (0, jump)), (air, (255, 255, 255), (0, 0)),
        (air, (255, 255, 255), (0, jump)), (air, (255, 255, 0), (0, dodge)), (ground, (255, 255, 0), (0, 0))]
    rows = []
    for i, (y, ring, action) in enumerate(script):
        detector.boxes = [box("wu_shang_unarmed", 900, y), box("mordex_unarmed", 1100, 582)]
        rows.append(obs.build(sense(frame(mordex=ring), 3 * i)))
        obs.acted(action)
    clip = observation.CLIP
    assert [round(value(v, "since_press_jump") * clip) for v in rows] == [clip, 1, 2, 3, 1, 2]  # held is not pressed again
    assert [round(value(v, "agent_jumps_since_ground") * 4) for v in rows] == [0, 1, 1, 1, 2, 0]
    assert [round(value(v, "agent_since_ground") * clip) for v in rows] == [0, 1, 2, 3, 4, 0]
    assert [round(value(v, "opponent_since_hit") * clip) for v in rows] == [clip, clip, clip, clip, 1, 2]
    assert round(value(rows[5], "since_press_dodge") * clip) == 1 and value(rows[5], "since_press_heavy") == 1


def test_stored_decisions_get_the_same_timers_as_the_builder_gives():
    rng = np.random.default_rng(0)
    n, builder = 300, Observation()
    grounded, present = rng.random((n, 2)) < 0.4, rng.random((n, 2)) < 0.95
    vy, taken, dealt = rng.normal(0, 0.4, n), rng.random(n) < 0.1, rng.random(n) < 0.1
    action = np.stack([rng.integers(9, size=n), rng.integers(6, size=n)], 1)
    base = np.zeros((n, len(observation.BASE)), np.float32)
    for name, values in (("agent_grounded", grounded[:, 0]), ("opponent_grounded", grounded[:, 1]),
                         ("agent_present", present[:, 0]), ("opponent_present", present[:, 1]), ("opponent_vy", vy)):
        base[:, observation.BASE.index(name)] = values
    live = []
    for i in range(n):  # what build() does with the same measurements
        hit = (taken[i - 1], dealt[i - 1]) if i else (False, False)
        live.append(builder.timers.update(grounded[i], present[i], base[i, observation.BASE.index("opponent_vy")], hit))
        builder.acted(action[i])
    stored = observation.add_timers(base, action, taken, dealt)
    assert stored.shape == (n, observation.SIZE) and np.allclose(stored[:, len(observation.BASE):], live)


def test_a_crop_is_the_same_window_from_a_half_size_or_full_size_frame():
    import cv2

    from brawl import vision

    full = np.zeros((1080, 1920, 3), np.uint8)
    full[500:540, 1000:1020] = 255           # a bright block 30 px right of the point
    half = cv2.resize(full, (960, 540), interpolation=cv2.INTER_AREA)
    a, b = vision.crop(full, (980, 520)), vision.crop(half, (980, 520))
    assert a.shape == b.shape == (96, 96) and np.abs(a.astype(int) - b.astype(int)).max() <= 40
    cols = np.flatnonzero(a.max(0) > 128)
    assert 48 + 7 <= cols.min() <= 48 + 9    # 20 screen px right of the centre, at 96 / 256 scale
    edge = vision.crop(full, (10, 10))
    assert edge[:40, :40].max() == 0         # beyond the screen: black
