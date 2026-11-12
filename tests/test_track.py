"""brawl.sense.track: the fighters' state across frames, from one frame's boxes at a time."""
from __future__ import annotations

import copy

import pytest

from brawl.sense.track import CARRY, MAJORITY, Tracker
from brawl.sense.truth import Box


def box(cls: str, x: float, y: float = 500.0, score: float = 0.9) -> Box:
    return Box(cls, x - 30, y - 40, x + 30, y + 40, score, point=(x, y))


def test_a_hidden_fighter_is_carried_at_its_velocity_then_absent():
    t = Tracker()
    t.update([box("mordex_unarmed", 500)])
    seen = t.update([box("mordex_unarmed", 510)])["mordex"]
    assert seen.seen and seen.velocity == pytest.approx((10, 0))
    for age in range(1, CARRY + 1):  # behind a hit flash: no box
        carried = t.update([])["mordex"]
        assert not carried.seen and carried.age == age and carried.box.position == pytest.approx((510 + 10 * age, 500))
    assert "mordex" not in t.update([])
    assert t.update([box("mordex_scythe", 900)])["mordex"].box.cls == "mordex_scythe"  # back, as it is seen


def test_a_doubtful_box_counts_only_where_the_fighter_should_be():
    t = Tracker()
    assert t.update([box("mordex_unarmed", 500, score=0.1)]) == {}  # nothing known yet: not believed
    t.update([box("mordex_unarmed", 500)])
    near = t.update([box("mordex_unarmed", 520, score=0.1)])["mordex"]
    assert near.seen and near.box.position == (520, 500)
    far = t.update([box("mordex_unarmed", 900, score=0.1)])["mordex"]
    assert not far.seen and far.box.position == pytest.approx((540, 500))  # carried instead


def test_a_sure_box_too_far_away_is_believed_only_when_the_next_frame_confirms_it():
    t = Tracker()
    t.update([box("wu_shang_spear", 500)])
    once = t.update([box("wu_shang_spear", 1300)])["wu_shang"]  # one wrong box
    assert not once.seen and once.box.position == pytest.approx((500, 500))
    assert t.update([box("wu_shang_spear", 503)])["wu_shang"].seen
    t.update([box("wu_shang_spear", 1300)])
    twice = t.update([box("wu_shang_spear", 1305)])["wu_shang"]  # it really is there
    assert twice.seen and twice.box.position == (1305, 500) and twice.velocity == (0.0, 0.0)


def test_the_held_weapon_changes_by_majority_not_by_one_frame():
    t = Tracker()
    for _ in range(7):
        t.update([box("mordex_gauntlets", 500)])
    assert t.update([box("mordex_unarmed", 500)])["mordex"].box.cls == "mordex_gauntlets"  # one stray frame
    assert t.update([box("mordex_gauntlets", 500)])["mordex"].box.cls == "mordex_gauntlets"
    read = [t.update([box("mordex_scythe", 500)])["mordex"].box.cls for _ in range(MAJORITY)]  # a real pickup
    assert read == ["mordex_gauntlets"] * (MAJORITY - 1) + ["mordex_scythe"]


def test_a_lost_stock_ends_the_carrying_and_the_state_survives_a_rewind():
    t = Tracker()
    t.update([box("mordex_scythe", 500), box("wu_shang_spear", 800)])
    saved = copy.deepcopy(t)
    after = t.update([], lost={"mordex"})
    assert "mordex" not in after and not after["wu_shang"].seen
    assert t.update([box("mordex_unarmed", 960, 200)])["mordex"].box.cls == "mordex_unarmed"  # respawned
    again = saved.update([box("mordex_scythe", 505), box("wu_shang_spear", 800)])  # the rewound frame, replayed
    assert again["mordex"].velocity == pytest.approx((5, 0)) and again["mordex"].box.cls == "mordex_scythe"


def test_a_fighter_carried_out_of_the_frame_is_absent_and_step_scales_the_distances():
    t = Tracker()
    t.update([box("wu_shang_unarmed", 1880)])
    t.update([box("wu_shang_unarmed", 1910)])
    assert t.update([]) == {}  # 1940 is beyond the right edge
    slow = Tracker(step=3)
    slow.update([box("mordex_unarmed", 500)])
    assert slow.update([box("mordex_unarmed", 800)])["mordex"].seen  # 300 px in three game frames
    assert slow.carry == 3


def test_a_spawn_marker_is_reported_once_confirmed_and_not_after_it_is_gone():
    def spawn(x, score=0.9):
        return Box("weapon_spawn", x - 15, 540, x + 15, 600, score)

    t = Tracker()
    t.update([spawn(700), spawn(1200, 0.3)])
    assert t.weapons == []  # seen once: could be smoke
    t.update([spawn(701), spawn(702, 0.5), Box("weapon_thrown", 900, 300, 960, 320, 0.6)])
    assert [b.cls for b in t.weapons] == ["weapon_spawn", "weapon_thrown"]  # one box per marker; 1200 was noise
    assert t.weapons[0].centre[0] == 701
    t.update([])
    assert t.weapons == []  # picked up: gone at once
    t.update([spawn(701)])
    assert len(t.weapons) == 1  # one missed frame does not undo what was seen before it
    t.update([Box("weapon_spawn", 0, 0, 30, 60, 0.2)])
    t.update([Box("weapon_spawn", 0, 0, 30, 60, 0.2)])
    assert [b.centre[0] for b in t.weapons] == []  # below the weapon score
