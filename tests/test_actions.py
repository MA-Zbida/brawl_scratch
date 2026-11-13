"""brawl.actions and brawl.stage: actions as keys and back, and where a point is on the stage."""
from __future__ import annotations

from brawl import actions, stage
from brawl.game.keys import DODGE, DOWN, HEAVY, JUMP, LEFT, LIGHT, RIGHT, UP, to_mask


def test_an_action_holds_its_direction_and_its_button_for_the_decision():
    action = (actions.DIRECTION_NAMES.index("down-left"), actions.BUTTON_NAMES.index("heavy"))
    assert actions.name(action) == "down-left+heavy"
    assert actions.to_plan(action) == [{DOWN, LEFT, HEAVY}] * 3
    assert len(actions.to_plan(action, frames=4)) == 4
    assert actions.to_plan((0, 0)) == [frozenset()] * 3
    assert actions.SHAPE == (9, 6)


def test_every_action_has_its_own_keys_and_a_mirror():
    plans = {tuple(map(frozenset, actions.to_plan((d, b)))) for d in range(9) for b in range(6)}
    assert len(plans) == 54
    assert actions.name(actions.mirror((actions.DIRECTION_NAMES.index("up-left"), 2))) == "up-right+light"
    assert all(actions.mirror(actions.mirror((d, 1))) == (d, 1) for d in range(9))


def test_recorded_keys_are_read_back_as_the_action_that_makes_them():
    for d in range(9):
        for b in range(6):
            plan = actions.to_plan((d, b))
            held = [to_mask(k) for k in plan]
            pressed = [to_mask({actions.BUTTONS[b]}) if b and i == 0 else 0 for i in range(3)]
            assert actions.from_demo(held, pressed) == ((d, b), 0)


def test_a_human_decision_keeps_the_main_direction_and_the_first_button():
    held = [to_mask({RIGHT}), to_mask({RIGHT, UP}), to_mask({RIGHT, UP, JUMP})]
    pressed = [0, to_mask({LIGHT}), to_mask({JUMP})]
    action, lost = actions.from_demo(held, pressed)
    assert actions.name(action) == "up-right+light" and lost == 1  # the jump does not fit
    assert actions.direction_of(frozenset({LEFT, RIGHT, UP})) == actions.DIRECTION_NAMES.index("up")
    assert actions.from_demo([to_mask({DODGE})] * 3, [0, 0, 0]) == ((0, 0), 0)  # held from long ago: no button


def test_a_charged_heavy_is_the_same_button_chosen_again():
    heavy = actions.BUTTON_NAMES.index("heavy")
    first = actions.from_demo([0, to_mask({HEAVY}), to_mask({HEAVY})], [0, to_mask({HEAVY}), 0])
    assert first == ((0, heavy), 0)
    assert actions.from_demo([to_mask({HEAVY})] * 3, [0, 0, 0], previous=heavy) == ((0, heavy), 0)  # still charging
    assert actions.from_demo([to_mask({HEAVY}), 0, 0], [0, 0, 0], previous=heavy) == ((0, 0), 0)    # released
    again = actions.from_demo([to_mask({HEAVY})] * 3, [to_mask({HEAVY}), 0, 0], previous=heavy)
    assert again == ((0, heavy), 1)  # pressed again at once: the agent's repeat cannot do that


def test_the_stage_tells_ground_ledges_and_recovery():
    standing = stage.place(900, 582)
    assert standing.grounded and not standing.offstage and standing.to_left_ledge == 295
    jumping = stage.place(900, 450)
    assert not jumping.grounded and jumping.height == 132
    off = stage.place(500, 700)
    assert off.offstage and off.below and off.to_left_ledge == -105 and not off.grounded


def test_the_scripted_player_shows_every_button_stays_on_the_stage_and_recovers():
    from types import SimpleNamespace

    from brawl.scripted import Scripted
    from brawl.sense.truth import Box

    def sense(x, y, cls="mordex_unarmed", spawns=()):
        me = SimpleNamespace(box=Box(cls, x - 30, y - 40, x + 30, y + 40, 0.9, point=(x, y)))
        return SimpleNamespace(fighters={"mordex": me},
                               weapons=[Box("weapon_spawn", s - 15, 540, s + 15, 620, 0.9) for s in spawns])

    script, seen = Scripted("mordex"), set()
    for _ in range(3000):
        seen.add(script(sense(950, 582, "mordex_scythe")))
    assert {b for _, b in seen} == set(range(6)) and len({d for d, _ in seen}) == 9
    quiet = [script(sense(950, 582, "mordex_scythe")) for _ in range(1000)]
    assert sum(a == (0, 0) for a in quiet) > 300  # pauses between moves
    right, left = actions.DIRECTION_NAMES.index("right"), actions.DIRECTION_NAMES.index("left")
    fresh = Scripted("mordex")
    assert fresh(sense(stage.LEFT + 20, 582)) == (right, 0) and fresh(sense(stage.RIGHT - 20, 582)) == (left, 0)
    off = [fresh(sense(stage.LEFT - 150, 700)) for _ in range(6)]
    assert off[0] == (actions.DIRECTION_NAMES.index("up-right"), 1) and off[1] == (right, 0) and off[5][1] == 1
    assert Scripted("mordex")(SimpleNamespace(fighters={}, weapons=[])) == (0, 0)  # not on the screen: nothing
    fetch = Scripted("mordex", seed=1)
    moves = [fetch(sense(900, 582, spawns=(1000,))) for _ in range(400)]
    assert (right, 0) in moves and any(fetch(sense(1000, 582, spawns=(1000,))) == (0, 5) for _ in range(400))
