"""Reproduce the KO-flash damage rebound without GPU or game access."""
from __future__ import annotations

import numpy as np
import pytest

from feature_extractor.memory.structured_memory import Memory
from reward.match_objective import MatchObjective
from reward.ui_probe import PixelStocksHealthProvider


def frame(op=(255, 126, 0), agent=(255, 255, 255), stock=(120, 120, 120)):
    image = np.zeros((1, 3, 3), dtype=np.uint8)
    image[0] = [agent[::-1], op[::-1], stock[::-1]]
    return image


def provider():
    return PixelStocksHealthProvider({'agent': (0, 0), 'op': (1, 0), 'stock': (2, 0)})


@pytest.mark.parametrize('bad', [(42, 93, 188), (255, 254, 121)])
def test_invalid_colour_is_not_zero_damage(bad):
    probe = provider()
    probe(frame(), [])
    assert probe(frame(op=bad), [])[3] is None


@pytest.mark.parametrize('flash', [(42, 93, 188), (255, 255, 255)])
def test_flash_rebound_never_creates_damage_reward(flash):
    probe, memory, reward = provider(), Memory(), MatchObjective(match_bonus=False)
    _, _, own, opponent = probe(frame(), [])
    memory.update_stocks_and_health(self_health=own, op_health=opponent)
    for image in (frame(op=flash), frame()):
        _, _, own, opponent = probe(image, [])
        memory.update_stocks_and_health(self_health=own, op_health=opponent)
        result = reward.step(dict(damage_valid=True, op_delta_damage=memory.op_delta_damage))
        assert result['dmg_dealt'] == 0


def test_stable_respawn_zero_is_allowed_without_blue_credit():
    probe = provider()
    probe(frame(), [])
    readings = [probe(frame(op=(255, 255, 255)), [])[3] for _ in range(3)]
    assert readings[:2] == [None, None]
    assert readings[2] == 351
    assert probe.op_stocks_left == 3  # No blue credited-KO event was observed.


def test_blue_credit_and_red_loss_are_debounced_without_damage_requirement(monkeypatch):
    now = [10.0]
    monkeypatch.setattr('reward.ui_probe.time.perf_counter', lambda: now[0])
    for colour, expected in [((0, 255, 255), (3, 2)), ((255, 0, 0), (2, 3))]:
        probe = provider()
        for _ in range(8):
            result = probe(frame(stock=colour), [])
            now[0] += .03
        assert result[:2] == expected
        assert result[2:] == (None, None), 'HUD animation is not a damage sample'


def test_damage_colour_curve_is_unchanged():
    from reward.rgb_to_dmg import get_dmg
    for rgb, damage in [((255, 255, 255), 0), ((255, 255, 0), 50),
                        ((255, 153, 0), 100), ((255, 0, 0), 150),
                        ((191, 0, 0), 200), ((140, 0, 0), 250), ((74, 0, 0), 300)]:
        assert get_dmg(np.array(rgb)) == pytest.approx(damage)


def test_valid_colour_flash_is_not_a_hit_and_real_hit_confirms_once():
    probe = provider()
    baseline = probe(frame(), [])[3]
    assert probe(frame(op=(255, 0, 0)), [])[3] is None
    assert probe(frame(), [])[3] == baseline
    readings = [probe(frame(op=(255, 0, 0)), [])[3] for _ in range(3)]
    assert readings == [None, None, 201.0]


def test_invalid_samples_break_confirmation_and_reset_can_clear_baselines():
    probe = provider()
    probe(frame(), [])
    assert probe(frame(op=(255, 255, 255)), [])[3] is None
    assert probe(frame(op=(40, 50, 60)), [])[3] is None
    assert probe(frame(op=(255, 255, 255)), [])[3] is None
    probe.reset(preserve_match_state=True)
    assert probe(frame(), [])[3] < 351
    probe.reset(preserve_match_state=False)
    assert probe(frame(op=(255, 255, 255)), [])[3] == 351


def test_unavailable_pixels_abstain():
    probe = PixelStocksHealthProvider({'agent': (-1, 0), 'op': (999, 0), 'stock': (999, 0)})
    assert probe(frame(), [])[2:] == (None, None)
    assert probe(None, [])[2:] == (None, None)
    assert PixelStocksHealthProvider({})(frame(), [])[2:] == (None, None)


@pytest.mark.parametrize('rgb', [[1, 2], [float('nan'), 0, 0], [-1, 0, 0], [256, 0, 0]])
def test_malformed_rgb_abstains(rgb):
    from reward.rgb_to_dmg import decode_damage
    assert decode_damage(rgb) is None


def test_first_valid_read_after_unreadable_start_is_a_baseline_not_a_hit():
    from test_observation_window import make_env
    env, _ = make_env()
    env.stocks_health_provider = provider()
    try:
        env._last_frame = frame(op=(42, 93, 188))
        env._update_game_logic([], action_jump=False, action_dodge=False)
        env._last_frame = frame()
        env._update_game_logic([], action_jump=False, action_dodge=False)
        assert env.memory.op_health < 351
        assert env.memory.op_delta_damage == 0
    finally:
        env.close()
