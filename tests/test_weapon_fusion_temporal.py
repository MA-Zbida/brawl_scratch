"""Action intent is immediate; visual corrections need independent evidence."""
from __future__ import annotations

import pytest

from brawl.perception.weapon_classifier import WeaponPrediction, WeaponStateFusion
from brawl.perception.weapon_evidence import WeaponHoldGate


def visual(armed, **kwargs):
    return WeaponPrediction(armed, .99, **kwargs)


def evidence(armed):
    return dict(player_weapon_visual_fresh=1., player_weapon_visual_valid=1.,
                player_weapon_visual_confidence=.99, player_weapon_visual_armed=float(armed))


def test_one_visual_positive_cannot_invent_possession():
    fusion = WeaponStateFusion()
    assert fusion.resolve(False, visual(True)) == (False, False)
    assert fusion.resolve(False, visual(False)) == (False, False)


def test_repeated_visual_evidence_can_discover_and_lose_weapon():
    fusion = WeaponStateFusion()
    fusion.resolve(False, visual(True))
    assert fusion.resolve(False, visual(True)) == (True, False)
    assert fusion.resolve(True, visual(False)) == (True, False)
    assert fusion.resolve(True, visual(False, fresh=False)) == (True, False)
    assert fusion.resolve(True, visual(False)) == (True, False)
    assert fusion.resolve(True, visual(False)) == (False, True)
    assert fusion.disarm_events == 1


def test_pickup_intent_is_provisional_and_bounded_without_visual_proof():
    fusion = WeaponStateFusion()
    fusion.observe_action(pickup=True, dropped=False)
    assert fusion.resolve(True, visual(False)) == (True, False)
    armed = True
    for _ in range(20):
        armed, disarm = fusion.resolve(armed, None)
        assert not disarm  # An unconfirmed pickup is not a confirmed disarm.
    assert not armed


def test_pickup_confirmation_survives_isolated_negative():
    fusion = WeaponStateFusion()
    fusion.observe_action(pickup=True, dropped=False)
    fusion.resolve(True, visual(True))
    assert not fusion.confirmed
    fusion.resolve(True, visual(True))
    assert fusion.confirmed
    assert fusion.resolve(True, visual(False)) == (True, False)
    assert fusion.resolve(True, visual(True)) == (True, False)


def test_throw_has_priority_over_lagging_positive_and_can_be_corrected_later():
    fusion = WeaponStateFusion()
    fusion.observe_action(pickup=False, dropped=True)
    for _ in range(4):
        assert fusion.resolve(False, visual(True)) == (False, False)
    for _ in range(20):
        armed, _ = fusion.resolve(False, visual(True))
    assert armed  # Do not permanently lock state if the game ignored a throw.


def test_unknown_gap_expires_contradiction_votes():
    fusion = WeaponStateFusion()
    fusion.resolve(True, visual(False))
    fusion.resolve(True, visual(False))
    for _ in range(12):
        fusion.resolve(True, None)
    assert fusion.resolve(True, visual(False)) == (True, False)


@pytest.mark.parametrize('confidence', [float('nan'), float('inf'), 1.1])
def test_invalid_confidence_cannot_change_state(confidence):
    fusion = WeaponStateFusion()
    for _ in range(4):
        assert fusion.resolve(True, WeaponPrediction(False, confidence)) == (True, False)


def test_single_positive_followed_by_negatives_is_not_a_dropped_pickup():
    gate = WeaponHoldGate()
    gate.update(True, evidence(True))
    for _ in range(8):
        gate.update(False, evidence(False))
    assert not gate.pickup_seen
    assert not gate.dropped
    assert not gate.success


def test_gate_uses_fusion_confirmation_and_keeps_real_loss_rejection():
    gate = WeaponHoldGate(required_steps=40)
    gate.update(True, evidence(True) | {'player_weapon_possession_confirmed': 0.})
    assert not gate.pickup_seen
    gate.update(True, evidence(True) | {'player_weapon_possession_confirmed': 1.})
    assert gate.pickup_seen
    for _ in range(4):
        gate.update(False, evidence(False))
    assert gate.dropped


def test_confirmed_hold_stops_immediately_on_executed_throw():
    gate = WeaponHoldGate()
    gate.update(True, evidence(True) | {'player_weapon_possession_confirmed': 1.})
    gate.update(False, evidence(True) | {'weapon_drop_action': 1.})
    assert gate.dropped
    assert not gate.success


def test_respawn_clears_fusion_proof_and_pending_input():
    from types import SimpleNamespace
    from test_observation_window import make_env
    env, _ = make_env(action_repeat_steps=1)
    env.weapon_classifier = SimpleNamespace(update=lambda frame, boxes: {'player': visual(True)})
    try:
        env._player_weapon_fusion.observe_action(pickup=True, dropped=False)
        env.memory.player.weapon_state = 1.
        env.memory.player_respawn_timer = 1.
        env._update_weapon_classification(None)
        assert env.memory.player.weapon_state == 0.
        assert not env._player_weapon_fusion.pickup_pending
        assert not env._player_weapon_fusion.confirmed
    finally:
        env.close()


def test_reset_clears_votes_and_action_guards():
    fusion = WeaponStateFusion()
    fusion.resolve(False, visual(True))
    fusion.observe_action(pickup=True, dropped=False)
    fusion.reset()
    assert not fusion.pickup_pending
    assert fusion.resolve(False, visual(True)) == (False, False)


def test_collector_calls_an_isolated_positive_unconfirmed_not_dropped():
    from test_semantic_pipeline import ScriptEnv, observation
    from brawl.state.state_spec import StateSpec
    from brawl.pipelines.semantic57.collect import collect_attempt

    class BlipEnv(ScriptEnv):
        def step(self, action):
            obs, reward, term, trunc, info = super().step(action)
            obs[StateSpec.index('player_has_weapon')] = float(self.i == 3)
            info.update(evidence(self.i == 3))
            if int(action) == 47:
                info['input_taps_pressed'] = ['num5']
            return obs, reward, term, trunc, info

    states = [observation() for _ in range(20)]
    for obs in states:
        obs[StateSpec.index('weapon_on_ground')] = 1.
        obs[StateSpec.index('weapon_dx')] = .01
    env = BlipEnv(states)
    result = collect_attempt(env, 'weapon', max_steps=12)
    assert result.pickup_requests > 0
    assert not result.accepted
    assert result.reason == 'pickup_unconfirmed'
    assert result.peak_hold_steps == 0
    assert env.released


def test_env_pickup_grace_is_wired_after_input_not_before_capture():
    from types import SimpleNamespace
    from brawl.control.action_space import Action
    from brawl.state.state_spec import StateSpec
    from test_observation_window import make_env, det
    env, extractor = make_env(action_repeat_steps=1)
    env.weapon_classifier = SimpleNamespace(update=lambda frame, boxes: {'player': visual(False)})
    extractor.detections = [det('character', .3, .6), det('character', .7, .6),
                           det('indicator_self', .3, .53, .02, .02),
                           det('weapon', .3, .64, .04, .04)]
    try:
        env.reset()
        env.step(Action.PICKUP)
        obs, _, _, _, info = env.step(Action.NOOP)
        assert StateSpec.get(obs, 'player_has_weapon') == 1.
        assert info['player_weapon_pickup_pending'] == 1.
        assert info['player_weapon_possession_confirmed'] == 0.
        for _ in range(20):
            obs, _, _, _, info = env.step(Action.NOOP)
        assert StateSpec.get(obs, 'player_has_weapon') == 0.
        assert info['player_weapon_visual_disarm_event'] == 0.
    finally:
        env.close()
