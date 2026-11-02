from __future__ import annotations

import json

import gymnasium as gym
import numpy as np
import pytest

from brawl.state.state_spec import StateSpec
from brawl.pipelines.semantic57.contract import ACTION_DIM, OBSERVATION_DIM
from brawl.pipelines.semantic57.demos import DemoBatch
from brawl.pipelines.semantic57.collect import collect_attempt
from brawl.pipelines.semantic57.bc import pretrain, agreement
from brawl.pipelines.semantic57.policy import build_model, load_model, save_model
from brawl.pipelines.semantic57.run import main


def test_default_classifier_matches_latest_validated_live_model():
    from brawl.pipelines.semantic57.run import parser
    assert parser().parse_args([]).weapon_classifier_checkpoint.name == 'weapon_classifier_r18_live_v1.pt'


def observation(offstage=0, grounded=1):
    obs = np.zeros(OBSERVATION_DIM, np.float32)
    obs[StateSpec.index('player_is_offstage')] = offstage
    obs[StateSpec.index('player_grounded')] = grounded
    obs[StateSpec.index('signed_dx_to_ledge')] = .1
    obs[-ACTION_DIM:] = 1
    return obs


class ScriptEnv(gym.Env):
    def __init__(self, observations):
        self.observations = observations
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBSERVATION_DIM,), np.float32)
        self.action_space = gym.spaces.Discrete(ACTION_DIM)
        self.i = 0
        self.released = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.i = 0
        return self.observations[0].copy(), {}

    def step(self, action):
        self.i = min(self.i + 1, len(self.observations) - 1)
        return self.observations[self.i].copy(), 0., False, False, {}

    def _release_all_inputs(self):
        self.released = True


def test_weapon_cannot_succeed_without_a_pickup_press():
    class PhantomWeapon(ScriptEnv):
        def step(self, action):
            obs, reward, term, trunc, info = super().step(action)
            armed = self.i >= 2
            obs[StateSpec.index('player_has_weapon')] = armed
            info.update(player_weapon_visual_fresh=1, player_weapon_visual_valid=1,
                        player_weapon_visual_confidence=.99, player_weapon_visual_armed=int(armed))
            return obs, reward, term, trunc, info
    result = collect_attempt(PhantomWeapon([observation()]*10), 'weapon', max_steps=8,
                             setup_steps=3, hold_steps=2, min_steps=2)
    assert not result.accepted
    assert result.reason != 'weapon_dropped_after_pickup'


def test_pickup_step_capture_cannot_confirm_the_new_input():
    class PreInputPositive(ScriptEnv):
        def step(self, action):
            obs, reward, term, trunc, info = super().step(action)
            armed = self.i == 2
            obs[StateSpec.index('player_has_weapon')] = armed
            info.update(player_weapon_visual_fresh=1, player_weapon_visual_valid=1,
                        player_weapon_visual_confidence=.99, player_weapon_visual_armed=int(armed))
            if int(action) == 47:
                info['input_taps_pressed'] = ['num5']
            return obs, reward, term, trunc, info
    states = [observation() for _ in range(12)]
    for obs in states:
        obs[StateSpec.index('weapon_on_ground')] = 1
        obs[StateSpec.index('weapon_dx')] = .01
    result = collect_attempt(PreInputPositive(states), 'weapon', max_steps=8, hold_steps=1)
    assert result.pickup_requests >= 1
    assert not result.accepted
    assert result.peak_hold_steps == 0
    assert result.reason == 'pickup_unconfirmed'


def test_cancelled_pending_label_is_removed_without_discarding_good_combat(monkeypatch):
    class InterruptedInput(ScriptEnv):
        def step(self, action):
            obs, reward, term, trunc, info = super().step(action)
            if self.i == 2:
                obs[-ACTION_DIM:] = 0
                obs[-ACTION_DIM] = 1
                info.update(semantic_execution_status='started', semantic_active_action='GROUND_LIGHT_TOWARD')
            elif self.i == 3:
                info.update(semantic_execution_status='cancelled',
                            semantic_execution_reason='context_changed_before_min_hold',
                            semantic_active_action='GROUND_LIGHT_TOWARD')
            return obs, reward, term, trunc, info
    states = [observation() for _ in range(10)]
    for i, obs in enumerate(states):
        obs[StateSpec.index('player_x')] = i / 10
        obs[StateSpec.index('opponent_exists')] = 1
        obs[StateSpec.index('rel_dx')] = .05
        obs[StateSpec.index('rel_distance')] = .1
    events = []
    result = collect_attempt(InterruptedInput(states), 'combat', max_steps=6, on_event=events.append)
    assert result.accepted
    assert result.context_interruptions == 1
    assert len(result.observations) == 4
    assert all(StateSpec.get(obs, 'player_x') > .2 for obs in result.observations)
    assert any(event['event'] == 'execution_interrupted' for event in events)


def test_recovery_discards_setup_and_requires_landing():
    env = ScriptEnv([observation(), observation(), observation(1, 0),
                     observation(1, 0), observation(), observation()])
    result = collect_attempt(env, 'recovery', setup_steps=5, max_steps=8, min_steps=2)
    assert result.accepted
    assert len(result.observations) == 3
    assert StateSpec.get(result.observations[0], 'player_is_offstage') == 1
    assert env.released


def test_never_offstage_is_rejected_without_bc_rows():
    result = collect_attempt(ScriptEnv([observation()]), 'recovery', setup_steps=3, max_steps=4)
    assert not result.accepted
    assert result.reason == 'setup_timeout'
    assert not result.observations


def test_bc_then_ppo_then_reload_cpu(tmp_path):
    env = ScriptEnv([observation()])
    model = build_model(env, n_steps=8, batch_size=4, n_epochs=1)
    data = DemoBatch(np.stack([observation()] * 40), np.ones(40, np.int64),
                     np.repeat(np.arange(4), 10), np.array(['movement'] * 40))
    before = agreement(model, data, np.arange(40))['all']['nll']
    rows = pretrain(model, data, epochs=3, batch_size=8, learning_rate=.005, output=tmp_path)
    assert rows[-1]['validation_nll'] < before
    model.learn(16)
    save_model(model, tmp_path / 'latest.zip')
    assert str(load_model(tmp_path / 'latest.zip').device) == 'cpu'
    assert (tmp_path / 'bc_metrics.csv').is_file()
    split = json.loads((tmp_path / 'bc_split.json').read_text())
    assert not set(split['train_episodes']) & set(split['validation_episodes'])


def test_live_requires_explicit_opt_in_before_environment_creation(tmp_path):
    with pytest.raises(SystemExit, match='experimental-controls'):
        main(['--stage', 'all', '--run-dir', str(tmp_path / 'not-created')])
    assert not (tmp_path / 'not-created').exists()


def test_smoke_exercises_full_artifact_path_without_hardware(tmp_path):
    path = tmp_path / 'smoke'
    main(['--stage', 'smoke', '--run-dir', str(path)])
    assert (path / 'bc_init.zip.metadata.json').is_file()
    assert (path / 'latest.zip.metadata.json').is_file()
    assert json.loads((path / 'status.json').read_text())['status'] == 'smoke_pass'
    with pytest.raises(FileExistsError):
        main(['--stage', 'smoke', '--run-dir', str(path)])


def test_landing_cancel_does_not_reject_successful_recovery():
    class LandingEnv(ScriptEnv):
        def step(self, action):
            obs, reward, term, trunc, info = super().step(action)
            if self.i == 4:
                info.update(semantic_execution_status='cancelled', semantic_execution_reason='context_changed')
            return obs, reward, term, trunc, info

    env = LandingEnv([observation(), observation(), observation(1, 0),
                      observation(1, 0), observation(), observation()])
    assert collect_attempt(env, 'recovery', setup_steps=5, max_steps=8, min_steps=2).accepted


def test_weapon_requires_fresh_unarmed_then_pickup_and_hold():
    class WeaponEnv(ScriptEnv):
        def step(self, action):
            obs, reward, term, trunc, info = super().step(action)
            armed = self.i >= 3
            obs[StateSpec.index('player_has_weapon')] = armed
            info.update(player_weapon_visual_fresh=1, player_weapon_visual_valid=1,
                        player_weapon_visual_confidence=.99, player_weapon_visual_armed=int(armed))
            if int(action) == 47:
                info['input_taps_pressed'] = ['num5']
            return obs, reward, term, trunc, info

    observations = [observation() for _ in range(8)]
    for obs in observations:
        obs[StateSpec.index('weapon_on_ground')] = 1
        obs[StateSpec.index('weapon_dx')] = .01
    result = collect_attempt(WeaponEnv(observations), 'weapon', setup_steps=4,
                             max_steps=8, hold_steps=3, min_steps=2)
    assert result.accepted
    assert result.steps == 5  # Hold starts after the second independent positive.


def test_collector_always_releases_on_error():
    env = ScriptEnv([observation()])
    def fail(action):
        raise RuntimeError('capture broke')
    env.step = fail
    with pytest.raises(RuntimeError, match='capture broke'):
        collect_attempt(env, 'combat')
    assert env.released


@pytest.mark.parametrize('status,reason', [('cancelled', 'sequence_deadline'),
                                          ('rejected', 'context_mask')])
def test_real_execution_failures_still_reject_attempt(status, reason):
    class FailedInput(ScriptEnv):
        def step(self, action):
            obs, reward, term, trunc, info = super().step(action)
            if self.i >= 2:
                info.update(semantic_execution_status=status, semantic_execution_reason=reason)
            return obs, reward, term, trunc, info
    env = FailedInput([observation()] * 8)
    events = []
    result = collect_attempt(env, 'combat', max_steps=6, on_event=events.append)
    assert not result.accepted
    assert result.reason == 'input_execution_rejected'
    assert events[-1]['execution_reason'] == reason
    assert env.released


def test_confirmed_pickup_followed_by_sustained_negative_evidence_is_rejected():
    class DroppedWeapon(ScriptEnv):
        def step(self, action):
            obs, reward, term, trunc, info = super().step(action)
            armed = 3 <= self.i <= 4
            obs[StateSpec.index('player_has_weapon')] = armed
            info.update(player_weapon_visual_fresh=1, player_weapon_visual_valid=1,
                        player_weapon_visual_confidence=.99, player_weapon_visual_armed=int(armed))
            if int(action) == 47:
                info['input_taps_pressed'] = ['num5']
            return obs, reward, term, trunc, info
    states = [observation() for _ in range(15)]
    for obs in states:
        obs[StateSpec.index('weapon_on_ground')] = 1
        obs[StateSpec.index('weapon_dx')] = .01
    events = []
    result = collect_attempt(DroppedWeapon(states), 'weapon', max_steps=12,
                             hold_steps=40, on_event=events.append)
    assert not result.accepted
    assert result.reason == 'weapon_dropped_after_pickup'
    assert result.pickup_requests >= 1
    assert result.peak_hold_steps == 1
    assert result.negative_refreshes == 4
    assert any(event['event'] == 'pickup_pressed' for event in events)


def test_collection_event_log_preserves_failure_details_and_is_never_overwritten(tmp_path, monkeypatch):
    import brawl.pipelines.semantic57.collect as collector
    def attempt(env, drill, *, on_event, **kwargs):
        on_event(dict(event='execution_failed', execution_reason='sequence_deadline'))
        return collector.Attempt(reason='input_execution_rejected',
                                 execution_status='cancelled', execution_reason='sequence_deadline')
    monkeypatch.setattr(collector, 'collect_attempt', attempt)
    path = tmp_path / 'collection_events.jsonl'
    _, summary = collector.collect_demonstrations(ScriptEnv([observation()]), episodes=2,
                                                  attempts_factor=1, events_path=path)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert not summary['complete']
    assert len(events) == 12
    assert events[0]['drill'] == 'weapon' and events[0]['attempt'] == 1
    assert events[-1]['event'] == 'attempt_end'
    assert events[-1]['execution_reason'] == 'sequence_deadline'
    with pytest.raises(FileExistsError):
        collector.collect_demonstrations(None, events_path=path)


@pytest.mark.parametrize('extra', [ ['--batch-size', '3'], ['--hours', 'nan'],
    ['--delay', '61'], ['--episodes-per-drill', '1'], ['--learning-rate', '0'],
    ['--stage', 'pretrain'], ['--stage', 'train', '--experimental-controls'] ])
def test_cli_rejects_invalid_configuration(extra):
    with pytest.raises(SystemExit):
        main(['--stage', 'smoke', *extra])


def test_all_stages_with_fake_live_environment(tmp_path, monkeypatch):
    from brawl.pipelines.semantic57.demos import save_demos
    import brawl.pipelines.semantic57.collect as collector
    import brawl.pipelines.semantic57.run as pipeline

    data = DemoBatch(np.stack([observation()] * 32), np.ones(32, np.int64),
                     np.repeat(np.arange(4), 8), np.array(['movement']*32))
    monkeypatch.setattr(collector, 'collect_demonstrations', lambda *a, **kw: (data, {'complete': True}))
    monkeypatch.setattr(pipeline, '_make_live', lambda args: ScriptEnv([observation()]))
    checkpoint = tmp_path / 'fake.pt'
    checkpoint.touch()
    path = tmp_path / 'all'
    main(['--stage', 'all', '--experimental-controls', '--no-prompt', '--delay', '0',
          '--weapon-classifier-checkpoint', str(checkpoint), '--run-dir', str(path),
          '--bc-epochs', '1', '--n-steps', '8', '--batch-size', '4', '--timesteps', '16',
          '--window-steps', '8', '--eval-every', '8', '--eval-episodes', '1',
          '--world-replay', str(tmp_path / 'replay')])
    assert (path / 'latest.zip').exists()
    assert (path / 'training_progress.png').exists()
    manifests = list((tmp_path / 'replay').glob('*/manifest.json'))
    assert len(manifests) == 2
    assert all(json.loads(p.read_text())['policy_contract']['action_dim'] == 57 for p in manifests)


def test_interrupted_collection_returns_previously_accepted_episodes(monkeypatch):
    import brawl.pipelines.semantic57.collect as collector
    calls = []
    def attempt(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return collector.Attempt(True, 'accepted', [observation()]*4, [1]*4, ['weapon']*4)
    monkeypatch.setattr(collector, 'collect_attempt', attempt)
    data, summary = collector.collect_demonstrations(ScriptEnv([observation()]), episodes=2)
    assert len(data.actions) == 4
    assert summary['interrupted'] and not summary['complete']
    assert summary['episodes_collected'] == 1


def test_separate_pretrain_train_and_evaluate_stages(tmp_path, monkeypatch):
    from brawl.pipelines.semantic57.demos import save_demos
    import brawl.pipelines.semantic57.run as pipeline
    data = DemoBatch(np.stack([observation()] * 32), np.ones(32, np.int64),
                     np.repeat(np.arange(4), 8), np.array(['movement']*32))
    demos = tmp_path / 'demos.npz'
    save_demos(demos, data, {})
    bc = tmp_path / 'bc'
    main(['--stage', 'pretrain', '--demos', str(demos), '--run-dir', str(bc), '--bc-epochs', '1'])
    monkeypatch.setattr(pipeline, '_make_live', lambda args: ScriptEnv([observation()]))
    checkpoint = tmp_path / 'classifier.pt'
    checkpoint.touch()
    common = ['--experimental-controls', '--no-prompt', '--delay', '0',
              '--weapon-classifier-checkpoint', str(checkpoint), '--n-steps', '8', '--batch-size', '4',
              '--window-steps', '8', '--eval-episodes', '1']
    trained = tmp_path / 'trained'
    main(['--stage', 'train', '--resume', str(bc / 'bc_init.zip'), '--demos', str(demos),
          '--run-dir', str(trained), '--timesteps', '16', '--eval-every', '0', *common])
    model = load_model(trained / 'latest.zip')
    assert model.n_steps == 8 and model.num_timesteps == 16
    assert (trained / 'bc_agreement.jsonl').is_file()
    evaluated = tmp_path / 'eval'
    main(['--stage', 'evaluate', '--resume', str(trained / 'latest.zip'),
          '--run-dir', str(evaluated), *common])
    assert (evaluated / 'evaluations.csv').is_file()
    assert json.loads((evaluated / 'status.json').read_text())['status'] == 'completed'


@pytest.mark.parametrize('interrupted', [False, True])
def test_partial_collection_saved_without_running_bc(tmp_path, monkeypatch, interrupted):
    import brawl.pipelines.semantic57.collect as collector
    import brawl.pipelines.semantic57.run as pipeline
    data = DemoBatch(np.stack([observation()] * 4), np.ones(4, np.int64),
                     np.zeros(4, np.int64), np.array(['weapon']*4))
    monkeypatch.setattr(collector, 'collect_demonstrations', lambda *a, **kw:
                        (data, {'complete': False, 'interrupted': interrupted}))
    monkeypatch.setattr(pipeline, '_make_live', lambda args: ScriptEnv([observation()]))
    checkpoint = tmp_path / 'classifier.pt'
    checkpoint.touch()
    out = tmp_path / 'partial'
    cmd = ['--stage', 'all', '--experimental-controls', '--no-prompt', '--delay', '0',
           '--weapon-classifier-checkpoint', str(checkpoint), '--run-dir', str(out)]
    if interrupted:
        with pytest.raises(SystemExit) as exc:
            main(cmd)
        assert exc.value.code == 130
    else:
        with pytest.raises(RuntimeError, match='quotas'):
            main(cmd)
    assert (out / 'partial_demos.npz').is_file()
    assert not (out / 'bc_init.zip').exists()


@pytest.mark.parametrize('failure', [RuntimeError('broken step'), KeyboardInterrupt()])
def test_training_failure_saves_checkpoint_and_releases(tmp_path, monkeypatch, failure):
    import brawl.pipelines.semantic57.run as pipeline
    env = ScriptEnv([observation()])
    model = build_model(env, n_steps=8, batch_size=4, n_epochs=1)
    source = tmp_path / 'source.zip'
    save_model(model, source)
    def broken(action):
        raise failure
    env.step = broken
    monkeypatch.setattr(pipeline, '_make_live', lambda args: env)
    checkpoint = tmp_path / 'classifier.pt'
    checkpoint.touch()
    out = tmp_path / 'failed'
    cmd = ['--stage', 'train', '--resume', str(source), '--experimental-controls', '--no-prompt',
           '--delay', '0', '--weapon-classifier-checkpoint', str(checkpoint), '--run-dir', str(out)]
    with pytest.raises((RuntimeError, SystemExit)):
        main(cmd)
    assert env.released
    name = 'interrupted' if isinstance(failure, KeyboardInterrupt) else 'failed_run'
    assert (out / f'{name}.zip.metadata.json').exists()
