"""Uncapped diagnosis must not train, change inputs, or hide failed blocks."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest


def test_uncapped_only_allowed_for_deployment():
    from brawl.pipelines.round_a.config import Protocol
    from brawl.pipelines.round_a.collection import cadence_assessment
    protocol = Protocol(control_hz=0, execution_mode='deployment')
    report = cadence_assessment([.02, .04], protocol)
    assert report['cadence_acceptable']
    assert report['overrun_fraction'] is None
    assert report['measured_hz'] == pytest.approx(1 / .03)
    assert not cadence_assessment([], protocol)['cadence_acceptable']
    with pytest.raises(ValueError):
        Protocol(control_hz=0)


def test_profile_order_balances_policy_and_logging():
    from scripts.round_a.profile_round_a import block_schedule
    schedule = block_schedule(2)
    assert len(schedule) == 8
    assert schedule[4:] == list(reversed(schedule[:4]))
    assert set(schedule) == {('bc', 'normal'), ('bc', 'minimal'),
                             ('ppo', 'normal'), ('ppo', 'minimal')}
    with pytest.raises(ValueError):
        block_schedule(0)


def test_trace_keeps_early_spikes_and_exclusive_time(tmp_path):
    from brawl.pipelines.round_a.diagnostic_timing import TraceProbe
    now = [0.]
    class Target:
        def inner(self): now[0] += .01
        def outer(self):
            now[0] += .002
            self.inner()
            now[0] += .003
    target = Target()
    probe = TraceProbe(clock=lambda: now[0])
    with probe:
        probe.wrap('outer', target, 'outer')
        probe.wrap('inner', target, 'inner')
        target.outer()
    assert 'outer' not in vars(target)
    probe.save(tmp_path)
    report = json.loads((tmp_path / 'timing.json').read_text())
    assert report['stages']['outer']['mean_ms'] == pytest.approx(15)
    assert report['stages']['outer']['self_total_s'] == pytest.approx(.005)
    assert report['stages']['inner']['p99_ms'] == pytest.approx(10)
    with np.load(tmp_path / 'timing_trace.npz') as data:
        assert len(data['elapsed_s']) == 2


def test_trace_restores_on_exception():
    from brawl.pipelines.round_a.diagnostic_timing import TraceProbe
    def original(): raise RuntimeError('failure')
    target = SimpleNamespace(call=original)
    probe = TraceProbe()
    with pytest.raises(RuntimeError, match='failure'):
        with probe:
            probe.wrap('call', target, 'call')
            target.call()
    assert target.call is original
    assert probe.summary()['stages']['call']['calls'] == 1


def test_trace_reports_truncation_without_losing_totals():
    from brawl.pipelines.round_a.diagnostic_timing import TraceProbe
    probe = TraceProbe(max_events=1)
    for _ in range(3):
        with probe.measure('step'): pass
    report = probe.summary()
    assert report['events_dropped'] == 2
    assert report['stages']['step']['calls'] == 3


def make_match(tmp_path):
    import gymnasium as gym
    from brawl.env.match import MatchTrainingEnv
    from brawl.state.state_spec import StateSpec
    class Base(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (146,), np.float32)
        action_space = gym.spaces.Discrete(27)
        _last_frame = np.zeros((36, 64, 3), np.uint8)
        releases = 0
        closed = False
        def reset(self, **kwargs):
            obs = np.zeros(146, np.float32)
            obs[StateSpec.index('identity_observed')] = 1
            return obs, {}
        def step(self, action):
            obs, _ = self.reset()
            return obs, 0., False, False, {'op_delta_damage': 1., 'damage_valid': True}
        def _release_all_inputs(self): self.releases += 1
        def close(self): self.closed = True
    return MatchTrainingEnv(Base(), mode='training_room', max_steps=3)


@pytest.mark.parametrize('mode', ['normal', 'minimal'])
def test_block_evaluates_and_flushes_without_training(tmp_path, mode):
    from scripts.round_a.profile_round_a import run_block
    from brawl.pipelines.round_a.config import Protocol
    match = make_match(tmp_path)
    model = SimpleNamespace(num_timesteps=50176,
                            predict=lambda obs, **kwargs: (np.array(1), None))
    now = [0.]
    def clock():
        now[0] += .01
        return now[0]
    report = run_block(model, match, Protocol().as_dict() | {'candidate': 'mlp'},
                       tmp_path, mode=mode, seconds=2., warmup=0,
                       clock=clock, shard_size=2)
    assert report['status'] == 'completed'
    assert report['interactions'] > 2
    assert report['damage_dealt'] == report['interactions']
    assert report['control_hz'] == 0
    assert report['source_control_hz'] == 30
    assert model.num_timesteps == 50176
    assert not match.unwrapped.closed
    assert not match._handles
    assert match.unwrapped.releases > 0
    assert (tmp_path / 'timing_trace.npz').exists()
    if mode == 'normal':
        assert (tmp_path / 'transitions.csv').exists()
        assert report['replay_shards'] > 0
    else:
        assert not (tmp_path / 'transitions.csv').exists()
        assert not (tmp_path / 'world_replay').exists()
    match.close()


def test_failed_block_keeps_evidence_and_releases_inputs(tmp_path):
    from scripts.round_a.profile_round_a import run_block
    from brawl.pipelines.round_a.config import Protocol
    match = make_match(tmp_path)
    def predict(*args, **kwargs): raise RuntimeError('predict broke')
    with pytest.raises(RuntimeError, match='predict broke'):
        run_block(SimpleNamespace(predict=predict), match,
                  Protocol().as_dict() | {'candidate': 'mlp'}, tmp_path,
                  mode='minimal', seconds=1, warmup=0)
    assert json.loads((tmp_path / 'result.json').read_text())['status'] == 'failed'
    assert (tmp_path / 'timing.json').exists()
    assert match.unwrapped.releases
    match.close()


def test_profile_contract_ignores_only_schedule_not_observations():
    from scripts.round_a.profile_round_a import validate_pair
    from brawl.pipelines.round_a.config import Protocol
    bc = dict(round_a=Protocol().as_dict() | {'candidate': 'mlp'},
              training_stage='bc_final_epoch', environment_steps=0)
    ppo = dict(round_a=bc['round_a'] | {'timesteps': 100000}, environment_steps=50176)
    validate_pair(bc, ppo)
    with pytest.raises(ValueError, match='image_width'):
        validate_pair(bc, ppo | {'round_a': ppo['round_a'] | {'image_width': 128}})
    with pytest.raises(ValueError, match='BC'):
        validate_pair(bc | {'environment_steps': 512}, ppo)


def test_summary_keeps_logging_conditions_separate():
    from scripts.round_a.profile_round_a import summarize
    rows = [dict(policy='bc', logging='normal', status='completed', seconds=10,
                 interactions=100, stocks_taken=2, stocks_lost=1,
                 damage_dealt=30, damage_received=10),
            dict(policy='bc', logging='minimal', status='failed', seconds=2,
                 interactions=20, stocks_taken=0, stocks_lost=1,
                 damage_dealt=0, damage_received=10)]
    report = summarize(rows)
    assert len(report['conditions']) == 1
    assert report['conditions'][0]['net_stocks_per_minute'] == 6
    assert len(report['blocks']) == 2
    assert not report['proves_ppo_improvement']


def test_bc_can_enter_explicit_uncapped_deployment_but_resume_cannot():
    from brawl.pipelines.round_a.config import Protocol, checkpoint_contract, validate_dataset_contract
    bc = dict(algorithm='round_a_v1', training_stage='bc_final_epoch', environment_steps=0,
              round_a=Protocol().as_dict() | {'candidate': 'mlp'})
    uncapped = Protocol(control_hz=0, execution_mode='deployment')
    assert checkpoint_contract(bc, uncapped, 'mlp', initialize_bc=True)['control_hz'] == 0
    with pytest.raises(ValueError, match='control_hz'):
        checkpoint_contract(bc | {'round_a': bc['round_a'] | {'execution_mode': 'deployment'}},
                            uncapped, 'mlp')
    dataset = dict(image_width=256, image_height=144, control_hz=30.)
    validate_dataset_contract(dataset, uncapped)
    with pytest.raises(ValueError, match='image_width'):
        validate_dataset_contract(dataset | {'image_width': 32}, uncapped)


def test_partial_block_retains_completed_interactions(tmp_path):
    from scripts.round_a.profile_round_a import run_block
    from brawl.pipelines.round_a.config import Protocol
    match = make_match(tmp_path)
    calls = [0]
    def predict(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 4:
            raise KeyboardInterrupt()
        return np.array(1), None
    with pytest.raises(KeyboardInterrupt):
        run_block(SimpleNamespace(predict=predict), match,
                  Protocol().as_dict() | {'candidate': 'mlp'}, tmp_path,
                  mode='normal', seconds=5, warmup=0, shard_size=2)
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['interactions'] == 3
    assert result['seconds'] > 0
    assert result['status'] == 'failed'
    match.close()


def test_main_orchestrates_read_only_policies_and_preserves_contracts(tmp_path, monkeypatch):
    from scripts.round_a import profile_round_a as module
    from brawl.pipelines.round_a.config import Protocol
    import brawl.pipelines.round_a.models as models
    import brawl.training.checkpoints as roles
    import brawl.pipelines.primitive27.run as pipeline
    bc, ppo, classifier = [tmp_path / name for name in ('bc.zip', 'ppo.zip', 'classifier.pt')]
    for path in (bc, ppo, classifier):
        path.write_bytes(b'fixture')
    contract = Protocol().as_dict() | {'candidate': 'mlp'}
    monkeypatch.setattr(roles, 'read_checkpoint_metadata', lambda path: dict(
        round_a=contract, environment_steps=0 if path == bc else 512,
        training_stage='bc_final_epoch' if path == bc else 'ppo'))
    def load(path, *a, **kw):
        return SimpleNamespace(num_timesteps=0 if path == bc else 512,
            policy=SimpleNamespace(set_training_mode=lambda flag: None),
            predict=lambda obs, **kw: (np.array(1), None))
    monkeypatch.setattr(models, 'load_model', load)
    base = make_match(tmp_path).unwrapped
    monkeypatch.setattr(pipeline, '_make_live', lambda args: base)
    monkeypatch.setattr('builtins.input', lambda *a: '')
    monkeypatch.setattr(module.time, 'sleep', lambda *a: None)
    real_block = module.run_block
    def fast_block(*a, **kw):
        now = [0.]
        def clock():
            now[0] += .01
            return now[0]
        return real_block(*a, **(kw | dict(clock=clock, warmup=0, shard_size=2)))
    monkeypatch.setattr(module, 'run_block', fast_block)
    output = tmp_path / 'output'
    args = ['--bc', str(bc), '--ppo', str(ppo), '--output', str(output),
            '--weapon-classifier-checkpoint', str(classifier), '--rounds', '1', '--block-seconds', '1']
    module.main(args)
    assert json.loads((output / 'status.json').read_text())['status'] == 'completed'
    summary = json.loads((output / 'summary.json').read_text())
    assert len(summary['conditions']) == 4
    assert base.closed
    assert bc.read_bytes() == ppo.read_bytes() == b'fixture'
    with pytest.raises(FileExistsError):
        module.main(args)
