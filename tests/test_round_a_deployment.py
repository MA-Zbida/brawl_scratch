"""Deployment benchmarks report latency without relaxing data or skill checks."""
from __future__ import annotations

from dataclasses import replace
import json

import gymnasium as gym
import numpy as np
import pytest


class FrameEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (146,), np.float32)
    action_space = gym.spaces.Discrete(27)

    def __init__(self):
        self._last_frame = np.zeros((36, 64, 3), np.uint8)

    def reset(self, **kwargs):
        return np.arange(146, dtype=np.float32), {}


@pytest.mark.parametrize('candidate', ['mlp', 'recurrent'])
def test_structured_deployment_skips_unused_images(candidate, monkeypatch):
    import cv2
    from train.round_a_data import ObservationAdapter
    raw = FrameEnv()
    env = ObservationAdapter(raw, candidate, width=64, height=36, hz=0, prepare_images=False)
    monkeypatch.setattr(cv2, 'resize', lambda *a, **k: pytest.fail('Unused pixel resize'))
    obs, _ = env.reset()
    np.testing.assert_array_equal(obs, np.arange(146, dtype=np.float32))
    obs[:] = -9
    assert env.policy_obs()[0] == 0
    assert 'frames' not in env.current
    with pytest.raises(RuntimeError, match='images'):
        env.begin_recording()
    raw._last_frame = None
    with pytest.raises(RuntimeError, match='Missing frame'):
        env.capture(np.zeros(146))


def test_visual_arm_cannot_disable_images():
    from train.round_a_data import ObservationAdapter
    with pytest.raises(ValueError, match='multi'):
        ObservationAdapter(FrameEnv(), 'multi', width=64, height=36, hz=0, prepare_images=False)


@pytest.mark.parametrize('mode,expected', [('fixed_cadence', False), ('deployment', True)])
def test_slow_cadence_is_reported_not_hidden(mode, expected):
    from train.round_a_config import Protocol
    from train.round_a_collection import cadence_assessment
    report = cadence_assessment([.05] * 100, Protocol(execution_mode=mode))
    assert report['cadence_acceptable'] is expected
    assert not report['cadence_passed']
    assert report['measured_hz'] == pytest.approx(20)
    assert report['overrun_fraction'] == 1
    assert report['decision_interval_p95_s'] == pytest.approx(.05)


@pytest.mark.parametrize('intervals', [[], [float('nan')], [float('inf')], [0.], [-1.]])
def test_missing_or_invalid_timing_still_blocks(intervals):
    from train.round_a_config import Protocol
    from train.round_a_collection import cadence_assessment
    assert not cadence_assessment(intervals, Protocol(execution_mode='deployment'))['cadence_acceptable']


def bc_payload():
    from train.round_a_config import Protocol
    return dict(algorithm='round_a_v1', training_stage='bc_final_epoch', environment_steps=0,
                round_a=dict(candidate='mlp', **Protocol(timesteps=4096).as_dict(), dataset_sha256='same'))


def test_bc_promotion_changes_schedule_not_learning_or_inputs():
    from train.round_a_config import Protocol, checkpoint_contract
    old = bc_payload()
    new = Protocol(timesteps=2_000_000, execution_mode='deployment')
    contract = checkpoint_contract(old, new, 'mlp', initialize_bc=True)
    assert contract['timesteps'] == 2_000_000
    assert contract['execution_mode'] == 'deployment'
    assert contract['dataset_sha256'] == 'same'
    assert old['round_a']['timesteps'] == 4096
    for field, value in [('image_width', 128), ('learning_rate', .002), ('bc_epochs', 41), ('control_hz', 20.)]:
        with pytest.raises(ValueError, match=field):
            checkpoint_contract(old, replace(new, **{field: value}), 'mlp', initialize_bc=True)


@pytest.mark.parametrize('mutation', [{'environment_steps': 512}, {'training_stage': 'ppo'}, {'environment_steps': None}])
def test_only_proven_zero_step_bc_can_be_promoted(mutation):
    from train.round_a_config import Protocol, checkpoint_contract
    with pytest.raises(ValueError, match='BC'):
        checkpoint_contract(bc_payload() | mutation, Protocol(execution_mode='deployment'), 'mlp', initialize_bc=True)


def test_resume_stays_strict_and_legacy_defaults_are_fixed_cadence():
    from train.round_a_config import Protocol, checkpoint_contract
    old = bc_payload()
    del old['round_a']['execution_mode']
    assert checkpoint_contract(old, Protocol(timesteps=4096), 'mlp')['execution_mode'] == 'fixed_cadence'
    with pytest.raises(ValueError, match='mismatch'):
        checkpoint_contract(old, Protocol(execution_mode='deployment'), 'mlp')
    with pytest.raises(ValueError, match='candidate'):
        checkpoint_contract(old, Protocol(timesteps=4096), 'multi')


def test_execution_mode_is_validated():
    from train.round_a_config import Protocol
    with pytest.raises(ValueError, match='execution_mode'):
        Protocol(execution_mode='ignore_everything')


@pytest.mark.parametrize('skills_pass', [False, True])
def test_deployment_qualification_does_not_bypass_skills(skills_pass, monkeypatch, tmp_path):
    from train.round_a_config import Protocol
    from train.round_a_collection import qualify
    from train.round_a_data import ObservationAdapter
    from train.semantic_collect import Attempt
    adapter = ObservationAdapter(FrameEnv(), 'mlp', width=64, height=36, hz=0, prepare_images=False)
    def attempt(view, drill, **kwargs):
        adapter.intervals.extend([.05] * 10)
        return Attempt(accepted=skills_pass, reason='accepted' if skills_pass else 'stock_lost')
    monkeypatch.setattr('train.round_a_collection.collect_attempt', attempt)
    report = qualify(None, adapter, Protocol(execution_mode='deployment'), tmp_path)
    assert report['passed'] is skills_pass
    assert report['skills_passed'] is skills_pass
    assert not report['cadence_passed']


def test_report_keeps_stopped_run_and_protocol_identity(tmp_path):
    from analysis.plot_round_a import summarize
    run = tmp_path / 'stopped'
    run.mkdir()
    (run / 'manifest.json').write_text(json.dumps(dict(
        arguments=dict(candidate='mlp', seed=42), protocol=dict(execution_mode='deployment', timesteps=2000000))))
    (run / 'status.json').write_text(json.dumps(dict(status='stopped', error='sensor failure')))
    summarize([run], tmp_path / 'report')
    inventory = json.loads((tmp_path / 'report' / 'run_inventory.json').read_text())
    assert len(inventory) == 1
    assert inventory[0]['status']['status'] == 'stopped'
    assert inventory[0]['protocol']['execution_mode'] == 'deployment'
