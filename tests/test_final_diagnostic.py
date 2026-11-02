from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch


class Clock:
    def __init__(self):
        self.t = 0.

    def __call__(self):
        self.t += .001
        return self.t


class Frames:
    def __init__(self):
        self.frame = np.zeros((96, 160, 3), dtype=np.uint8)

    def get_frame(self):
        return self.frame

    def close(self):
        pass


class Arena(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (146,), dtype=np.float32)
    action_space = gym.spaces.Discrete(27)

    def __init__(self, fail=False):
        self.frame_provider = Frames()
        self.input_controller = SimpleNamespace(
            set_pressed=lambda keys: None, key_down=lambda key: None,
            key_up=lambda key: None, reset=lambda: None)
        self.closed = False
        self.fail = fail
        self.resets = 0
        self.steps = 0

    def reset(self, **kwargs):
        self.resets += 1
        self.frame_provider.get_frame()
        return np.zeros(146, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        self.frame_provider.get_frame()
        self.input_controller.set_pressed({'d'})
        if self.fail and self.steps == 2:
            raise RuntimeError('synthetic failure')
        return np.full(146, .1, dtype=np.float32), 999., False, False, {
            'effective_action': int(action), 'damage_valid': True,
            'op_delta_damage': 10., 'self_delta_damage': 0.,
            'op_stock_lost_step': 0., 'self_stock_lost_step': 0.,
            'detections': [],
        }

    def close(self):
        self.input_controller.reset()
        self.frame_provider.close()
        self.closed = True


class Policy:
    def set_training_mode(self, training):
        assert training is False

    def obs_to_tensor(self, obs):
        return torch.as_tensor(obs).unsqueeze(0), False

    def get_distribution(self, obs):
        probs = torch.zeros((1, 27))
        probs[:, 1] = 1.
        return SimpleNamespace(distribution=torch.distributions.Categorical(probs=probs))


def test_weapon_crop_recording_requires_visual_fusion():
    from scripts.final_agent.record_final_diagnostic import parse_args
    with pytest.raises(SystemExit):
        parse_args(['--model', 'test.zip', '--weapon-crops'])
    args = parse_args(['--model', 'test.zip', '--weapon-crops',
                       '--weapon-state-source', 'visual_fusion'])
    assert args.weapon_crops


def test_weapon_crop_sink_preserves_pixels_and_capture_identity(tmp_path):
    import cv2
    from brawl.perception.weapon_classifier import WeaponPrediction
    from scripts.final_agent.record_final_diagnostic import WeaponCropSink
    capture = SimpleNamespace(capture_id=42)
    patch = np.arange(12 * 15 * 3, dtype=np.uint8).reshape(12, 15, 3)
    with (tmp_path / 'crops.jsonl').open('x', encoding='utf-8') as stream:
        sink = WeaponCropSink(tmp_path, stream, capture)
        sink('player', (.2, .3, .04, .09), patch, WeaponPrediction(False, .95))
    entry = json.loads((tmp_path / 'crops.jsonl').read_text())
    assert entry['observed_capture_id'] == 42
    assert entry['fighter'] == 'player'
    assert entry['label'] is None  # A prediction is never a training label.
    assert entry['prediction']['armed'] is False
    np.testing.assert_array_equal(cv2.imread(str(tmp_path / entry['path'])), patch)


@pytest.mark.parametrize('fail', [False, True])
def test_recording_crops_restores_observer_and_closes_on_error(tmp_path, fail):
    from brawl.perception.weapon_classifier import WeaponPrediction
    from scripts.final_agent.record_final_diagnostic import record_diagnostic

    class CropArena(Arena):
        def step(self, action):
            result = super().step(action)
            self.weapon_classifier.prediction_observer(
                'player', (.5, .5, .1, .1), np.zeros((6, 9, 3), np.uint8),
                WeaponPrediction(False, .9))
            return result

    base = CropArena(fail=fail)
    original = lambda *args: None
    base.weapon_classifier = SimpleNamespace(
        prediction_observer=original, width=96, height=144, pad=.45,
        classes=['unarmed', 'armed'], device='cpu')
    kwargs = dict(windows=1, max_steps=3, weapon_crops=True, clock=Clock())
    if fail:
        with pytest.raises(RuntimeError, match='synthetic failure'):
            record_diagnostic(SimpleNamespace(policy=Policy()), base, tmp_path, **kwargs)
    else:
        record_diagnostic(SimpleNamespace(policy=Policy()), base, tmp_path, **kwargs)
    assert base.closed
    assert base.weapon_classifier.prediction_observer is original
    entries = [json.loads(line) for line in (tmp_path / 'weapon_crops/crops.jsonl').read_text().splitlines()]
    assert len(entries) == (1 if fail else 3)
    assert (tmp_path / 'weapon_crops/contract.json').exists()


def test_disabled_classifier_cannot_record_crops_and_closes_arena(tmp_path):
    from scripts.final_agent.record_final_diagnostic import record_diagnostic
    base = Arena()
    with pytest.raises(ValueError, match='enabled visual classifier'):
        record_diagnostic(SimpleNamespace(policy=Policy()), base, tmp_path, weapon_crops=True)
    assert base.closed


def test_capture_probe_owns_frame_and_does_not_claim_sensor_freshness():
    from scripts.final_agent.record_final_diagnostic import CaptureProbe
    provider = Frames()
    probe = CaptureProbe(provider, Clock())
    first = probe.get_frame()
    probe.get_frame()
    assert probe.capture_id == 1 and probe.repeated_pixels
    provider.frame[:] = 255
    assert first.max() == 0
    probe.get_frame()
    assert not probe.repeated_pixels


@pytest.mark.parametrize('video', [False, True])
def test_recording_aligns_actions_frames_probabilities_and_training_reward(tmp_path, video):
    from scripts.final_agent.record_final_diagnostic import record_diagnostic
    base = Arena()
    model = SimpleNamespace(policy=Policy())
    result = record_diagnostic(model, base, tmp_path, windows=2, seconds=120.,
                               max_steps=3, video=video, clock=Clock())
    assert result['steps'] == 6 and result['status'] == 'complete'
    assert base.closed and base.resets == 1  # windows do not physically reset game
    rows = list(csv.DictReader((tmp_path / 'transitions.csv').open()))
    assert len(rows) == 6
    assert [int(r['window']) for r in rows] == [1, 1, 1, 2, 2, 2]
    assert all(float(r['reward']) == pytest.approx(.1) for r in rows)
    assert all(float(r['p_1']) == 1. for r in rows)
    assert int(rows[0]['policy_capture_id']) == 0
    assert int(rows[0]['observed_capture_id']) == 1
    assert float(rows[0]['frame_returned_s']) < float(rows[0]['first_input_call_s'])
    assert float(rows[0]['pre_player_x']) == 0
    assert float(rows[0]['post_player_x']) == pytest.approx(.1)
    events = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert events[0]['input_calls'][0]['keys'] == ['d']
    if not video:
        assert all(r['video_frame'] == '-1' for r in rows)
    assert (tmp_path / 'diagnostic.mp4').exists() is video
    if video:
        import cv2
        cap = cv2.VideoCapture(str(tmp_path / 'diagnostic.mp4'))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
        assert cap.read()[0]
        cap.release()
        assert [int(r['video_frame']) for r in rows] == list(range(6))


def test_failure_flushes_partial_data_and_closes_env(tmp_path):
    from scripts.final_agent.record_final_diagnostic import record_diagnostic
    base = Arena(fail=True)
    with pytest.raises(RuntimeError, match='synthetic failure'):
        record_diagnostic(SimpleNamespace(policy=Policy()), base, tmp_path,
                          windows=1, max_steps=3, clock=Clock())
    assert base.closed
    assert len(list(csv.DictReader((tmp_path / 'transitions.csv').open()))) == 1
    assert json.loads((tmp_path / 'summary.json').read_text())['status'] == 'error'


def test_cli_defaults_are_frozen_cpu_and_video_is_opt_in():
    from scripts.final_agent.record_final_diagnostic import parse_args
    args = parse_args(['--model', 'test.zip'])
    assert args.windows == 2 and args.seconds == 120
    assert args.device == 'cpu' and not args.stochastic and not args.video
    with pytest.raises(SystemExit):
        parse_args(['--model', 'test.zip', '--seconds', '0'])


def test_real_env_and_ppo_keep_weights_and_show_capture_before_input(tmp_path):
    from brawl.env.live import BrawlDeepEnv, EnvConfig, NullInputController
    from brawl.pipelines.final_agent.policy import build_final_model
    from brawl.reward.match_objective import NoMatchReward
    from scripts.final_agent.record_final_diagnostic import record_diagnostic

    base = BrawlDeepEnv(
        frame_provider=Frames(), extractor=SimpleNamespace(predict=lambda frame: []),
        input_controller=NullInputController(), reward_provider=NoMatchReward(),
        config=EnvConfig(ui_regions=None, max_episode_steps=0, action_repeat_steps=1,
                         tap_latch_steps=1, weapon_classifier_enabled=False))
    model = build_final_model(base, n_steps=8, batch_size=4, verbose=0)
    weights = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    result = record_diagnostic(model, base, tmp_path, windows=1, max_steps=3)
    assert result['steps'] == 3
    assert model.num_timesteps == 0
    assert all(torch.equal(v, weights[k]) for k, v in model.policy.state_dict().items())
    rows = list(csv.DictReader((tmp_path / 'transitions.csv').open()))
    assert all(float(r['frame_returned_s']) <= float(r['first_input_call_s']) for r in rows)
    assert all(sum(float(r[f'p_{i}']) for i in range(27)) == pytest.approx(1) for r in rows)


def test_none_frame_has_no_video_frame_or_fabricated_capture_id(tmp_path):
    from scripts.final_agent.record_final_diagnostic import record_diagnostic
    base = Arena()
    sequence = iter([np.zeros((96, 160, 3), dtype=np.uint8), None])
    base.frame_provider.get_frame = lambda: next(sequence)
    record_diagnostic(SimpleNamespace(policy=Policy()), base, tmp_path,
                      windows=1, max_steps=1, video=True, clock=Clock())
    row = next(csv.DictReader((tmp_path / 'transitions.csv').open()))
    assert row['frame_valid'] == '0' and row['video_frame'] == '-1'
    assert row['observed_capture_id'] == row['policy_capture_id']


def test_encoder_failure_releases_writer_and_environment(tmp_path, monkeypatch):
    import cv2
    from scripts.final_agent.record_final_diagnostic import record_diagnostic
    released = []
    writer = SimpleNamespace(isOpened=lambda: False, release=lambda: released.append(True))
    monkeypatch.setattr(cv2, 'VideoWriter', lambda *args: writer)
    base = Arena()
    with pytest.raises(RuntimeError, match='Cannot open video encoder'):
        record_diagnostic(SimpleNamespace(policy=Policy()), base, tmp_path,
                          windows=1, max_steps=1, video=True, clock=Clock())
    assert released and base.closed


def test_keyboard_interrupt_preserves_partial_rows(tmp_path):
    from scripts.final_agent.record_final_diagnostic import record_diagnostic
    base = Arena()
    original = base.step
    def step(action):
        if base.steps == 1:
            raise KeyboardInterrupt()
        return original(action)
    base.step = step
    result = record_diagnostic(SimpleNamespace(policy=Policy()), base, tmp_path,
                               windows=1, max_steps=3, clock=Clock())
    assert result['status'] == 'interrupted' and result['steps'] == 1
    assert base.closed


def test_wall_clock_window_ends_before_step_cap(tmp_path):
    from scripts.final_agent.record_final_diagnostic import record_diagnostic
    base = Arena()
    result = record_diagnostic(SimpleNamespace(policy=Policy()), base, tmp_path,
                               windows=2, seconds=.02, max_steps=100, clock=Clock())
    assert result['completed_windows'] == 2 and result['steps'] < 200


def test_failed_capture_does_not_make_cached_policy_input_look_younger(tmp_path):
    from scripts.final_agent.record_final_diagnostic import record_diagnostic
    base = Arena()
    sequence = iter([np.zeros((96, 160, 3), dtype=np.uint8), None,
                     np.ones((96, 160, 3), dtype=np.uint8)])
    base.frame_provider.get_frame = lambda: next(sequence)
    record_diagnostic(SimpleNamespace(policy=Policy()), base, tmp_path,
                      windows=1, max_steps=2, clock=Clock())
    rows = list(csv.DictReader((tmp_path / 'transitions.csv').open()))
    assert rows[0]['policy_frame_returned_s'] == rows[1]['policy_frame_returned_s']


@pytest.mark.parametrize('source', ['action', 'visual_fusion'])
def test_entrypoint_writes_checkpoint_provenance_without_training(tmp_path, monkeypatch, source):
    import hashlib
    from brawl.env.live import EnvConfig
    from scripts.final_agent import record_final_diagnostic as diagnostic
    from brawl.pipelines.final_agent import policy as final_policy, run as final_agent
    from brawl.training import checkpoints as checkpoint_roles
    checkpoint = tmp_path / 'source.zip'
    checkpoint.write_bytes(b'synthetic-checkpoint')
    classifier_checkpoint = tmp_path / 'new classifier.pt'
    classifier_checkpoint.write_bytes(b'synthetic classifier')
    args = diagnostic.parse_args(['--model', str(checkpoint), '--out-dir', str(tmp_path / 'runs'),
                                  '--windows', '1', '--max-steps', '2', '--delay', '0',
                                  '--weapon-state-source', source,
                                  '--weapon-classifier-checkpoint', str(classifier_checkpoint)])
    monkeypatch.setattr(diagnostic, 'parse_args', lambda: args)
    monkeypatch.setattr(final_policy, 'load_final_model', lambda *a, **kw: SimpleNamespace(policy=Policy()))
    monkeypatch.setattr(checkpoint_roles, 'read_checkpoint_metadata', lambda p: {'policy_type': 'final_agent'})
    base = Arena()
    base.config = EnvConfig(ui_regions={'stock': (1, 2)},
                           weapon_classifier_checkpoint=str(classifier_checkpoint))
    if source == 'visual_fusion':
        base.weapon_classifier = object()
    def make_env(cfg):
        assert cfg.weapon_state_source == source
        assert cfg.weapon_classifier_checkpoint == str(classifier_checkpoint)
        return base
    monkeypatch.setattr(final_agent, '_make_env', make_env)
    diagnostic.main()
    run = next((tmp_path / 'runs').iterdir())
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert manifest['deterministic'] and manifest['reward_version'] == 'stock_damage_v1'
    assert manifest['env_config']['ui_regions']['stock'] == [1, 2]
    assert manifest['input_contract_version'] == 'poll_visible_edges_v2'
    assert manifest['hud_contract_version'] == 'validated_hud_v2'
    assert manifest['identity_contract_version'] == 'bounded_triangle_v2'
    assert manifest['training_env_config']['weapon_state_source'] == source
    assert manifest['training_env_config']['weapon_classifier_checkpoint'] == str(classifier_checkpoint)
    if source == 'visual_fusion':
        assert manifest['weapon_classifier_sha256'] == hashlib.sha256(classifier_checkpoint.read_bytes()).hexdigest()
    else:
        assert 'weapon_classifier_sha256' not in manifest
    assert json.loads((run / 'summary.json').read_text())['steps'] == 2
    assert base.closed
