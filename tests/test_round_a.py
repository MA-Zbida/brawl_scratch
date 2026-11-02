"""Round A contracts must work without capture, game input or a CUDA device."""
from __future__ import annotations

import json
import numpy as np
import pytest
import gymnasium as gym


class PixelEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (146,), np.float32)
    action_space = gym.spaces.Discrete(27)

    def __init__(self):
        self._last_frame = np.zeros((36, 64, 3), np.uint8)
        self._mirrored = False
        self.index = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return np.zeros(146, np.float32), {}

    def step(self, action):
        self.index += 1
        self._last_frame[:] = self.index % 256
        return np.full(146, self.index, np.float32), 0., False, self.index % 8 == 0, {}

    def _release_all_inputs(self):
        pass


def test_protocol_rejects_unknown_and_nonfinite_fields(tmp_path):
    from train.round_a_config import Protocol
    path = tmp_path / 'protocol.json'
    for values in ({'typo': 1}, {'control_hz': float('nan')}, {'image_width': 0}, {'timesteps': True}):
        path.write_text(json.dumps(values))
        with pytest.raises(ValueError):
            Protocol.load(path)


def test_pixel_history_is_owned_aligned_and_currently_mirrored():
    from train.round_a_data import ObservationAdapter
    raw = PixelEnv()
    env = ObservationAdapter(raw, 'multi', width=64, height=36, hz=0)
    obs, _ = env.reset()
    original = obs['frames'].copy()
    for _ in range(3):
        obs, *_ = env.step(0)
    assert obs['frames'].shape == (12, 36, 64)
    assert obs['frames'][:, 0, 0].tolist() == [3]*3 + [1]*3 + [0]*6
    assert not original.any()
    raw._last_frame[:, :32] = 200
    raw._mirrored = True
    env.capture(np.zeros(146, np.float32))
    assert env.current['frames'][0, 0, -1] == 200


@pytest.mark.parametrize('candidate', ['mlp', 'recurrent', 'multi'])
def test_candidates_learn_reload_and_predict(candidate, tmp_path):
    import torch
    torch.set_num_threads(1)
    from train.round_a_models import build_model, save_model, load_model, Predictor
    from train.round_a_data import ObservationAdapter
    env = ObservationAdapter(PixelEnv(), candidate, width=64, height=36, hz=0)
    model = build_model(candidate, env, n_steps=8, batch_size=4, n_epochs=1)
    model.learn(16)
    obs, _ = env.reset()
    a = Predictor(model).predict(obs, start=True)
    path = tmp_path / 'model.zip'
    save_model(model, path, {'candidate': candidate, 'image_width': 64, 'image_height': 36})
    restored = load_model(path, candidate, env)
    assert Predictor(restored).predict(obs, start=True) == a
    with pytest.raises(ValueError, match='candidate'):
        load_model(path, 'multi' if candidate != 'multi' else 'mlp')


def test_predictor_preserves_memory_and_resets_on_boundary():
    from train.round_a_models import Predictor
    class Model:
        def predict(self, obs, state=None, episode_start=None, deterministic=None):
            assert deterministic
            state = 0 if episode_start[0] else state
            return np.array([state]), state + 1
    predictor = Predictor(Model())
    assert predictor.predict(None, start=True) == 0
    assert predictor.predict(None) == 1
    assert predictor.predict(None, start=True) == 0


def test_dataset_contract_rejects_missing_frames_and_keeps_episode_split(tmp_path):
    from train.round_a_data import EpisodeWriter, Dataset
    writer = EpisodeWriter(tmp_path, width=64, height=36)
    for ep in range(10):
        frames = np.zeros((9, 36, 64, 3), np.uint8)
        writer.append(obs=np.zeros((8, 146), np.float32), actions=np.zeros(8, np.int64),
                      valid=np.array([False, True]*4), families=np.array(['combat']*8),
                      drill='combat', frames=frames, frame_refs=np.tile([0, 0, 0, 0], (8, 1)),
                      mirrored=np.zeros(8, bool))
    writer.finish()
    data = Dataset(tmp_path)
    split = data.split(42)
    assert not set(split['train']) & set(split['validation'])
    assert not set(split['test']) & set(split['validation'])
    assert len(split['test']) > 0
    (tmp_path / 'episode_000000' / 'frames.npy').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        Dataset(tmp_path)


def test_timed_evaluation_counts_seconds_not_steps_and_releases():
    from train.round_a_runtime import evaluate_block
    class Clock:
        t = 0.
        def __call__(self): return self.t
    clock = Clock()
    class Env:
        released = False
        def step(self, action):
            clock.t += .4
            return np.zeros(146), 0., False, False, {'op_delta_damage': 2., 'self_stock_lost_step': 1.}
        def release_inputs(self): self.released = True
    class Predictor:
        def predict(self, obs, start=False): return 0
    env = Env()
    row, _ = evaluate_block(env, Predictor(), np.zeros(146), 1., clock=clock)
    assert row['interactions'] == 3
    assert row['stocks_lost'] == 3
    assert row['damage_dealt'] == 6
    assert row['seconds'] == pytest.approx(1.2)
    assert env.released


def test_collector_labels_have_source_step_indices():
    from train.semantic_collect import Attempt
    assert Attempt().observation_steps == []


def make_dataset(path):
    from train.round_a_data import EpisodeWriter, Dataset
    writer = EpisodeWriter(path, width=64, height=36, extra={'control_hz': 30., 'complete': True})
    for drill, action in [('weapon', 26), ('recovery', 7), ('combat', 19)]:
        for _ in range(5):
            obs = np.ones((6, 146), np.float32)
            writer.append(obs=obs, actions=np.full(6, action, np.int64),
                valid=np.array([False, True, True, False, True, True]), families=np.array([drill]*6),
                drill=drill, frames=np.zeros((7, 36, 64, 3), np.uint8),
                frame_refs=np.tile(np.arange(6)[:, None], (1, 4)), mirrored=np.zeros(6, bool))
    writer.finish()
    return Dataset(path)


@pytest.mark.parametrize('candidate', ['mlp', 'recurrent', 'multi'])
def test_ordered_bc_then_complete_live_runner_without_hardware(candidate, tmp_path):
    import torch
    from dataclasses import replace
    from train.round_a_config import Protocol
    from train.round_a import space_env
    from train.round_a_data import ObservationAdapter
    from train.round_a_models import build_model, load_model
    from train.round_a_bc import pretrain, evaluate_imitation
    from train.round_a_runtime import run_training
    from train.match_env import MatchTrainingEnv
    torch.set_num_threads(1)
    data = make_dataset(tmp_path / 'data')
    protocol = Protocol(image_width=64, image_height=36, n_steps=8, batch_size=4,
        n_epochs=1, bc_epochs=1, sequence_length=3, timesteps=16,
        baseline_seconds=.02, interim_seconds=.02, final_seconds=.03,
        eval_block_seconds=.02, eval_every=8, checkpoint_every=8)
    directory = tmp_path / 'run'
    directory.mkdir()
    model = build_model(candidate, space_env(candidate, protocol), n_steps=8, batch_size=4, n_epochs=1)
    before = {key: value.clone() for key, value in model.policy.state_dict().items()}
    pretrain(model, data, protocol, directory, seed=42)
    assert any(not torch.equal(before[k], v) for k, v in model.policy.state_dict().items())
    report = evaluate_imitation(model, data, data.split()['test'])
    assert report['slices']['pickup']['samples'] == 4
    match = MatchTrainingEnv(PixelEnv(), mode='training_room', max_steps=8)
    adapter = ObservationAdapter(match, candidate, width=64, height=36, hz=0)
    contract = dict(candidate=candidate, **protocol.as_dict())
    run_training(model, adapter, match, protocol, directory, contract, data=data)
    summary = json.loads((directory / 'training_summary.json').read_text())
    assert summary['actual_training_interactions'] == 16
    assert summary['total_environment_interactions'] > 16  # eval/bridges aren't training
    assert summary['status'] == 'completed'
    rows = [json.loads(x) for x in (directory / 'evaluations.jsonl').read_text().splitlines()]
    assert {x['label'] for x in rows} == {'baseline', 'interim', 'final'}
    assert load_model(directory / 'latest.zip', candidate).num_timesteps == 16
    adapter.close()


def test_collection_context_and_label_alignment(monkeypatch, tmp_path):
    from train.round_a_collection import collect
    from train.round_a_config import Protocol
    from train.round_a_data import ObservationAdapter, Dataset
    from train.semantic_collect import Attempt
    from train.match_env import MatchTrainingEnv
    def attempt(env, drill, **kwargs):
        env.reset()
        env.step(0)  # refresh, not a label
        first = env.policy_obs()
        env.step(1)
        env.step(0)  # an input that could not be labeled
        third = env.policy_obs()
        env.step(2)
        return Attempt(accepted=True, reason='accepted', observations=[first, third],
                       actions=[1, 2], families=[drill, drill], observation_steps=[1, 3], steps=3)
    monkeypatch.setattr('train.round_a_collection.collect_attempt', attempt)
    adapter = ObservationAdapter(MatchTrainingEnv(PixelEnv(), mode='training_room'), 'mlp', width=64, height=36, hz=0)
    collect(adapter, Protocol(image_width=64, image_height=36), tmp_path, episodes=5)
    data = Dataset(tmp_path / 'demos')
    episode = data.episode(data.manifest['episodes'][0])
    assert episode['valid'].tolist() == [False, True, False, True]
    assert episode['obs'][1, 0] == 1
    assert episode['frames'][episode['frame_refs'][1, 0], 0, 0, 0] == 1
    assert episode['actions'][episode['valid']].tolist() == [1, 2]
    adapter.close()


def test_round_a_cli_import_does_not_import_torch():
    import subprocess, sys
    result = subprocess.run([sys.executable, '-c',
        "import sys; import train.round_a; assert 'torch' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_clock_paces_without_catchup_bursts():
    from train.round_a_data import ObservationAdapter
    class Clock:
        t = 0.
        def __call__(self): return self.t
        def sleep(self, seconds): self.t += seconds
    clock = Clock()
    env = ObservationAdapter(PixelEnv(), 'mlp', width=64, height=36, hz=20, clock=clock, sleep=clock.sleep)
    env.reset()
    env.step(0)
    env.step(0)
    assert clock.t == pytest.approx(.05)
    clock.t += 1
    env.step(0)
    env.step(0)
    assert clock.t == pytest.approx(1.1)
    report = env.cadence_report()
    assert report['all']['intervals'] == 3
    assert report['all']['overruns'] == 1


def test_round_a_cli_pretrain_and_preflight(tmp_path):
    from train.round_a import main
    make_dataset(tmp_path / 'data')
    shared = ['--demos', str(tmp_path/'data'), '--set', 'image_width=64', '--set', 'image_height=36']
    main(['--stage', 'preflight', '--run-dir', str(tmp_path/'preflight'), *shared])
    assert (tmp_path/'preflight'/'resolution_preview.png').is_file()
    main(['--stage', 'pretrain', '--run-dir', str(tmp_path/'bc'), '--set', 'bc_epochs=1', *shared])
    assert (tmp_path/'bc'/'bc_init.zip').is_file()


def test_bad_dataset_alignment_rejected(tmp_path):
    from train.round_a_data import Dataset
    from train.run_utils import sha256
    data = make_dataset(tmp_path / 'data')
    folder = tmp_path/'data'/'episode_000000'
    refs = np.full((6, 4), 900, dtype=np.int64)
    np.save(folder/'frame_refs.npy', refs)
    manifest = data.manifest
    manifest['episodes'][0]['hashes']['frame_refs'] = sha256(folder/'frame_refs.npy')
    (tmp_path/'data'/'dataset.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='references'):
        Dataset(tmp_path/'data')


def test_resumed_training_uses_remaining_budget_only(tmp_path):
    import torch
    from train.round_a_config import Protocol
    from train.round_a_data import ObservationAdapter
    from train.round_a_models import build_model
    from train.round_a_runtime import run_training
    from train.match_env import MatchTrainingEnv
    torch.set_num_threads(1)
    protocol = Protocol(n_steps=8, batch_size=4, n_epochs=1, timesteps=16,
        image_width=64, image_height=36, baseline_seconds=.01, final_seconds=.01)
    match = MatchTrainingEnv(PixelEnv(), mode='training_room', max_steps=8)
    adapter = ObservationAdapter(match, 'mlp', width=64, height=36, hz=0)
    model = build_model('mlp', adapter, n_steps=8, batch_size=4, n_epochs=1)
    model.num_timesteps = 8
    run_training(model, adapter, match, protocol, tmp_path, dict(candidate='mlp', **protocol.as_dict()))
    assert model.num_timesteps == 16
    adapter.close()


def test_visual_metrics_distinguish_recovery_from_death():
    from train.round_a_runtime import SkillMetrics
    from feature_extractor.memory.state_spec import StateSpec
    def obs(**kwargs):
        value = np.zeros(146, np.float32)
        for key, v in dict(identity_observed=1, **kwargs).items():
            value[StateSpec.index(key)] = v
        return value
    metrics = SkillMetrics()
    metrics.observe(obs(player_is_offstage=1), {}, .1)
    metrics.observe(obs(player_grounded=1), {}, .1)
    metrics.observe(obs(player_grounded=1), {}, .1)
    metrics.observe(obs(player_is_offstage=1), {}, .1)
    metrics.observe(obs(player_grounded=1), {'self_stock_lost_step': 1}, .1)
    assert metrics.row()['recoveries'] == 1
    assert metrics.row()['offstage_deaths'] == 1


def test_dataset_rejects_fractional_actions(tmp_path):
    from train.round_a_data import Dataset
    from train.run_utils import sha256
    data = make_dataset(tmp_path/'data')
    folder = tmp_path/'data'/'episode_000000'
    np.save(folder/'actions.npy', np.full(6, 1.5))
    data.manifest['episodes'][0]['hashes']['actions'] = sha256(folder/'actions.npy')
    (tmp_path/'data'/'dataset.json').write_text(json.dumps(data.manifest))
    with pytest.raises(ValueError, match='actions'):
        Dataset(tmp_path/'data')


@pytest.mark.parametrize('candidate', ['mlp', 'recurrent', 'multi'])
@pytest.mark.parametrize('initialize_bc', [False, True])
def test_live_entrypoint_train_and_evaluate_with_fake_game(candidate, initialize_bc, monkeypatch, tmp_path):
    from dataclasses import replace
    import torch
    from train import round_a
    from train.round_a_config import Protocol
    from train.round_a_models import build_model, save_model
    from train.round_a_data import ObservationAdapter
    from train.run_utils import sha256
    from train.match_env import MatchTrainingEnv
    torch.set_num_threads(1)
    data = make_dataset(tmp_path/'data')
    protocol = Protocol(image_width=64, image_height=36, timesteps=8, n_steps=8,
        batch_size=4, n_epochs=1, baseline_seconds=.01, final_seconds=.01)
    cfg = tmp_path/'protocol.json'
    cfg.write_text(json.dumps(protocol.as_dict()))
    model = build_model(candidate, round_a.space_env(candidate, protocol), n_steps=8, batch_size=4, n_epochs=1)
    checkpoint = tmp_path/'initial.zip'
    save_model(model, checkpoint, dict(candidate=candidate, **protocol.as_dict(),
                                     dataset_sha256=sha256(tmp_path/'data'/'dataset.json')),
               training_stage='bc_final_epoch')
    if initialize_bc:
        protocol = replace(protocol, timesteps=16, execution_mode='deployment')
        cfg.write_text(json.dumps(protocol.as_dict()))
    def live(args, protocol, directory, stage):
        match = MatchTrainingEnv(PixelEnv(), mode='training_room', max_steps=8)
        return ObservationAdapter(match, candidate, width=64, height=36, hz=0,
                                  prepare_images=not initialize_bc or candidate == 'multi'), match
    monkeypatch.setattr(round_a, 'make_live', live)
    monkeypatch.setattr(round_a, 'prompt', lambda *args: None)
    monkeypatch.setattr('builtins.input', lambda *args: 'YES')
    monkeypatch.setattr('train.round_a_collection.qualify', lambda *args: {'passed': True})
    shared = ['--candidate', candidate, '--protocol', str(cfg), '--demos', str(tmp_path/'data')]
    round_a.main(['--stage', 'train', '--profile-timing', '--run-dir', str(tmp_path/'train'),
                 '--init-bc' if initialize_bc else '--resume', str(checkpoint), *shared])
    assert json.loads((tmp_path/'train'/'status.json').read_text())['status'] == 'completed'
    assert (tmp_path/'train'/'qualification_timing.json').is_file()
    if initialize_bc:
        initialization = json.loads((tmp_path/'train'/'initialization.json').read_text())
        assert initialization['source_contract']['timesteps'] == 8
        assert initialization['destination_contract']['timesteps'] == 16
        checkpoint = tmp_path/'train'/'latest.zip'
    round_a.main(['--stage', 'evaluate', '--run-dir', str(tmp_path/'eval'), '--resume', str(checkpoint), *shared])
    assert (tmp_path/'eval'/'imitation_test.json').is_file()


def test_qualification_uses_real_predictor_and_reports_deadlines(monkeypatch, tmp_path):
    from train.round_a_collection import qualify
    from train.round_a_config import Protocol
    from train.round_a_data import ObservationAdapter
    from train.round_a_models import build_model
    from train.semantic_collect import Attempt
    from train.match_env import MatchTrainingEnv
    env = ObservationAdapter(MatchTrainingEnv(PixelEnv(), mode='training_room'), 'recurrent', width=64, height=36, hz=0)
    model = build_model('recurrent', env, n_steps=8, batch_size=4, n_epochs=1)
    def attempt(view, drill, teacher_factory, **kwargs):
        obs, _ = view.reset()
        controller = teacher_factory()
        for _ in range(3):
            decision = controller.action(obs)
            obs, *_ = view.step(decision.action)
        return Attempt(accepted=True, reason='accepted')
    monkeypatch.setattr('train.round_a_collection.collect_attempt', attempt)
    assert qualify(model, env, Protocol(), tmp_path)['passed']
    env.close()


def test_evaluation_exception_releases_keys():
    from train.round_a_runtime import evaluate_block
    class Env:
        released = False
        def step(self, action): raise RuntimeError('sensor failed')
        def release_inputs(self): self.released = True
    class Predictor:
        def predict(self, *args, **kwargs): return 0
    env = Env()
    with pytest.raises(RuntimeError, match='sensor failed'):
        evaluate_block(env, Predictor(), np.zeros(146), .01)
    assert env.released


def test_round_a_summary_plots_and_metrics(tmp_path):
    from analysis.plot_round_a import summarize
    run = tmp_path/'run'
    run.mkdir()
    (run/'manifest.json').write_text(json.dumps({'arguments': {'candidate': 'mlp', 'seed': 42}}))
    row = dict(label='final', training_step=2000384, seconds=600, stocks_taken=3,
               elapsed_run_seconds=72000,
               stocks_lost=1, damage_dealt=300, damage_received=200)
    (run/'evaluations.jsonl').write_text(json.dumps(row)+'\n')
    result = summarize([run], tmp_path/'report')
    assert result[0]['net_stocks_per_minute'] == pytest.approx(.2)
    assert result[0]['damage_trade_per_minute'] == pytest.approx(10.)
    assert (tmp_path/'report'/'learning_curves.png').is_file()
    assert (tmp_path/'report'/'wall_clock_curves.png').is_file()


def test_eval_video_is_opt_in_bounded_and_timestamped(tmp_path):
    from train.round_a_video import EvaluationClip
    clip = EvaluationClip(tmp_path, 'test', seconds=.5, fps=30)
    frame = np.zeros((36, 64, 3), np.uint8)
    clip.write(frame, elapsed=.1, action=19, info={'effective_action': 19})
    clip.write(frame, elapsed=.6, action=0, info={})
    clip.close()
    rows = [json.loads(x) for x in (tmp_path/'test.frames.jsonl').read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]['elapsed_s'] == .1
    assert (tmp_path/'test.mp4').stat().st_size > 0


def test_visual_checkpoint_cannot_lie_about_resolution(tmp_path):
    from train.round_a_data import ObservationAdapter
    from train.round_a_models import build_model, save_model
    env = ObservationAdapter(PixelEnv(), 'multi', width=64, height=36, hz=0)
    model = build_model('multi', env, n_steps=8, batch_size=4)
    with pytest.raises(ValueError, match='space'):
        save_model(model, tmp_path/'bad.zip', dict(candidate='multi', image_width=128, image_height=36))


def test_live_factory_wraps_world_replay_below_pixels(monkeypatch, tmp_path):
    from train.round_a import parser, make_live
    from train.round_a_config import Protocol
    from train.primitive_pipeline import _make_live
    monkeypatch.setattr('train.primitive_pipeline._make_live', lambda args: PixelEnv())
    args = parser().parse_args(['--stage', 'collect', '--world-replay', str(tmp_path/'world')])
    env, match = make_live(args, Protocol(image_width=64, image_height=36), tmp_path, 'collect')
    env.reset()
    env.step(0)
    env.close()
    assert list((tmp_path/'world').glob('*/manifest.json'))


def test_invalid_observation_never_reaches_policy():
    from train.round_a_data import ObservationAdapter
    env = ObservationAdapter(PixelEnv(), 'mlp', width=64, height=36, hz=0)
    with pytest.raises(RuntimeError, match='Nonfinite'):
        env.capture(np.full(146, np.nan, np.float32))
