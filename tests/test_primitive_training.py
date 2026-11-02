from __future__ import annotations

import json
import numpy as np
import pytest
import gymnasium as gym

from brawl.control.action_space import Action
from brawl.state.state_spec import StateSpec


def observation(**values):
    obs = np.zeros(StateSpec.observation_dim((2, 4, 8)), np.float32)
    for key, value in (dict(opponent_exists=1, identity_observed=1,
                           player_grounded=1, player_has_weapon=1) | values).items():
        obs[StateSpec.index(key)] = value
    return obs


def test_close_grounded_opponent_gets_light_not_automatic_retreat():
    from brawl.pipelines.primitive27.training import PrimitiveTeacher
    for dx in (.02, .04, .055):
        result = PrimitiveTeacher().action(observation(rel_dx=dx, rel_distance=dx))
        assert result.action in (Action.LIGHT_NEUTRAL, Action.LIGHT_TOWARD)


def dataset():
    from brawl.pipelines.semantic57.demos import DemoBatch
    rows, actions, episodes, families, drills = [], [], [], [], []
    for drill_id, drill in enumerate(('weapon', 'recovery', 'combat')):
        for ep in range(4):
            for j in range(20):
                rows.append(observation(rel_dx=.04, rel_distance=.04))
                actions.append([26, 7, 19][drill_id])
                episodes.append(drill_id*4+ep)
                families.append(drill)
                drills.append(drill)
    return DemoBatch(np.stack(rows), np.array(actions), np.array(episodes),
                     np.array(families), np.array(drills))


def test_split_holds_out_whole_episodes_from_every_drill():
    from brawl.pipelines.primitive27.training import split_episodes
    data = dataset()
    train, val = split_episodes(data)
    assert not set(data.episodes[train]) & set(data.episodes[val])
    assert set(data.drills[train]) == set(data.drills[val]) == {'weapon', 'recovery', 'combat'}


def test_movement_only_policy_fails_even_with_high_overall_accuracy():
    from brawl.pipelines.primitive27.training import gate_predictions
    data = dataset()
    report = gate_predictions(data, np.ones(len(data.actions), dtype=int))
    assert not report['passed']
    assert report['slices']['attack']['accuracy'] == 0
    assert report['slices']['pickup']['accuracy'] == 0


def test_correct_predictions_and_missing_evidence_are_distinguished():
    from brawl.pipelines.primitive27.training import gate_predictions
    data = dataset()
    assert gate_predictions(data, data.actions)['passed']
    data.actions[:] = int(Action.MOVE_TOWARD)
    report = gate_predictions(data, data.actions)
    assert not report['passed']
    assert 'pickup' in report['failures']


def test_sampling_balances_behaviours_without_using_validation():
    from brawl.pipelines.primitive27.training import sampling_probabilities
    data = dataset()
    data.actions[:79] = int(Action.MOVE_TOWARD)
    indices = np.arange(80)
    probs = sampling_probabilities(data, indices)
    assert probs.shape == (80,)
    assert probs.sum() == pytest.approx(1)
    assert probs[-1] > probs[0]
    assert probs[-1] < .25


def test_weapon_geometry_is_identical_during_training_and_inference():
    import torch
    from brawl.pipelines.primitive27.training import build_model
    model = build_model(FakeEnv())
    obs = observation(weapon_dx=.03, weapon_dy=-.04)
    original = obs.copy()
    tensor = torch.as_tensor(obs[None])
    extractor = model.policy.features_extractor
    extractor.train()
    features = extractor(tensor)
    assert features.shape == (1, len(obs) + 1)
    assert features[0, StateSpec.index('weapon_dx')].item() == pytest.approx(.6)
    assert features[0, StateSpec.index('weapon_dy')].item() == pytest.approx(-.8)
    assert features[0, -1].item() == pytest.approx(1.)
    extractor.eval()
    assert torch.equal(features, extractor(tensor))
    np.testing.assert_array_equal(obs, original)
    assert torch.isfinite(extractor(torch.zeros_like(tensor))).all()


def test_weapon_geometry_survives_checkpoint_and_ppo(tmp_path):
    import torch
    from brawl.pipelines.primitive27.training import build_model, save_model, load_model
    torch.set_num_threads(1)
    model = build_model(FakeEnv(), n_steps=8, batch_size=4, n_epochs=1)
    model.learn(16)
    obs = observation(weapon_dx=-.03, weapon_dy=.04)[None]
    before = model.policy.get_distribution(torch.as_tensor(obs)).distribution.probs.detach()
    save_model(model, tmp_path/'geometry.zip')
    restored = load_model(tmp_path/'geometry.zip')
    assert restored.observation_space.shape == obs.shape[1:]
    assert restored.policy.features_extractor.features_dim == obs.shape[1] + 1
    after = restored.policy.get_distribution(torch.as_tensor(obs)).distribution.probs.detach()
    assert torch.equal(before, after)


def test_pickup_report_exposes_false_positives_without_changing_gate():
    from brawl.pipelines.primitive27.training import gate_predictions
    data = dataset()
    pred = data.actions.copy()
    pred[-2:] = int(Action.PICKUP)
    report = gate_predictions(data, pred)
    diagnostic = report['pickup_diagnostics']
    assert report['passed']
    assert diagnostic['predicted'] == 82
    assert diagnostic['correct'] == 80
    assert diagnostic['precision'] == pytest.approx(80 / 82)
    assert diagnostic['false_positives'] == 2
    assert diagnostic['predicted_while_armed'] == 82
    assert diagnostic['predicted_without_ground_weapon'] == 82
    empty = gate_predictions(data, np.ones(len(pred), dtype=int))['pickup_diagnostics']
    assert empty['predicted'] == 0 and empty['precision'] is None


def test_archive_contract_rejects_semantic57_and_preserves_drills(tmp_path):
    from brawl.pipelines.primitive27.training import save_demos, load_demos, metadata, validate_metadata
    from brawl.pipelines.semantic57.contract import metadata as semantic_metadata
    path = tmp_path / 'demos.npz'
    save_demos(path, dataset(), {})
    assert set(load_demos(path).drills) == {'weapon', 'combat', 'recovery'}
    assert metadata()['action_dim'] == 27
    with pytest.raises(ValueError):
        validate_metadata(semantic_metadata())
    with pytest.raises(FileExistsError):
        save_demos(path, dataset(), {})


def test_failed_bc_gate_cannot_start_live_training(tmp_path, monkeypatch):
    import brawl.pipelines.primitive27.run as pipeline
    from brawl.pipelines.primitive27.training import save_demos
    path = tmp_path / 'demos.npz'
    save_demos(path, dataset(), {})
    def forbidden(*a, **kw):
        pytest.fail('Failed offline gate must not start live control')
    monkeypatch.setattr(pipeline, '_make_live', forbidden)
    with pytest.raises(RuntimeError, match='BC gate'):
        pipeline.main(['--stage', 'pretrain', '--demos', str(path), '--bc-epochs', '1',
                       '--run-dir', str(tmp_path / 'run')])
    assert json.loads((tmp_path/'run'/'bc_gate.json').read_text())['passed'] is False


class FakeEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (StateSpec.observation_dim((2, 4, 8)),), np.float32)
    action_space = gym.spaces.Discrete(27)
    released = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.i = 0
        return observation(), {}

    def step(self, action):
        self.i += 1
        return observation(), 0., False, False, {'op_delta_damage': 1.}

    def _release_all_inputs(self):
        self.released = True


def test_full_runner_reuses_27_model_and_records_both_stages(tmp_path, monkeypatch):
    import brawl.pipelines.primitive27.run as pipeline
    import brawl.pipelines.primitive27.training as backend
    import brawl.pipelines.semantic57.collect as collector
    monkeypatch.setattr(collector, 'collect_demonstrations', lambda *a, **kw: (dataset(), {'complete': True}))
    monkeypatch.setattr(backend, 'pretrain', lambda *a, **kw: {'passed': True})
    monkeypatch.setattr(backend, 'agreement',
                        lambda *a, **kw: backend.gate_predictions(dataset(), dataset().actions))
    monkeypatch.setattr(pipeline, 'qualify', lambda *a, **kw: {'passed': True})
    monkeypatch.setattr(pipeline, '_make_live', lambda args: FakeEnv())
    monkeypatch.setattr('builtins.input', lambda *a: 'YES')
    classifier = tmp_path/'classifier.pt'
    classifier.touch()
    out = tmp_path/'all'
    pipeline.main(['--stage', 'all', '--run-dir', str(out), '--delay', '0',
        '--weapon-classifier-checkpoint', str(classifier), '--n-steps', '8', '--batch-size', '4',
        '--timesteps', '16', '--eval-every', '8', '--eval-episodes', '1', '--window-steps', '8',
        '--world-replay', str(tmp_path/'replay')])
    model = backend.load_model(out/'latest.zip')
    assert model.action_space.n == 27 and model.num_timesteps == 16
    assert str(model.device) == 'cpu'
    manifests = list((tmp_path/'replay').glob('*/manifest.json'))
    assert len(manifests) == 3
    assert all(json.loads(p.read_text())['policy_contract']['action_dim'] == 27 for p in manifests)
    assert json.loads((out/'operator_gate.json').read_text())['confirmed']
    assert (out/'training_progress.png').is_file()


def test_live_gate_uses_policy_not_teacher_and_stops_movement_only(tmp_path):
    from brawl.pipelines.primitive27.run import qualify, parser
    class MovementOnly:
        def predict(self, obs, deterministic=False):
            assert deterministic
            return 1, None
    args = parser().parse_args(['--setup-steps', '2', '--collection-steps', '5', '--live-check-attempts', '1'])
    env = FakeEnv()
    result = qualify(MovementOnly(), env, args, tmp_path)
    assert not result['passed']
    assert all(not a['accepted'] for attempts in result['drills'].values() for a in attempts)
    assert env.released


def test_27_collector_records_drill_strata_and_requires_actual_hits(monkeypatch):
    from brawl.pipelines.semantic57.collect import collect_attempt
    from brawl.pipelines.primitive27.training import PrimitiveControls
    env = FakeEnv()
    # Damage alone is not enough: this fake env never injects an attack.
    failed = collect_attempt(env, 'combat', controls=PrimitiveControls(), max_steps=4)
    assert not failed.accepted and failed.reason == 'no_confirmed_combat'
    assert not failed.actions  # An attack request without its key edge is not a BC label.
    original_step = env.step
    def attack(action):
        obs, r, t, tr, info = original_step(action)
        info['input_taps_pressed'] = ['num4']
        return obs, r, t, tr, info
    monkeypatch.setattr(env, 'step', attack)
    result = collect_attempt(env, 'combat', controls=PrimitiveControls(), max_steps=4)
    assert result.accepted and result.attack_presses == 4


def test_primitive_smoke_and_non_overwrite(tmp_path):
    from brawl.pipelines.primitive27.run import main
    main(['--stage', 'smoke', '--run-dir', str(tmp_path/'smoke')])
    assert json.loads((tmp_path/'smoke'/'status.json').read_text())['synthetic']
    with pytest.raises(FileExistsError):
        main(['--stage', 'smoke', '--run-dir', str(tmp_path/'smoke')])


@pytest.mark.parametrize('flags', [['--live-check-attempts', '0'], ['--batch-size', '3'], ['--hours', 'nan']])
def test_primitive_cli_rejects_invalid_configuration(flags):
    from brawl.pipelines.primitive27.run import main
    with pytest.raises(SystemExit):
        main(['--stage', 'smoke', *flags])


def test_typed_model_rejects_changed_contract(tmp_path):
    from brawl.pipelines.primitive27.training import build_model, save_model, load_model
    path = tmp_path/'policy.zip'
    save_model(build_model(FakeEnv(), n_steps=8, batch_size=4), path)
    meta_path = path.with_name(path.name+'.metadata.json')
    meta = json.loads(meta_path.read_text())
    meta['input_contract'] = 'different_timing'
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='input_contract'):
        load_model(path)


def test_bc_can_learn_distinct_actions_instead_of_movement_only(tmp_path):
    from brawl.pipelines.primitive27.training import pretrain, build_model
    data = dataset()
    for drill_id, drill in enumerate(('weapon', 'recovery', 'combat')):
        selected = data.drills == drill
        data.obs[selected, 0:3] = 0
        data.obs[selected, drill_id] = 1
    model = build_model(FakeEnv(), n_steps=8, batch_size=4)
    report = pretrain(model, data, epochs=30, batch_size=32, learning_rate=.003, output=tmp_path)
    assert report['passed']


def test_ppo_stops_when_heldout_skill_gate_regresses(tmp_path, monkeypatch):
    from brawl.pipelines.primitive27.run import _wrap, parser
    from brawl.pipelines.semantic57.run import _train
    import brawl.pipelines.primitive27.training as backend
    calls = []
    def agreement(*a, **kw):
        calls.append(1)
        data = dataset()
        predictions = data.actions.copy() if len(calls) == 1 else np.ones_like(data.actions)
        return backend.gate_predictions(data, predictions)
    monkeypatch.setattr(backend, 'agreement', agreement)
    args = parser().parse_args(['--n-steps', '8', '--batch-size', '4', '--timesteps', '24',
                               '--eval-every', '0', '--window-steps', '8'])
    wrapped, match = _wrap(FakeEnv(), args, tmp_path, 'train')
    model = backend.build_model(wrapped, n_steps=8, batch_size=4)
    with pytest.raises(RuntimeError, match='retention gate'):
        _train(model, wrapped, match, args, tmp_path, dataset(), backend=backend)
    assert model.num_timesteps < 24
    assert (tmp_path/'retention_stop.zip').exists()


def test_ppo_can_continue_after_one_pickup_label_flips(tmp_path, monkeypatch):
    from brawl.pipelines.primitive27.run import _wrap, parser
    from brawl.pipelines.semantic57.run import _train
    import brawl.pipelines.primitive27.training as backend
    def agreement(model, *a, **kw):
        data = dataset()
        report = backend.gate_predictions(data, data.actions)
        correct = 20 if model.num_timesteps == 0 else 19
        report['slices']['pickup'].update(samples=33, accuracy=correct/33, passed=correct >= 20)
        report.update(passed=correct >= 20, failures=[] if correct >= 20 else ['pickup'])
        return report
    monkeypatch.setattr(backend, 'agreement', agreement)
    args = parser().parse_args(['--n-steps', '8', '--batch-size', '4', '--timesteps', '16',
                               '--eval-every', '0', '--window-steps', '8'])
    wrapped, match = _wrap(FakeEnv(), args, tmp_path, 'train')
    model = backend.build_model(wrapped, n_steps=8, batch_size=4)
    _train(model, wrapped, match, args, tmp_path, dataset(), backend=backend)
    assert model.num_timesteps == 16
    assert (tmp_path/'latest.zip').exists()
    assert not (tmp_path/'retention_stop.zip').exists()
    last = json.loads((tmp_path/'bc_agreement.jsonl').read_text().splitlines()[-1])
    assert last['retention']['action'] == 'continue'
    assert not last['teacher_agreement']['passed']  # Keep the raw failed threshold visible.


def test_primitive_cli_does_not_expose_semantic_opt_in():
    from brawl.pipelines.primitive27.run import parser
    assert '--experimental-controls' not in parser().format_help()


def test_default_collection_can_supply_five_heldout_pickup_episodes():
    from brawl.pipelines.primitive27.run import parser
    assert np.ceil(.2 * parser().parse_args([]).episodes_per_drill) >= 5


@pytest.mark.parametrize('values,family', [
    ({'identity_observed': 0}, 'unobserved'),
    ({'player_is_offstage': 1, 'player_grounded': 0, 'player_jumps_norm': 1}, 'recovery'),
    ({'player_has_weapon': 0, 'weapon_on_ground': 1, 'weapon_dx': .01}, 'weapon'),
    ({'opponent_exists': 0, 'signed_dx_to_stage_center': .2}, 'movement'),
    ({'opponent_is_offstage': 1, 'signed_dx_to_stage_center': -.2}, 'movement'),
])
def test_teacher_priorities_remain_explicit(values, family):
    from brawl.pipelines.primitive27.training import PrimitiveTeacher
    assert PrimitiveTeacher().action(observation(**values)).family == family


def test_live_qualification_pass_records_all_drills(tmp_path, monkeypatch):
    from brawl.pipelines.primitive27.run import qualify, parser
    import brawl.pipelines.semantic57.collect as collector
    seen = []
    class Policy:
        def predict(self, obs, deterministic):
            return 24, None
    def attempt(env, drill, *, teacher_factory, **kw):
        # Exact policy output, even when the heuristic would have used a light.
        assert teacher_factory().action(observation(rel_dx=.04)).action == 24
        seen.append(drill)
        return collector.Attempt(accepted=True, reason='accepted')
    monkeypatch.setattr(collector, 'collect_attempt', attempt)
    assert qualify(Policy(), FakeEnv(), parser().parse_args([]), tmp_path)['passed']
    assert seen == ['weapon', 'recovery', 'combat']


def test_identity_loss_releases_primitive_inputs_and_marks_unexecuted(monkeypatch):
    from types import SimpleNamespace
    from brawl.pipelines.primitive27.env import PrimitiveBrawlEnv
    env = PrimitiveBrawlEnv.__new__(PrimitiveBrawlEnv)
    env.memory = SimpleNamespace(identity_observed=False)
    calls = []
    monkeypatch.setattr(env, '_release_all_inputs', lambda: calls.append('released'))
    assert env._apply_action(int(Action.LIGHT_TOWARD)) == set()
    assert env.input_blocked and calls == ['released']
