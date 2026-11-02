from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_legacy_primitive_defaults_are_preserved():
    from brawl.pipelines.primitive27.run import parser
    args = vars(parser().parse_args([]))
    expected = dict(stage='all', run_dir=None, demos=None, resume=None, no_prompt=False,
        episodes_per_drill=30, attempts_factor=4, setup_steps=240, collection_steps=600,
        weapon_hold_steps=40, bc_epochs=40, bc_learning_rate=.0003, timesteps=4096,
        hours=0., n_steps=512, batch_size=256, learning_rate=.0001, ent_coef=.01,
        seed=42, checkpoint_every=10240, eval_every=4096, eval_episodes=3,
        window_steps=1200, delay=5., world_replay=None, live_check_attempts=3,
        weapon_classifier_checkpoint=Path('train/models/weapon_classifier_r18_live_v1.pt'))
    assert {key: args[key] for key in expected} == expected


def test_primitive_entrypoint_does_not_import_experimental_runner_or_gpu():
    result = subprocess.run([sys.executable, '-c',
        'import sys; import brawl.pipelines.primitive27.run; '
        'assert "brawl.pipelines.semantic57.run" not in sys.modules; '
        'assert "torch" not in sys.modules; assert "brawl.capture_first" not in sys.modules'],
        env=os.environ | {'BRAWL_NO_EARLY_CAPTURE': '1'}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def experiment(tmp_path, parameters=None, **extra):
    from brawl.training.experiment_config import COMPONENTS
    path = tmp_path/'experiment.json'
    path.write_text(json.dumps(dict(schema_version=1, name='test', components=COMPONENTS,
                                    parameters=parameters or {}) | extra))
    return path


def test_config_cli_and_escape_hatch_have_explicit_precedence(tmp_path):
    from brawl.training.experiment_config import parse_experiment_args
    from brawl.pipelines.primitive27.run import parser
    path = experiment(tmp_path, {'timesteps': 8192, 'learning_rate': .0002, 'no_prompt': False})
    args, spec = parse_experiment_args(parser(), ['--experiment', str(path), '--timesteps', '2048',
                                               '--set', 'learning_rate=0.00005'])
    assert args.timesteps == 2048 and args.learning_rate == .00005
    assert args.n_steps == 512 and spec['name'] == 'test'
    # Parsing must not contaminate a later invocation's defaults.
    assert parser().parse_args([]).learning_rate == .0001


@pytest.mark.parametrize('parameters', [dict(typo=1), dict(n_steps=True), dict(n_steps=1.5),
    dict(no_prompt='false'), dict(stage='bogus'), dict(learning_rate=float('nan'))])
def test_invalid_configuration_is_rejected(tmp_path, parameters):
    from brawl.training.experiment_config import parse_experiment_args
    from brawl.pipelines.primitive27.run import parser
    path = experiment(tmp_path, parameters)
    with pytest.raises(ValueError):
        parse_experiment_args(parser(), ['--experiment', str(path)])


def test_unimplemented_method_fails_before_capture(tmp_path):
    from brawl.pipelines.primitive27.run import main
    from brawl.training.experiment_config import COMPONENTS
    path = experiment(tmp_path, components=COMPONENTS | {'policy': 'lstm'})
    with pytest.raises(ValueError, match='Unsupported'):
        main(['--experiment', str(path), '--stage', 'smoke', '--run-dir', str(tmp_path/'run')])
    assert not (tmp_path/'run').exists()


def test_baseline_config_and_cli_are_equivalent():
    from brawl.training.experiment_config import parse_experiment_args
    from brawl.pipelines.primitive27.run import parser
    args, spec = parse_experiment_args(parser(), [
        '--experiment', 'experiments/baselines/structured27_bc_ppo_v1.json'])
    assert spec['name'] == 'structured27_bc_ppo_v1'
    assert (args.timesteps, args.n_steps, args.batch_size, args.eval_episodes) == (4096, 512, 256, 2)


def test_configured_smoke_records_resolved_configuration_and_hashes(tmp_path):
    from brawl.pipelines.primitive27.run import main
    path = experiment(tmp_path, {'stage': 'smoke', 'seed': 42})
    out = tmp_path/'smoke'
    main(['--experiment', str(path), '--run-dir', str(out)])
    resolved = json.loads((out/'resolved_experiment.json').read_text())
    assert resolved['parameters']['stage'] == 'smoke'
    assert resolved['components']['algorithm'] == 'ppo'
    assert resolved['artifacts']['experiment']['sha256']
    assert resolved['source_sha256']['brawl/training/runner.py']
    assert resolved['protocol']['retention']['min_retention'] == .9
    assert json.loads((out/'status.json').read_text())['synthetic']
    model = json.loads((out/'model_config.json').read_text())
    assert model['n_steps'] == 8 and model['n_epochs'] == 1  # Effective smoke overrides.
    assert model['gamma'] == .999 and model['clip_range_at_start'] == .1
    assert model['observation_shape'] == [146] and model['action_dim'] == 27


@pytest.mark.parametrize('setting', ['unknown=4', 'seed=true', 'seed=1.5', 'no_prompt=falsey',
                                    'learning_rate=NaN', 'stage=bogus', 'timesteps'])
def test_invalid_escape_hatch_is_rejected(setting):
    from brawl.training.experiment_config import parse_experiment_args
    from brawl.pipelines.primitive27.run import parser
    with pytest.raises(ValueError):
        parse_experiment_args(parser(), ['--set', setting])


def test_boolean_and_path_override(tmp_path):
    from brawl.training.experiment_config import parse_experiment_args
    from brawl.pipelines.primitive27.run import parser
    args, _ = parse_experiment_args(parser(), ['--set', 'no_prompt=false',
                                             '--set', f'world_replay={tmp_path}'])
    assert args.no_prompt is False and args.world_replay == tmp_path


def test_duplicate_keys_and_schema_typos_rejected(tmp_path):
    from brawl.training.experiment_config import parse_experiment_args
    from brawl.pipelines.primitive27.run import parser
    path = tmp_path/'invalid.json'
    for contents in ['{"schema_version":1,"schema_version":2}', '[]',
                     '{"schema_version":1,"typo":true}']:
        path.write_text(contents)
        with pytest.raises(ValueError):
            parse_experiment_args(parser(), ['--experiment', str(path)])


def test_explicit_geometry_experiment_rejects_a_different_checkpoint_encoder():
    from brawl.training.experiment_config import validate_model_components
    from brawl.pipelines.primitive27.run import _space_env
    from brawl.pipelines.primitive27.training import build_model
    from stable_baselines3.common.torch_layers import FlattenExtractor
    model = build_model(_space_env())
    validate_model_components(model)
    model = build_model(_space_env(), policy_kwargs={
        'net_arch': {'pi': [128, 128], 'vf': [128, 128]},
        'features_extractor_class': FlattenExtractor})
    with pytest.raises(ValueError, match='representation'):
        validate_model_components(model)
