"""Versioned experiment inputs and provenance, without importing ML or capture."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
import json
import math
from pathlib import Path

from brawl.paths import REPO_ROOT, source_files
from brawl.training.run_utils import sha256, write_json


COMPONENTS = dict(representation='structured_geometry_v1', policy='mlp_128x128',
                  algorithm='ppo', action_space='discrete27', objective='stock_damage_v1')
ROOT = REPO_ROOT


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate configuration key: {key}')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f'Nonfinite JSON number: {value}')


def _convert(action, value):
    if value is None and action.default is None:
        return None
    if isinstance(action, argparse._StoreTrueAction):
        if not isinstance(value, bool):
            raise ValueError(f'{action.dest} must be a boolean')
        return value
    if action.type is int and (isinstance(value, bool) or not isinstance(value, int)):
        raise ValueError(f'{action.dest} must be an integer')
    if action.type is float and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise ValueError(f'{action.dest} must be numeric')
    if action.type in (None, str, Path) and not isinstance(value, str):
        raise ValueError(f'{action.dest} must be a string')
    converted = action.type(value) if action.type else value
    if isinstance(converted, float) and not math.isfinite(converted):
        raise ValueError(f'{action.dest} must be finite')
    if action.choices and converted not in action.choices:
        raise ValueError(f'Unsupported {action.dest}: {converted}')
    return converted


def parse_experiment_args(parser, argv=None):
    """Precedence: old defaults < JSON parameters < explicit CLI < --set.

    Paths deliberately retain CLI semantics: relative to the invocation's cwd,
    normally the repository root, not to the configuration file's directory.
    """
    parser = deepcopy(parser)
    initial = parser.parse_args(argv)
    spec = dict(schema_version=1, name='structured27_bc_ppo_v1',
                components=deepcopy(COMPONENTS), parameters={})
    actions = {a.dest: a for a in parser._actions if a.dest not in ('help', 'experiment', 'settings')}
    if initial.experiment:
        spec = json.loads(initial.experiment.read_text(encoding='utf-8'),
                          object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        if not isinstance(spec, dict) or set(spec) != {'schema_version', 'name', 'components', 'parameters'}:
            raise ValueError('Experiment requires schema_version, name, components and parameters only')
        if type(spec['schema_version']) is not int or spec['schema_version'] != 1:
            raise ValueError('Unsupported experiment schema_version')
        if not isinstance(spec['name'], str) or not spec['name'].strip():
            raise ValueError('Experiment name must be a nonempty string')
        # Do not pretend a selector implements a method: add its adapter and
        # compatibility tests before admitting a new combination here.
        if spec['components'] != COMPONENTS:
            raise ValueError(f'Unsupported experiment components; currently supported: {COMPONENTS}')
        if not isinstance(spec['parameters'], dict):
            raise ValueError('Experiment parameters must be an object')
        defaults = {}
        for key, value in spec['parameters'].items():
            if key not in actions:
                raise ValueError(f'Unknown experiment parameter: {key}')
            defaults[key] = _convert(actions[key], value)
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    for setting in args.settings:
        key, separator, raw = setting.partition('=')
        key = key.replace('-', '_')
        if not separator or key not in actions:
            raise ValueError(f'Expected known CLI parameter KEY=VALUE, got {setting!r}')
        try:
            value = json.loads(raw, parse_constant=_invalid_constant)
        except (ValueError, json.JSONDecodeError):
            value = raw
        setattr(args, key, _convert(actions[key], value))
    return args, spec


def write_resolved_experiment(directory, args, spec, *, model=None):
    """Record what ran, including defaults, file identities and control protocol."""
    from brawl.training.primitive_retention import PrimitiveRetentionGuard
    artifacts = {}
    paths = {key: getattr(args, key) for key in ('experiment', 'demos', 'resume', 'weapon_classifier_checkpoint')}
    paths.update(detector=ROOT/'assets/yolo/best.engine', calibration=ROOT/'brawl/calibration.py')
    for name, path in paths.items():
        if path is not None:
            path = Path(path)
            artifacts[name] = dict(path=str(path.resolve()), sha256=sha256(path) if path.is_file() else None,
                                   present=path.is_file())
    # Hash source contents rather than relying on a clean Git commit; this repo
    # is often run with local changes. Hashing does not mutate those files.
    sources = source_files()
    source_hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted(set(sources)) if path.is_file()}
    versions = {}
    for package in ('torch', 'stable-baselines3', 'numpy', 'gymnasium'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    loaded = None if model is None else dict(
        policy_class=f'{type(model.policy).__module__}.{type(model.policy).__name__}',
        features_class=f'{type(model.policy.features_extractor).__module__}.{type(model.policy.features_extractor).__name__}')
    write_json(directory/'resolved_experiment.json', dict(
        schema_version=1, name=spec['name'], components=spec['components'],
        parameters={k: v for k, v in vars(args).items() if k not in ('experiment', 'settings')},
        precedence='defaults < experiment parameters < explicit CLI < --set',
        artifacts=artifacts, source_sha256=source_hashes, package_versions=versions,
        loaded_model=loaded, device='cpu',
        protocol=dict(mode='training_room_stock_exchanges_not_match_wins',
                      evaluation='deterministic', bot_configuration='manual_not_detected',
                      collection_and_qualification='non_attacking_medium_mordex',
                      training_and_evaluation='attacking_medium_mordex',
                      retention=dict(min_retention=PrimitiveRetentionGuard.MIN_RETENTION,
                                     collapse_retention=PrimitiveRetentionGuard.COLLAPSE_RETENTION,
                                     patience=PrimitiveRetentionGuard.PATIENCE))))


def write_model_config(directory, model):
    """Record effective SB3 settings after construction/load overrides, not guesses."""
    fields = ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef',
              'vf_coef', 'max_grad_norm', 'target_kl', 'normalize_advantage', 'seed',
              'learning_rate', 'policy_kwargs')
    values = {key: getattr(model, key) for key in fields}
    values.update(device=str(model.device), action_dim=int(model.action_space.n),
                  observation_shape=list(model.observation_space.shape),
                  policy_class=f'{type(model.policy).__module__}.{type(model.policy).__name__}',
                  features_class=f'{type(model.policy.features_extractor).__module__}.{type(model.policy.features_extractor).__name__}',
                  clip_range_at_start=model.clip_range(1.),
                  activation=type(model.policy.mlp_extractor.policy_net[1]).__name__)
    write_json(directory/'model_config.json', values)


def validate_model_components(model):
    """An explicit experiment must not silently inherit a different encoder."""
    extractor = type(model.policy.features_extractor)
    if (extractor.__module__, extractor.__name__) != ('brawl.pipelines.primitive27.features', 'PrimitiveGeometry'):
        raise ValueError('Checkpoint representation does not match structured_geometry_v1')
    if model.policy.net_arch != {'pi': [128, 128], 'vf': [128, 128]}:
        raise ValueError('Checkpoint policy does not match mlp_128x128')
    if (type(model).__module__, type(model).__name__) != ('stable_baselines3.ppo.ppo', 'PPO'):
        raise ValueError('Checkpoint algorithm does not match ppo')
