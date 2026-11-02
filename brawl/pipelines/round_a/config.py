"""Strict, shared Round A protocol. Safe to import before capture/CUDA."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path

CANDIDATES = ('mlp', 'recurrent', 'multi')


@dataclass(frozen=True)
class Protocol:
    schema_version: int = 1
    timesteps: int = 10_000
    control_hz: float = 30.
    image_width: int = 256
    image_height: int = 144
    n_steps: int = 512
    batch_size: int = 256
    n_epochs: int = 4
    learning_rate: float = .0001
    ent_coef: float = .01
    bc_epochs: int = 40
    bc_learning_rate: float = .0003
    sequence_length: int = 32
    split_seed: int = 42
    eval_every: int = 250_000
    baseline_seconds: float = 900.
    interim_seconds: float = 300.
    final_seconds: float = 3600.
    eval_block_seconds: float = 600.
    window_steps: int = 1200
    checkpoint_every: int = 50_000
    # Legacy experiments require matched cadence; deployment measures its cost.
    execution_mode: str = 'fixed_cadence'

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == 'execution_mode':
                if value not in ('fixed_cadence', 'deployment'):
                    raise ValueError('Invalid execution_mode')
                continue
            expected = type(field.default)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'Invalid {field.name}')
            if expected is int and not isinstance(value, int):
                raise ValueError(f'{field.name} must be an integer')
            zero_allowed = field.name in ('ent_coef', 'split_seed') or (
                field.name == 'control_hz' and self.execution_mode == 'deployment')
            if not math.isfinite(value) or value < 0 or (value == 0 and not zero_allowed):
                raise ValueError(f'Invalid {field.name}')
        if self.schema_version != 1:
            raise ValueError('Unknown schema_version')
        if self.n_steps < 2 or self.batch_size < 2 or self.n_steps % self.batch_size:
            raise ValueError('batch_size must divide n_steps; both >= 2')
        if self.image_width < 32 or self.image_height < 32:
            raise ValueError('Images must be at least 32x32')

    @classmethod
    def load(cls, path: Path):
        def unique(pairs):
            out = {}
            for key, value in pairs:
                if key in out:
                    raise ValueError(f'Duplicate protocol key: {key}')
                out[key] = value
            return out
        payload = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)
        if not isinstance(payload, dict) or set(payload) - {f.name for f in fields(cls)}:
            raise ValueError('Unknown protocol fields')
        return cls(**payload)

    def as_dict(self):
        return asdict(self)

    @property
    def rollout_budget(self):
        # All candidates get the same complete-rollout rounding (2,000,384).
        return math.ceil(self.timesteps / self.n_steps) * self.n_steps


def checkpoint_contract(payload: dict, protocol: Protocol, candidate: str,
                        *, initialize_bc: bool = False) -> dict:
    """Only a zero-interaction BC artifact may enter a new deployment schedule.

    Optimizer settings, data, features and resolution remain strict. A PPO resume
    cannot masquerade as a fresh 2M-step experiment or silently change its mode.
    """
    old = {'execution_mode': 'fixed_cadence', **payload.get('round_a', {})}
    if payload.get('algorithm') != 'round_a_v1' or old.get('candidate') != candidate:
        raise ValueError('Wrong Round A checkpoint candidate')
    allowed = set()
    if initialize_bc:
        if payload.get('training_stage') != 'bc_final_epoch' or payload.get('environment_steps') != 0:
            raise ValueError('--init-bc requires a proven zero-step BC initializer')
        allowed = {'timesteps', 'eval_every', 'baseline_seconds', 'interim_seconds',
                   'final_seconds', 'eval_block_seconds', 'checkpoint_every', 'execution_mode'}
        if protocol.execution_mode == 'deployment' and protocol.control_hz == 0:
            # Explicit deployment experiment, recorded in initialization.json.
            # Never relax trained-checkpoint resume or silently resample history.
            allowed.add('control_hz')
    for key, value in protocol.as_dict().items():
        if old.get(key) != value and key not in allowed:
            raise ValueError(f'Checkpoint/protocol mismatch: {key}')
    return old | protocol.as_dict()


def validate_dataset_contract(manifest: dict, protocol: Protocol) -> None:
    for key in ('image_width', 'image_height', 'control_hz'):
        if key == 'control_hz' and protocol.execution_mode == 'deployment' and protocol.control_hz == 0:
            cadence = manifest.get(key)
            if isinstance(cadence, bool) or not isinstance(cadence, (int, float)) or not math.isfinite(cadence) or cadence < 0:
                raise ValueError('Invalid dataset control_hz')
            continue
        if manifest.get(key) != getattr(protocol, key):
            raise ValueError(f'Dataset/protocol mismatch: {key}')
