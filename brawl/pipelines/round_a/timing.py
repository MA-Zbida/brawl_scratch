"""Opt-in wall-time attribution; never synchronizes CUDA or changes cadence."""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import time
from typing import Callable

import numpy as np

from brawl.training.run_utils import write_json


class TimingProbe:
    """Bounded samples with original methods restored even on qualification failure."""

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.perf_counter,
                 max_samples: int = 8192):
        if max_samples < 1:
            raise ValueError('max_samples must be positive')
        self.path, self.clock, self.max_samples = path, clock, max_samples
        self.stages: dict = {}
        self.originals: list = []

    def wrap(self, label: str, target, name: str) -> None:
        original = getattr(target, name, None)
        if not callable(original):
            return
        owned = name in vars(target)
        row = self.stages.setdefault(label, dict(calls=0, total_s=0., samples=deque(maxlen=self.max_samples)))

        @wraps(original)
        def timed(*args, **kwargs):
            began = self.clock()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = self.clock() - began
                row['calls'] += 1
                row['total_s'] += elapsed
                row['samples'].append(elapsed)

        setattr(target, name, timed)
        self.originals.append((target, name, original, owned))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for target, name, original, owned in reversed(self.originals):
            if owned:
                setattr(target, name, original)
            else:
                delattr(target, name)
        rows = {}
        for name, row in self.stages.items():
            values = np.asarray(row['samples']) * 1000
            rows[name] = dict(calls=row['calls'], samples=len(values),
                total_s=row['total_s'], mean_ms=1000 * row['total_s'] / max(row['calls'], 1),
                p50_ms=float(np.median(values)) if len(values) else None,
                p95_ms=float(np.quantile(values, .95)) if len(values) else None,
                max_ms=float(np.max(values)) if len(values) else None)
        write_json(self.path, dict(schema_version=1, stages=rows,
            interpretation='Inclusive host wall times: nested stages overlap; do not sum them. '
                           'No CUDA synchronization is added. Initialization is included. '
                           'Percentiles use the most recent bounded samples; totals cover all calls.',
            max_samples=self.max_samples))
        return False


@contextmanager
def qualification_timing(model, adapter, path: Path):
    """Attach only during qualification, with no steady-state training overhead."""
    with TimingProbe(path) as probe:
        base = adapter.unwrapped
        for label, target, method in (
            ('policy_predict', model, 'predict'),
            ('image_history', adapter, 'capture'),
            ('observation_copy', adapter, 'policy_obs'),
            ('pacing_sleep', adapter, 'sleep'),
            ('base_step', base, 'step'),
            ('capture', getattr(base, 'frame_provider', None), 'get_frame'),
            ('detection', base, '_get_detections'),
            ('input_execution', base, '_apply_action'),
            ('state_estimation', getattr(base, 'memory', None), 'update_from_detections'),
            ('weapon_classification_and_fusion', base, '_update_weapon_classification'),
            ('game_logic_and_hud', base, '_update_game_logic'),
            ('structured_observation', base, '_get_obs'),
            ('reward_memory', getattr(base, 'reward_provider', None), 'update_memory'),
        ):
            probe.wrap(label, target, method)
        wrapper = adapter.env
        while wrapper is not base:
            if type(wrapper).__name__ == 'MatchTrainingEnv':
                probe.wrap('match_step', wrapper, 'step')
                probe.wrap('csv_write', wrapper, '_write')
            recorder = vars(wrapper).get('recorder')
            if recorder is not None:
                probe.wrap('replay_append', recorder.writer, 'append')
            wrapper = wrapper.env
        yield probe
