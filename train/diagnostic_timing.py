"""Opt-in host-wall tracing. No CUDA synchronization or per-event disk writes."""
from __future__ import annotations

from array import array
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from train.run_utils import write_json


class TraceProbe:
    """Keep early stalls as well as late ones; report bounded-trace overflow.

    Exclusive time subtracts instrumented children, not unobserved GPU work.
    CUDA launches may complete later, so these are host latencies, not kernel times.
    """

    def __init__(self, *, clock: Callable = time.perf_counter, max_events: int = 1_000_000):
        if max_events < 1:
            raise ValueError('max_events must be positive')
        self.clock, self.max_events = clock, max_events
        self.origin = clock()
        self.decision = -1
        self.stages: dict[str, dict] = {}
        self.events = {key: array('d') for key in ('stage', 'decision', 'started_s', 'elapsed_s', 'self_s')}
        self.stack: list[list[float]] = []
        self.originals: list = []
        self.missing: list[str] = []
        self.dropped = 0

    @contextmanager
    def measure(self, label: str):
        row = self.stages.setdefault(label, dict(id=len(self.stages), calls=0, total_s=0.,
                                                 self_total_s=0., max_ms=0., values=array('d')))
        frame = [self.clock(), 0.]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = self.clock() - frame[0]
            self.stack.pop()
            own = max(0., elapsed - frame[1])
            if self.stack:
                self.stack[-1][1] += elapsed
            row['calls'] += 1
            row['total_s'] += elapsed
            row['self_total_s'] += own
            row['max_ms'] = max(row['max_ms'], elapsed * 1000)
            if len(self.events['stage']) < self.max_events:
                row['values'].append(elapsed * 1000)
                for key, value in zip(self.events, (row['id'], self.decision,
                                                     frame[0] - self.origin, elapsed, own)):
                    self.events[key].append(value)
            else:
                self.dropped += 1

    def wrap(self, label: str, target: Any, name: str) -> None:
        original = getattr(target, name, None)
        if not callable(original):
            self.missing.append(label)
            return
        owned = name in vars(target)

        @wraps(original)
        def timed(*args, **kwargs):
            with self.measure(label):
                return original(*args, **kwargs)

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
        self.originals.clear()
        return False

    def summary(self) -> dict:
        rows = {}
        for name, row in self.stages.items():
            values = np.asarray(row['values'])
            rows[name] = {key: value for key, value in row.items() if key != 'values'}
            rows[name].update(mean_ms=row['total_s'] * 1000 / row['calls'], samples=len(values))
            rows[name].update({f'p{q}_ms': float(np.percentile(values, q)) if len(values) else None
                               for q in (50, 95, 99)})
        return dict(stages=rows, events_dropped=self.dropped, missing_stages=self.missing,
                    interpretation='Inclusive host wall times overlap; do not sum. Self times subtract '
                    'instrumented children only. No added CUDA synchronization. Quantiles cover all '
                    'retained events (prefix if overflow); totals/max cover all calls. Trace storage '
                    'and profiler overhead are not eliminated. No capture-freshness guarantee.')

    def save(self, directory: Path) -> None:
        write_json(directory / 'timing.json', self.summary())
        np.savez_compressed(directory / 'timing_trace.npz',
                            **{key: np.asarray(value) for key, value in self.events.items()})


def attach_live_probe(probe: TraceProbe, adapter: Any, match: Any, writer: Any = None) -> None:
    """Attach AFTER warm-up: reset can replace Memory; YOLO lazily builds its predictor."""
    base = adapter.unwrapped
    extractor = getattr(base, 'extractor', None)
    predictor = getattr(getattr(extractor, 'yolo', None), 'predictor', None)
    for label, target, method in (
        ('adapter_step', adapter, 'step'),
        ('image_history', adapter, 'capture'),
        ('policy_observation_copy', adapter, 'policy_obs'),
        ('pacing_sleep', adapter, 'sleep'),
        ('match_step', match, 'step'),
        ('csv_write_and_flush', match, '_write'),
        ('base_step', base, 'step'),
        ('capture', getattr(base, 'frame_provider', None), 'get_frame'),
        ('input_execution', base, '_apply_action'),
        ('detection', base, '_get_detections'),
        ('detector_preprocess', predictor, 'preprocess'),
        ('detector_inference', predictor, 'inference'),
        ('detector_postprocess', predictor, 'postprocess'),
        ('detector_output_to_cpu', extractor, '_results_to_detections'),
        ('state_estimation', getattr(base, 'memory', None), 'update_from_detections'),
        ('weapon_classification_and_fusion', base, '_update_weapon_classification'),
        ('game_logic_and_hud', base, '_update_game_logic'),
        ('structured_observation', base, '_get_obs'),
        ('reward', match.objective, 'step'),
    ):
        probe.wrap(label, target, method)
    if writer is not None:
        probe.wrap('replay_append', writer, 'append')
        probe.wrap('replay_shard_flush', writer, 'flush')
        probe.wrap('replay_manifest', writer, '_write_manifest')
