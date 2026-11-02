"""Reuse qualified primitive teachers while preserving unlabeled context."""
from __future__ import annotations

import json
import numpy as np

from brawl.pipelines.primitive27.training import PrimitiveControls
from brawl.pipelines.semantic57.collect import collect_attempt
from brawl.pipelines.round_a.data import EpisodeWriter
from brawl.training.run_utils import write_json


def collect(adapter, protocol, directory, *, episodes=30, attempts_factor=4, setup_steps=240, max_steps=600):
    if episodes < 5:
        raise ValueError('At least five episodes per drill are needed for three disjoint splits')
    writer = EpisodeWriter(directory / 'demos', width=protocol.image_width,
        height=protocol.image_height, extra={'control_hz': protocol.control_hz})
    summary = dict(complete=False, drills={})
    adapter.reset()
    try:
        for drill in ('weapon', 'recovery', 'combat'):
            accepted, attempts = 0, []
            for attempt in range(episodes * attempts_factor):
                adapter.begin_recording()
                result = collect_attempt(adapter, drill, controls=PrimitiveControls(),
                                         setup_steps=setup_steps, max_steps=max_steps)
                adapter.recording = False
                attempts.append(dict(accepted=result.accepted, reason=result.reason,
                                     labels=len(result.actions), context=len(adapter.records)))
                if result.accepted:
                    rows = adapter.records
                    indices = np.asarray(result.observation_steps) + result.setup_steps
                    valid = np.zeros(len(rows), bool)
                    valid[indices] = True
                    families = np.full(len(rows), 'context', dtype='U16')
                    families[indices] = result.families
                    obs = np.stack([row[0] for row in rows])
                    actions = np.asarray([row[1] for row in rows], np.int64)
                    # Do not use approximate image matching to recover alignment.
                    if not np.array_equal(obs[indices], result.observations) or not np.array_equal(actions[indices], result.actions):
                        raise RuntimeError('Collector/source alignment mismatch')
                    writer.append(obs=obs, actions=actions, valid=valid, families=families, drill=drill,
                        frames=np.asarray(adapter.frames, np.uint8),
                        frame_refs=np.asarray([row[2] for row in rows], np.int64),
                        mirrored=np.asarray([row[3] for row in rows], bool))
                    accepted += 1
                summary['drills'][drill] = dict(accepted=accepted, attempts=attempts)
                print(f'[{drill}] {accepted}/{episodes} accepted; {result.reason}', flush=True)
                if accepted == episodes:
                    break
            if accepted != episodes:
                raise RuntimeError(f'{drill} quotas not met; partial dataset saved, do not pretrain')
        summary['complete'] = True
    finally:
        adapter.recording = False
        writer.manifest['complete'] = summary['complete']
        writer.finish()
        write_json(directory / 'collection_summary.json', summary)


def cadence_assessment(intervals, protocol):
    values = np.asarray(intervals, dtype=float)
    valid = bool(values.size and np.isfinite(values).all() and (values > 0).all())
    overrun = float(np.mean(values > 1.1 / protocol.control_hz)) if valid and protocol.control_hz else None
    passed = bool(valid and (not protocol.control_hz or overrun <= .05))
    return dict(execution_mode=protocol.execution_mode, timing_valid=valid,
                decision_interval_p95_s=float(np.quantile(values, .95)) if valid else None,
                measured_hz=float(1 / values.mean()) if valid else None,
                overrun_fraction=overrun, cadence_passed=passed,
                cadence_acceptable=bool(valid and (passed or protocol.execution_mode == 'deployment')))


def qualify(model, adapter, protocol, directory, *, attempts=3):
    import gymnasium as gym
    from brawl.pipelines.round_a.models import Predictor
    from brawl.pipelines.primitive27.training import PrimitiveTeacher
    from brawl.pipelines.semantic57.teacher import TeacherDecision
    boundary = [True]

    class StructuredView(gym.Wrapper):
        def reset(self, **kwargs):
            boundary[0] = True
            _, info = self.env.reset(**kwargs)
            return adapter.current['structured'].copy(), info
        def step(self, action):
            _, r, done, truncated, info = self.env.step(action)
            return adapter.current['structured'].copy(), r, done, truncated, info

    class Controller:
        def __init__(self):
            self.predictor = Predictor(model)
            self.teacher = PrimitiveTeacher()
        def action(self, obs):
            action = self.predictor.predict(adapter.policy_obs(), start=boundary[0])
            boundary[0] = False
            return TeacherDecision(action, self.teacher.action(obs).family)

    report = dict(passed=True, drills={})
    adapter.intervals.clear()
    for drill in ('weapon', 'recovery', 'combat'):
        rows = []
        for _ in range(attempts):
            result = collect_attempt(StructuredView(adapter), drill, controls=PrimitiveControls(), teacher_factory=Controller)
            rows.append(dict(accepted=result.accepted, reason=result.reason, labels=len(result.actions),
                             pickup_requests=result.pickup_requests, peak_hold_steps=result.peak_hold_steps,
                             negative_refreshes=result.negative_refreshes))
            if result.accepted:
                break
        report['drills'][drill] = rows
        report['passed'] &= any(row['accepted'] for row in rows)
    report['skills_passed'] = bool(report['passed'])
    report.update(cadence_assessment(adapter.intervals, protocol))
    report['passed'] &= report['cadence_acceptable']
    if not report['cadence_passed'] and report['cadence_acceptable']:
        print(f'WARNING: cadence target missed (p95={report["decision_interval_p95_s"]*1000:.1f} ms, '
              f'mean={report["measured_hz"]:.1f} Hz). Deployment protocol records this cost; '
              'it is not a fixed-cadence algorithm comparison.', flush=True)
    write_json(directory / 'qualification.json', report)
    return report
