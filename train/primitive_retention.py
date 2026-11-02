"""Monitor imitation drift without treating every argmax flip as skill loss."""
from __future__ import annotations

from copy import deepcopy
import math


class PrimitiveRetentionGuard:
    """Engineering guardrail, not a statistical test or live retention score.

    Admission still requires the original BC gate. Afterwards, compare against
    that fixed starting policy, not the admission threshold or a moving anchor.
    A 10% relative loss must persist for three distinct updates; a loss greater
    than half the starting agreement stops immediately. These tolerances permit
    small action-choice changes while bounding unnoticed imitation collapse.
    """

    SLICE_NAMES = ('attack', 'ground_attack', 'pickup', 'recovery')
    MIN_RETENTION = .90
    COLLAPSE_RETENTION = .50
    PATIENCE = 3

    def __init__(self) -> None:
        self.baseline: dict | None = None
        self.last_step: int | None = None
        self.last_result: dict | None = None
        self.streaks = dict.fromkeys(self.SLICE_NAMES, 0)

    def observe(self, report: dict, step: int) -> dict:
        result = dict(action='continue', reason='within_retention_budget', failures=[], slices={},
                      min_retention=self.MIN_RETENTION, patience=self.PATIENCE,
                      collapse_retention=self.COLLAPSE_RETENTION,
                      interpretation='Baseline-relative imitation; not measured live skill retention')

        def stop(reason: str) -> dict:
            result.update(action='stop', reason=reason)
            return result

        try:
            values = report['slices']
            current = {name: dict(samples=int(values[name]['samples']),
                                  accuracy=float(values[name]['accuracy']))
                       for name in self.SLICE_NAMES}
            for name, value in current.items():
                if (value['samples'] <= 0 or value['samples'] != values[name]['samples']
                        or not math.isfinite(value['accuracy'])
                        or not 0 <= value['accuracy'] <= 1):
                    return stop('invalid_imitation_evidence')
            if step < 0 or (self.last_step is not None and step < self.last_step):
                return stop('nonmonotonic_training_step')
        except (KeyError, TypeError, ValueError, OverflowError):
            return stop('invalid_imitation_evidence')

        if self.baseline is None:
            if not report.get('passed') or any(v['accuracy'] <= 0 for v in current.values()):
                return stop('initial_bc_gate_failed')
            self.baseline = deepcopy(current)
            result['reason'] = 'baseline_established'
        elif any(current[name]['samples'] != self.baseline[name]['samples'] for name in self.SLICE_NAMES):
            return stop('changed_imitation_evidence')

        # The final update is logged both before and after live evaluation. It
        # remains one update, not two independent observations of a regression.
        if step == self.last_step:
            return deepcopy(self.last_result)

        severe = []
        persistent = []
        for name, value in current.items():
            anchor = self.baseline[name]['accuracy']
            retention = value['accuracy'] / anchor
            regressed = retention < self.MIN_RETENTION - 1e-12
            self.streaks[name] = self.streaks[name] + 1 if regressed else 0
            result['slices'][name] = dict(**value, baseline_accuracy=anchor, retention=retention,
                                         consecutive_regressions=self.streaks[name])
            if retention < self.COLLAPSE_RETENTION:
                severe.append(name)
            if self.streaks[name] >= self.PATIENCE:
                persistent.append(name)
        if severe or persistent:
            result.update(action='stop', failures=sorted(set(severe + persistent)),
                          reason='severe_imitation_regression' if severe else 'persistent_imitation_regression')
        elif any(self.streaks.values()):
            result.update(action='warn', reason='imitation_regression_pending_confirmation')
        self.last_step, self.last_result = step, deepcopy(result)
        return result
