from __future__ import annotations

import copy

import pytest


def report(pickup=20/33, attack=.75, recovery=.79):
    values = dict(attack=(912, attack), ground_attack=(336, .67),
                  pickup=(33, pickup), recovery=(209, recovery))
    slices = {name: dict(samples=n, accuracy=acc, min_samples=5,
                        min_accuracy=.6, passed=acc >= .6)
              for name, (n, acc) in values.items()}
    failures = [name for name, value in slices.items() if not value['passed']]
    return dict(passed=not failures, failures=failures, slices=slices)


def test_one_pickup_flip_after_update_is_not_forgetting():
    from train.primitive_retention import PrimitiveRetentionGuard
    guard = PrimitiveRetentionGuard()
    assert guard.observe(report(), 0)['action'] == 'continue'
    result = guard.observe(report(pickup=19/33), 512)
    assert result['action'] == 'continue'
    assert result['slices']['pickup']['retention'] == pytest.approx(.95)


def test_initial_bc_qualification_is_not_relaxed():
    from train.primitive_retention import PrimitiveRetentionGuard
    result = PrimitiveRetentionGuard().observe(report(pickup=19/33), 0)
    assert result['action'] == 'stop'
    assert result['reason'] == 'initial_bc_gate_failed'


def test_persistent_drop_stops_after_three_distinct_updates_not_duplicate_logs():
    from train.primitive_retention import PrimitiveRetentionGuard
    guard = PrimitiveRetentionGuard()
    baseline = report()
    guard.observe(baseline, 0)
    baseline['slices']['pickup']['accuracy'] = 0  # Anchor must not alias caller data.
    for step in (512, 512, 512, 1024):
        result = guard.observe(report(pickup=17/33), step)
        assert result['action'] == 'warn'
    result = guard.observe(report(pickup=17/33), 1536)
    assert result['action'] == 'stop'
    assert result['failures'] == ['pickup']
    assert result['slices']['pickup']['consecutive_regressions'] == 3


def test_recovery_resets_streak_and_does_not_move_baseline():
    from train.primitive_retention import PrimitiveRetentionGuard
    guard = PrimitiveRetentionGuard()
    guard.observe(report(), 0)
    guard.observe(report(pickup=17/33), 512)
    guard.observe(report(pickup=17/33), 1024)
    assert guard.observe(report(pickup=18/33), 1536)['action'] == 'continue'
    result = guard.observe(report(pickup=17/33), 2048)
    assert result['action'] == 'warn'
    assert result['slices']['pickup']['baseline_accuracy'] == pytest.approx(20/33)
    assert result['slices']['pickup']['consecutive_regressions'] == 1


def test_severe_collapse_stops_immediately():
    from train.primitive_retention import PrimitiveRetentionGuard
    guard = PrimitiveRetentionGuard()
    guard.observe(report(), 0)
    result = guard.observe(report(attack=.2), 512)
    assert result['action'] == 'stop'
    assert result['failures'] == ['attack']
    assert result['reason'] == 'severe_imitation_regression'


@pytest.mark.parametrize('change', ['nan', 'missing', 'samples', 'backwards'])
def test_corrupt_or_changed_evidence_fails_closed(change):
    from train.primitive_retention import PrimitiveRetentionGuard
    guard = PrimitiveRetentionGuard()
    guard.observe(report(), 512)
    current = copy.deepcopy(report())
    if change == 'nan':
        current['slices']['pickup']['accuracy'] = float('nan')
    elif change == 'missing':
        del current['slices']['pickup']
    elif change == 'samples':
        current['slices']['pickup']['samples'] = 34
    step = 0 if change == 'backwards' else 1024
    assert guard.observe(current, step)['action'] == 'stop'
