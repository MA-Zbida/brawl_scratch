"""Versioned action intent and physical sequencing, without a game or GPU."""
from __future__ import annotations

from dataclasses import replace

import pytest

from control.semantic_actions import ACTIONS, BY_NAME, ControlContext, action_mask, contract, require_contract
from control.semantic_executor import ExecutionTiming, SemanticExecutor


class RecordingInput:
    def __init__(self):
        self.held = set()
        self.taps = set()
        self.events = []

    def set_pressed(self, keys):
        self.held = set(keys)
        self.events.append(("held", set(keys)))

    def key_down(self, key):
        self.taps.add(key)
        self.events.append(("down", key))

    def key_up(self, key):
        self.taps.discard(key)
        self.events.append(("up", key))

    def reset(self):
        self.held.clear()
        self.taps.clear()


def test_registry_distinguishes_ground_heavy_recovery_and_gc():
    from control.semantic_actions import _registry
    assert _registry() == ACTIONS
    assert len(ACTIONS) == 57
    assert len(ACTIONS) == len(BY_NAME)
    assert [a.id for a in ACTIONS] == list(range(len(ACTIONS)))
    names = ["SIGNATURE_TOWARD", "RECOVERY_TOWARD", "GC_SIGNATURE_TOWARD"]
    assert len({BY_NAME[n].id for n in names}) == 3


def test_context_masks_recovery_not_only_when_offstage():
    ground = action_mask(ControlContext(grounded=True, armed=True))
    air = action_mask(ControlContext(grounded=False, armed=True))
    assert not ground[BY_NAME["RECOVERY_TOWARD"].id]
    assert air[BY_NAME["RECOVERY_TOWARD"].id]
    assert ground[BY_NAME["SIGNATURE_TOWARD"].id]
    assert not air[BY_NAME["SIGNATURE_TOWARD"].id]
    assert air[BY_NAME["GC_SIGNATURE_TOWARD"].id]


def test_pickup_throw_and_uncertain_resource_masks():
    unarmed = action_mask(ControlContext(grounded=True, armed=False))
    assert unarmed[BY_NAME["PICKUP"].id]
    assert not unarmed[BY_NAME["THROW_TOWARD"].id]
    unknown = action_mask(ControlContext())
    assert unknown[BY_NAME["PICKUP"].id]
    assert unknown[BY_NAME["THROW_TOWARD"].id]
    assert not any(action_mask(ControlContext(controllable=False))[1:])


def test_contract_rejects_legacy_and_reordered_same_dimension():
    require_contract(contract())
    with pytest.raises(ValueError, match="action contract"):
        require_contract({"action_dim": 27})
    altered = contract()
    altered["actions"] = list(reversed(altered["actions"]))
    with pytest.raises(ValueError, match="action contract"):
        require_contract(altered)


def test_recovery_after_dodge_is_rejected_until_guard_expires():
    keys = RecordingInput()
    executor = SemanticExecutor(keys)
    air = ControlContext(grounded=False)
    assert executor.submit("AIR_DODGE_TOWARD", air, now=0).status == "started"
    executor.advance(air, now=.03)
    executor.advance(air, now=.06)
    assert executor.submit("RECOVERY_TOWARD", air, now=.07).reason == "post_dodge_guard"
    assert "num6" not in keys.taps
    assert executor.submit("RECOVERY_TOWARD", air, now=.3).status == "started"
    assert "num6" in keys.taps


def test_gravity_cancel_is_sequential_and_locks_physical_direction():
    keys = RecordingInput()
    executor = SemanticExecutor(keys)
    air = ControlContext(grounded=False, mirrored=True)
    executor.submit("GC_SIGNATURE_TOWARD", air, now=0)
    assert keys.taps == {"e"}
    assert keys.held == set()  # The first input must be a neutral dodge.
    executor.advance(replace(air, mirrored=False), now=.04)
    assert keys.taps == {"num6"}
    assert keys.held == {"a"}  # Crossing sides cannot change an in-flight sequence.
    executor.advance(air, now=.07)
    executor.advance(air, now=.1)
    assert not executor.busy
    assert not keys.taps


def test_late_gc_attack_is_cancelled_not_emitted_as_wrong_move():
    keys = RecordingInput()
    executor = SemanticExecutor(keys)
    air = ControlContext(grounded=False)
    executor.submit("GC_LIGHT_DOWN", air, now=0)
    result = executor.advance(air, now=.5)
    assert result.status == "cancelled"
    assert result.reason == "sequence_deadline"
    assert ("down", "num4") not in keys.events
    assert not keys.taps


def test_landing_or_missing_identity_cancels_air_sequence():
    for context in [ControlContext(grounded=True), ControlContext(controllable=False)]:
        keys = RecordingInput()
        executor = SemanticExecutor(keys)
        executor.submit("GC_LIGHT_DOWN", ControlContext(grounded=False), now=0)
        assert executor.advance(context, now=.04).status == "cancelled"
        assert not keys.taps


@pytest.mark.parametrize('context', [ControlContext(grounded=True), ControlContext(controllable=False)])
def test_released_input_does_not_require_identity_or_old_ground_context(context):
    keys = RecordingInput()
    executor = SemanticExecutor(keys)
    executor.submit('AIR_LIGHT_TOWARD', ControlContext(grounded=False), now=0.)
    assert executor.advance(ControlContext(grounded=False), now=.025).status == 'released'
    assert executor.advance(context, now=.05).status == 'completed'
    assert not executor.busy and not keys.held and not keys.taps


@pytest.mark.parametrize('context', [ControlContext(grounded=True), ControlContext(controllable=False)])
def test_already_injected_input_finishes_when_context_changes_after_minimum_hold(context):
    keys = RecordingInput()
    executor = SemanticExecutor(keys)
    executor.submit('AIR_LIGHT_TOWARD', ControlContext(grounded=False), now=0.)
    result = executor.advance(context, now=.025)
    assert result.status == 'completed'
    assert result.reason == 'context_changed_after_input'
    assert not keys.held and not keys.taps


def test_early_identity_loss_still_cancels_and_releases_input():
    keys = RecordingInput()
    executor = SemanticExecutor(keys)
    executor.submit('AIR_LIGHT_TOWARD', ControlContext(grounded=False), now=0.)
    result = executor.advance(ControlContext(controllable=False), now=.005)
    assert result.status == 'cancelled'
    assert result.reason == 'context_changed_before_min_hold'
    assert not keys.held and not keys.taps


def test_busy_executor_rejects_instead_of_queuing_obsolete_actions():
    keys = RecordingInput()
    executor = SemanticExecutor(keys)
    air = ControlContext(grounded=False)
    executor.submit("RECOVERY_TOWARD", air, now=0)
    assert executor.submit("PICKUP", air, now=.001).reason == "busy"
    executor.advance(air, now=.03)
    executor.advance(air, now=.06)
    assert ("down", "num5") not in keys.events


def test_context_manager_releases_every_key_on_exception():
    keys = RecordingInput()
    with pytest.raises(RuntimeError):
        with SemanticExecutor(keys) as executor:
            executor.submit("RECOVERY_TOWARD", ControlContext(grounded=False), now=0)
            raise RuntimeError("interrupted")
    assert not keys.held and not keys.taps


def test_time_must_be_finite_and_monotonic():
    executor = SemanticExecutor(RecordingInput())
    ctx = ControlContext()
    executor.advance(ctx, now=10)
    for now in [9, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            executor.advance(ctx, now=now)


@pytest.mark.parametrize("action", ACTIONS, ids=lambda action: action.name)
def test_every_registered_sequence_completes_and_releases_keys(action):
    keys = RecordingInput()
    executor = SemanticExecutor(keys)
    ctx = ControlContext(grounded=action.context != "air", armed=action.possession)
    assert executor.submit(action.name, ctx, now=0).status == "started"
    for tick in range(1, 30):
        executor.advance(ctx, now=tick * .01)
    assert not executor.busy
    assert not keys.taps and not keys.held


@pytest.mark.parametrize("kwargs", [{"level_s": 0}, {"gc_delay_s": float("nan")},
                                   {"gc_delay_s": .001}, {"gc_deadline_s": .9}])
def test_invalid_timing_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ExecutionTiming(**kwargs)


def test_masked_submission_never_injects():
    keys = RecordingInput()
    executor = SemanticExecutor(keys)
    assert executor.submit("RECOVERY_TOWARD", ControlContext(grounded=True), now=0).reason == "context_mask"
    assert not keys.events


def test_sink_error_cleans_up_and_propagates():
    class BrokenInput(RecordingInput):
        def key_down(self, key):
            super().key_down(key)
            if key == "num6":
                raise OSError("injection failed")

    for name in ["RECOVERY_TOWARD", "GC_SIGNATURE_TOWARD"]:
        keys = BrokenInput()
        executor = SemanticExecutor(keys)
        ctx = ControlContext(grounded=False)
        with pytest.raises(OSError, match="injection failed"):
            executor.submit(name, ctx, now=0)
            executor.advance(ctx, now=.04)
        assert not executor.busy and not keys.taps and not keys.held
