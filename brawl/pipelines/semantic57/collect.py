"""Bounded live drills: setup is never a behavior-cloning label."""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from brawl.control.semantic_actions import BY_NAME
from brawl.state.state_spec import StateSpec
from brawl.perception.weapon_evidence import WeaponEvidence, WeaponHoldGate
from brawl.training.teachers import RecoverySetupController
from brawl.pipelines.semantic57.contract import ACTION_DIM, OBSERVATION_DIM
from brawl.pipelines.semantic57.demos import DemoBatch
from brawl.pipelines.semantic57.teacher import IntegratedTeacher, semantic_from_legacy


@dataclass
class Attempt:
    accepted: bool = False
    reason: str = 'step_cap'
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    families: list = field(default_factory=list)
    setup_steps: int = 0
    steps: int = 0
    context_interruptions: int = 0
    pickup_requests: int = 0
    peak_hold_steps: int = 0
    negative_refreshes: int = 0
    execution_status: str = ''
    execution_reason: str = ''
    attack_presses: int = 0
    damage_dealt: float = 0.
    # Source decision indices preserve gaps for sequence/frame-aligned datasets.
    observation_steps: list = field(default_factory=list)


class SemanticControls:
    observation_dim = OBSERVATION_DIM
    pickup_id = BY_NAME['PICKUP'].id
    throw_id = BY_NAME['THROW_NEUTRAL'].id
    require_combat_hit = False
    teacher = IntegratedTeacher
    setup_action = staticmethod(semantic_from_legacy)

    @staticmethod
    def ready(obs):
        return obs[-ACTION_DIM:].sum() > 1

    @staticmethod
    def allowed(obs, action):
        return bool(obs[-ACTION_DIM + action])

    @staticmethod
    def recordable(action, info):
        return True


def release_inputs(env) -> None:
    release = getattr(env.unwrapped, '_release_all_inputs', None)
    if release is not None:
        release()


def _step(env, action):
    obs, _, terminated, truncated, info = env.step(action)
    # Training-room windows are bookkeeping boundaries, not physical resets.
    if terminated or truncated:
        obs, _ = env.reset()
    return obs, info


def collect_attempt(env, drill: str, *, setup_steps=240, max_steps=600,
                    min_steps=4, hold_steps=40,
                    on_event: Callable[[dict], None] | None = None,
                    controls=None, teacher_factory=None) -> Attempt:
    if drill not in ('weapon', 'recovery', 'combat'):
        raise ValueError(f'Unknown drill: {drill}')
    if min(setup_steps, max_steps, min_steps, hold_steps) < 1:
        raise ValueError('Collection budgets must be positive')
    result = Attempt()
    controls = controls or SemanticControls()
    release_inputs(env)
    try:
        obs, _ = env.reset()
        obs, info = _step(env, 0)  # Refresh after release/reset or a console pause.
        setup = RecoverySetupController()
        evidence = WeaponEvidence()
        armed = False
        for step in range(setup_steps):
            free = controls.ready(obs)
            visual = evidence.update(info)
            armed = bool(StateSpec.get(obs, 'player_has_weapon') > .5)
            ready = (drill == 'combat' and free or
                     drill == 'recovery' and free and StateSpec.get(obs, 'player_is_offstage') > .5 or
                     drill == 'weapon' and free and not armed and visual is False)
            if ready:
                break
            action = 0
            if free and drill == 'recovery':
                action = controls.setup_action(setup.action(obs), obs)
            elif free and drill == 'weapon' and armed and step % 12 == 0:
                action = controls.throw_id
            if not controls.allowed(obs, action):
                action = 0
            obs, info = _step(env, action)
            result.setup_steps += 1
        else:
            result.reason = 'setup_timeout'
            return result

        teacher = (teacher_factory or controls.teacher)()
        gate = WeaponHoldGate(required_steps=hold_steps)
        landed = 0
        pending_label: int | None = None
        for step in range(max_steps):
            decision = teacher.action(obs)
            # Forced executor continuations have no action choice and therefore
            # no BC signal. They remain in the optional dynamics replay.
            record = controls.ready(obs)
            previous = obs.copy()
            obs, info = _step(env, decision.action)
            record = bool(record and controls.recordable(decision.action, info))
            result.steps += 1
            result.attack_presses += int(bool({'num4', 'num6'} & set(info.get('input_taps_pressed', ()))))
            result.damage_dealt += max(0., float(info.get('op_delta_damage', 0.)))
            status = result.execution_status = str(info.get('semantic_execution_status', ''))
            reason = result.execution_reason = str(info.get('semantic_execution_reason', ''))

            def emit(event):
                if on_event is not None:
                    on_event(dict(event=event, step=result.steps, action=decision.action,
                        execution_status=status, execution_reason=reason,
                        active_action=info.get('semantic_active_action'),
                        identity_before=StateSpec.get(previous, 'identity_observed'),
                        identity_after=StateSpec.get(obs, 'identity_observed'),
                        grounded_before=StateSpec.get(previous, 'player_grounded'),
                        grounded_after=StateSpec.get(obs, 'player_grounded'),
                        has_weapon=StateSpec.get(obs, 'player_has_weapon'),
                        pickup_requests=result.pickup_requests, hold_steps=gate.hold_steps,
                        negative_refreshes=gate.negative_refreshes,
                        **{key: float(info.get(key, 0.)) for key in (
                            'player_weapon_visual_fresh', 'player_weapon_visual_valid',
                            'player_weapon_visual_armed', 'player_weapon_visual_confidence')}))

            interrupted = status == 'cancelled' and reason in (
                'context_changed', 'context_changed_before_min_hold')
            if interrupted:
                # Only the uncompleted request is an invalid label. Earlier
                # trusted decisions do not become wrong when identity disappears.
                if pending_label is not None:
                    del result.observations[pending_label:]
                    del result.actions[pending_label:]
                    del result.families[pending_label:]
                    del result.observation_steps[pending_label:]
                pending_label = None
                record = False
                result.context_interruptions += 1
                emit('execution_interrupted')
            elif status in ('rejected', 'cancelled', 'aborted'):
                emit('execution_failed')
                result.reason = 'input_execution_rejected'
                return result
            elif reason:
                emit('execution_cleanup')
            if record:
                result.observations.append(previous)
                result.actions.append(decision.action)
                result.families.append(decision.family)
                result.observation_steps.append(result.steps)
                if status == 'started':
                    pending_label = len(result.actions) - 1
            if status in ('completed', 'held'):
                pending_label = None
            if float(info.get('self_stock_lost_step', 0.)) > .5:
                result.reason = 'stock_lost'
                return result
            success = False
            if drill == 'weapon':
                pickup_pressed = (decision.action == controls.pickup_id
                                  and 'num5' in info.get('input_taps_pressed', ()))
                if pickup_pressed:
                    result.pickup_requests += 1
                    emit('pickup_pressed')
                elif result.pickup_requests:
                    # This step's capture predates its input. Never use that
                    # same capture to confirm the pickup we have just pressed.
                    gate.update(StateSpec.get(obs, 'player_has_weapon') > .5, info)
                result.peak_hold_steps = max(result.peak_hold_steps, gate.hold_steps)
                result.negative_refreshes = gate.negative_refreshes
                if float(info.get('player_weapon_visual_fresh', 0.)) > .5:
                    emit('weapon_evidence')
                if gate.dropped:
                    result.reason = 'weapon_dropped_after_pickup'
                    return result
                success = gate.success
            elif drill == 'recovery':
                onstage = StateSpec.get(obs, 'player_is_offstage') < .5
                grounded = StateSpec.get(obs, 'player_grounded') > .5
                landed = landed + 1 if onstage and grounded and controls.ready(obs) else 0
                success = landed >= 2
            elif step == max_steps - 1:
                success = 'combat' in result.families
                if controls.require_combat_hit:
                    success = success and result.attack_presses > 0 and result.damage_dealt > 0
                    if not success:
                        result.reason = 'no_confirmed_combat'
            if success:
                result.accepted = len(result.actions) >= min_steps
                result.reason = 'accepted' if result.accepted else 'trivial'
                return result
        if drill == 'weapon' and result.pickup_requests and not gate.pickup_seen:
            result.reason = 'pickup_unconfirmed'
        return result
    finally:
        release_inputs(env)


def _collect_demonstrations(env, *, episodes=20, attempts_factor=4, on_event=None, controls=None, **kwargs):
    if episodes < 2 or attempts_factor < 1:
        raise ValueError('At least two episodes per drill and a positive attempt factor are required')
    controls = controls or SemanticControls()
    observations, actions, families, episode_ids, drills = [], [], [], [], []
    summary = {'drills': {}, 'complete': True, 'interrupted': False,
               'episodes_attempted': 0, 'episodes_collected': 0}
    for drill in ('weapon', 'recovery', 'combat'):
        accepted = 0
        reasons = Counter()
        for attempt in range(episodes * attempts_factor):
            summary['episodes_attempted'] += 1
            def emit(event):
                if on_event is not None:
                    on_event(dict(drill=drill, attempt=attempt + 1, **event))
            try:
                result = collect_attempt(env, drill, on_event=emit, controls=controls, **kwargs)
            except KeyboardInterrupt:
                summary['interrupted'] = True
                reasons['interrupted'] += 1
                break
            reasons[result.reason] += 1
            if result.accepted:
                episode = summary['episodes_collected']
                summary['episodes_collected'] += 1
                observations.extend(result.observations)
                actions.extend(result.actions)
                families.extend(result.families)
                episode_ids.extend([episode] * len(result.actions))
                drills.extend([drill] * len(result.actions))
                accepted += 1
            print(f'[{drill}] attempt={attempt + 1}/{episodes * attempts_factor} '
                  f'accepted={accepted}/{episodes} labels={len(result.actions)} '
                  f'setup={result.setup_steps} steps={result.steps} reason={result.reason} '
                  f'exec={result.execution_status}/{result.execution_reason or "-"} '
                  f'interruptions={result.context_interruptions} pickup={result.pickup_requests} '
                  f'hold_peak={result.peak_hold_steps} neg={result.negative_refreshes}', flush=True)
            emit(dict(event='attempt_end', accepted=result.accepted, reason=result.reason,
                      steps=result.steps, labels=len(result.actions),
                      context_interruptions=result.context_interruptions,
                      pickup_requests=result.pickup_requests, peak_hold_steps=result.peak_hold_steps,
                      negative_refreshes=result.negative_refreshes,
                      attack_presses=result.attack_presses, damage_dealt=result.damage_dealt,
                      execution_status=result.execution_status, execution_reason=result.execution_reason))
            if accepted >= episodes:
                break
        summary['drills'][drill] = dict(accepted=accepted, reasons=dict(reasons))
        summary['complete'] &= accepted == episodes
        if summary['interrupted']:
            summary['complete'] = False
            break
    summary['family_counts'] = dict(Counter(families))
    summary['action_counts'] = dict(Counter(map(str, actions)))
    batch = DemoBatch(np.asarray(observations, np.float32).reshape(-1, controls.observation_dim),
                      np.asarray(actions, np.int64), np.asarray(episode_ids, np.int64),
                      np.asarray(families, dtype='U16'), np.asarray(drills, dtype='U16'))
    return batch, summary


def collect_demonstrations(env, *, events_path: Path | None = None, **kwargs):
    if events_path is None:
        return _collect_demonstrations(env, **kwargs)
    # Event-only diagnostics, not video or a second full observation archive.
    with events_path.open('x', encoding='utf-8') as stream:
        def emit(event):
            stream.write(json.dumps(event) + '\n')
            if event['event'] == 'attempt_end':
                stream.flush()
        return _collect_demonstrations(env, on_event=emit, **kwargs)
