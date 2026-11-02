"""Frozen discrete27 warm start: explicit data contract and behavioural gates.

No live capture or torch imports here. The experimental semantic57 interface is
not reinterpreted: only its array container and collection bookkeeping are reused.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from action_space import ACTION_DIM, Action, COMPONENT_TABLE, to_keys
from control.taps import TapSequencer
from feature_extractor.memory.state_spec import StateSpec
from train.curriculum_goals import GOAL_INDEX, default_goal_target
from train.heuristic_teachers import HeuristicTeacher
from train.semantic_demos import DemoBatch
from train.semantic_teacher import TeacherDecision

HISTORY_OFFSETS = (2, 4, 8)
OBSERVATION_DIM = StateSpec.observation_dim(HISTORY_OFFSETS)
DRILLS = ('weapon', 'recovery', 'combat')
VERSION = 'primitive27_fighting_v1'


def metadata():
    return dict(schema_version=1, policy_type='final_agent', goal_conditioning=False,
                action_dim=ACTION_DIM, observation_dim=OBSERVATION_DIM,
                primitive_contract=VERSION, actions=[a.name for a in Action],
                components=COMPONENT_TABLE.tolist(), history_offsets=list(HISTORY_OFFSETS),
                features=StateSpec.observation_names(HISTORY_OFFSETS),
                input_contract=TapSequencer.VERSION, tap_min_level_s=TapSequencer.MIN_LEVEL_S,
                algorithm='ppo_cpu_primitive27', action_repeat_steps=1)


def validate_metadata(payload):
    for key, value in metadata().items():
        if payload.get(key) != value:
            raise ValueError(f'Incompatible primitive27 contract: {key}')


class PrimitiveTeacher:
    def __init__(self):
        self.teacher = HeuristicTeacher()

    def action(self, obs):
        value = lambda key: StateSpec.get(obs, key)
        if not PrimitiveControls.ready(obs):
            return TeacherDecision(int(Action.NOOP), 'unobserved')
        goal = default_goal_target()
        if value('player_is_offstage') > .5:
            phase, family = 'recovery_mastery', 'recovery'
        elif value('player_has_weapon') < .5 and value('weapon_on_ground') > .5:
            phase, family = 'weapon_acquisition', 'weapon'
        elif value('opponent_exists') < .5 or value('opponent_is_offstage') > .5:
            phase, family = 'movement_fluency', 'movement'
            goal[GOAL_INDEX['player_x']] = np.clip(value('player_x') + value('signed_dx_to_stage_center'), 0, 1)
            goal[GOAL_INDEX['player_y']] = value('player_y')
        else:
            # Proximity alone is an attack opportunity, not a reason to flee.
            # The combat teacher already disengages when its threat cues fire.
            phase, family = 'combat_execution', 'combat'
        action = self.teacher.action(phase, obs, {'goal_target': goal})
        return TeacherDecision(int(action), family)


class PrimitiveControls:
    observation_dim = OBSERVATION_DIM
    pickup_id = throw_id = int(Action.PICKUP)
    require_combat_hit = True
    teacher = PrimitiveTeacher

    @staticmethod
    def ready(obs):
        return bool(StateSpec.get(obs, 'identity_observed') > .5
                    and StateSpec.get(obs, 'player_missing_frames') == 0)

    @staticmethod
    def setup_action(action, obs):
        return int(action)

    @staticmethod
    def allowed(obs, action):
        return 0 <= action < ACTION_DIM and (action == 0 or PrimitiveControls.ready(obs))

    @staticmethod
    def recordable(action, info):
        if info.get('input_execution_blocked') or info.get('effective_action', action) != action:
            return False
        _, taps = to_keys(int(action), mirrored=False)
        return taps.issubset(set(info.get('input_taps_pressed', ())))


def validate_rows(data):
    n = len(data.actions)
    if data.obs.shape != (n, OBSERVATION_DIM) or data.actions.shape != (n,):
        raise ValueError('Primitive demo shape mismatch')
    if not n or not np.isfinite(data.obs).all() or not np.issubdtype(data.actions.dtype, np.integer):
        raise ValueError('Empty/nonfinite observations or noninteger actions')
    if np.any((data.actions < 0) | (data.actions >= ACTION_DIM)):
        raise ValueError('Action outside discrete27')
    if any(x is None or x.shape != (n,) for x in (data.episodes, data.families, data.drills)):
        raise ValueError('Episode, family and drill labels are required')
    if not np.issubdtype(data.episodes.dtype, np.integer) or not np.isin(data.drills, DRILLS).all():
        raise ValueError('Invalid episode or drill labels')
    for episode in np.unique(data.episodes):
        if len(np.unique(data.drills[data.episodes == episode])) != 1:
            raise ValueError('An episode cannot span different drills')


def save_demos(path, data, summary):
    validate_rows(data)
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, obs=data.obs, actions=data.actions, episodes=data.episodes,
                            families=data.families, drills=data.drills,
                            contract_json=np.asarray(json.dumps(metadata())),
                            summary_json=np.asarray(json.dumps(summary)))


def load_demos(path):
    with np.load(path, allow_pickle=False) as archive:
        validate_metadata(json.loads(str(archive['contract_json'].item())))
        data = DemoBatch(*(archive[key].copy() for key in ('obs', 'actions', 'episodes', 'families', 'drills')))
    validate_rows(data)
    return data


def split_episodes(data, *, seed=42):
    validate_rows(data)
    rng = np.random.default_rng(seed)
    heldout = []
    for drill in DRILLS:
        episodes = np.unique(data.episodes[data.drills == drill])
        if len(episodes) < 2:
            raise ValueError(f'Need at least two complete {drill} episodes for a held-out split')
        rng.shuffle(episodes)
        heldout.extend(episodes[:max(1, int(np.ceil(.2 * len(episodes))))])
    val = np.isin(data.episodes, heldout)
    return np.flatnonzero(~val), np.flatnonzero(val)


def slices(data):
    attack = (data.actions >= int(Action.LIGHT_NEUTRAL)) & (data.actions <= int(Action.HEAVY_DOWN))
    return dict(attack=attack, ground_attack=attack & (data.obs[:, StateSpec.index('player_grounded')] > .5),
                pickup=data.actions == int(Action.PICKUP), recovery=data.drills == 'recovery')


def gate_predictions(data, predictions):
    # These are declared minimum imitation checks, not claims of fighting mastery.
    thresholds = dict(attack=(20, .60), ground_attack=(10, .60), pickup=(5, .60), recovery=(20, .60))
    predictions = np.asarray(predictions)
    if predictions.shape != data.actions.shape:
        raise ValueError('Predictions must align with labels')
    report = dict(passed=True, failures=[], slices={}, interpretation='Held-out imitation, not live mastery')
    for name, selected in slices(data).items():
        count, (minimum, threshold) = int(selected.sum()), thresholds[name]
        accuracy = float((predictions[selected] == data.actions[selected]).mean()) if count else 0.
        passed = count >= minimum and accuracy >= threshold
        report['slices'][name] = dict(samples=count, accuracy=accuracy, min_samples=minimum,
                                      min_accuracy=threshold, passed=passed)
        if not passed:
            report['failures'].append(name)
    report['passed'] = not report['failures']
    pickup = predictions == int(Action.PICKUP)
    correct = int((pickup & (data.actions == int(Action.PICKUP))).sum())
    count = int(pickup.sum())
    # Exact precision penalizes differences in the teacher's hidden retry timer;
    # report it separately from impossible-context requests, not as a new gate.
    report['pickup_diagnostics'] = dict(
        predicted=count, correct=correct, false_positives=count-correct,
        precision=correct/count if count else None,
        predicted_while_armed=int((pickup & (data.obs[:, StateSpec.index('player_has_weapon')] > .5)).sum()),
        predicted_without_ground_weapon=int((pickup & (data.obs[:, StateSpec.index('weapon_on_ground')] <= .5)).sum()),
        predicted_beyond_teacher_range=int((pickup & (np.hypot(
            data.obs[:, StateSpec.index('weapon_dx')],
            data.obs[:, StateSpec.index('weapon_dy')]) > .05)).sum()))
    return report


def subset(data, indices):
    return DemoBatch(*(getattr(data, k)[indices] for k in ('obs', 'actions', 'episodes', 'families', 'drills')))


def sampling_probabilities(data, indices):
    # Square-root inverse frequency lifts rare inputs without making a single
    # pickup row as influential as hundreds of distinct movement examples.
    actions = data.actions[indices]
    counts = np.bincount(actions, minlength=ACTION_DIM)
    weights = np.minimum(8., np.sqrt(counts.max() / counts[actions]))
    return weights / weights.sum()


def build_model(env, **overrides):
    from stable_baselines3 import PPO
    from train.primitive_features import PrimitiveGeometry
    options = dict(n_steps=512, batch_size=256, n_epochs=4, learning_rate=1e-4,
                   gamma=.999, gae_lambda=.95, clip_range=.1, ent_coef=.01,
                   target_kl=.01, device='cpu', seed=42, verbose=0,
                   policy_kwargs={'net_arch': {'pi': [128, 128], 'vf': [128, 128]},
                                  'features_extractor_class': PrimitiveGeometry})
    options.update(overrides)
    return PPO('MlpPolicy', env, **options)


def save_model(model, path, **extra):
    from train.checkpoint_roles import save_typed_checkpoint
    return save_typed_checkpoint(model, path, metadata() | extra)


def load_model(path, env=None, **overrides):
    from stable_baselines3 import PPO
    from train.checkpoint_roles import read_checkpoint_metadata
    validate_metadata(read_checkpoint_metadata(path))
    model = PPO.load(path, env=env, device='cpu', **overrides)
    validate_metadata(model.checkpoint_metadata)
    if model.observation_space.shape != (OBSERVATION_DIM,) or model.action_space.n != ACTION_DIM:
        raise ValueError('Serialized policy spaces do not match primitive27')
    return model


def agreement(model, data, indices):
    selected = subset(data, indices)
    predictions, _ = model.predict(selected.obs, deterministic=True)
    report = gate_predictions(selected, predictions)
    report['overall_accuracy'] = float((predictions == selected.actions).mean())
    return report


def pretrain(model, data, *, epochs=40, batch_size=256, learning_rate=3e-4, seed=42, output):
    import torch as th
    train, val = split_episodes(data, seed=seed)
    if min(epochs, batch_size, learning_rate) <= 0 or not np.isfinite(learning_rate):
        raise ValueError('Invalid BC configuration')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'bc_split.json').write_text(json.dumps(dict(seed=seed,
        train_episodes=np.unique(data.episodes[train]).tolist(),
        validation_episodes=np.unique(data.episodes[val]).tolist(), stratified_by='drill'), indent=2))
    rng = np.random.default_rng(seed)
    th.manual_seed(seed)
    optimizer = th.optim.Adam(model.policy.parameters(), lr=learning_rate)
    probabilities = sampling_probabilities(data, train)
    metrics = []
    for epoch in range(1, epochs + 1):
        model.policy.set_training_mode(True)
        order = rng.choice(train, size=len(train), replace=True, p=probabilities)
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start+batch_size]
            dist = model.policy.get_distribution(th.as_tensor(data.obs[idx], device=model.device))
            loss = -dist.log_prob(th.as_tensor(data.actions[idx], device=model.device)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), .5)
            optimizer.step()
            losses.append(float(loss.item()))
        report = agreement(model, data, val)
        row = dict(epoch=epoch, loss=float(np.mean(losses)), accuracy=report['overall_accuracy'],
                   **{name: value['accuracy'] for name, value in report['slices'].items()})
        metrics.append(row)
        print(f'[BC {epoch}/{epochs}] {row}', flush=True)
    with (output/'bc_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    (output/'bc_gate.json').write_text(json.dumps(report, indent=2))
    return report
