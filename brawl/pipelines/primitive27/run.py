"""Discrete27: heuristic drills -> gated BC -> live qualification -> CPU PPO.

No semantic57 checkpoint conversion, no automatic menu input, no recurrent model.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time
import uuid


from brawl.training.run_utils import write_json as _json, prompt as _prompt, sha256 as _sha256
from brawl.training.cli import _validate, parser as shared_parser
from brawl.training.runner import TrainingHooks, run_training
from brawl.training.experiment_config import (parse_experiment_args, write_resolved_experiment,
                                     write_model_config, validate_model_components)
from brawl.paths import LEGACY_RUNS


def parser():
    p = shared_parser(primitive=True)
    p.description = __doc__
    # Six held-out weapon episodes can supply the five-pickup evidence minimum
    # even when the teacher needs just one successful pickup per episode.
    p.set_defaults(episodes_per_drill=30, bc_epochs=40, timesteps=4096, eval_every=4096, window_steps=1200)
    for action in p._actions:
        if action.dest == 'run_dir':
            action.help = 'NEW output directory (default: timestamped train/models/primitive27 run)'
        elif action.dest in ('demos', 'resume'):
            action.help = 'Typed primitive27 artifact only; legacy and semantic57 artifacts are rejected'
    p.add_argument('--live-check-attempts', type=int, default=3,
                   help='Bounded deterministic attempts per drill before PPO; must pass each drill')
    p.add_argument('--experiment', type=Path, help='Versioned JSON experiment configuration')
    p.add_argument('--set', dest='settings', action='append', default=[], metavar='KEY=VALUE',
                   help='Override a named CLI parameter (highest precedence)')
    return p


def _train(model, wrapped, match, args, directory, data=None, *, backend):
    from brawl.training.primitive_retention import PrimitiveRetentionGuard
    hooks = TrainingHooks(backend.metadata, backend.agreement, backend.split_episodes,
                          PrimitiveRetentionGuard, True)
    return run_training(model, wrapped, match, args, directory, data, hooks=hooks)


def _space_env():
    import gymnasium as gym
    import numpy as np
    from brawl.pipelines.primitive27.training import OBSERVATION_DIM, ACTION_DIM
    class SpaceEnv(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (OBSERVATION_DIM,), np.float32)
        action_space = gym.spaces.Discrete(ACTION_DIM)

        def reset(self, **kwargs):
            raise RuntimeError('Offline spaces cannot collect live rollouts')

        def step(self, action):
            raise RuntimeError('Offline spaces cannot collect live rollouts')
    return SpaceEnv()


def _make_live(args):
    from brawl.env.live import EnvConfig
    from brawl.reward.match_objective import NoMatchReward
    from brawl.pipelines.primitive27.env import PrimitiveBrawlEnv
    return PrimitiveBrawlEnv(reward_provider=NoMatchReward(), config=EnvConfig(
        terminate_on_stock_out=False, max_episode_steps=0,
        action_repeat_steps=1, action_repeat_min_steps=1, action_repeat_max_steps=1,
        weapon_classifier_enabled=True,
        weapon_classifier_checkpoint=str(args.weapon_classifier_checkpoint)))


def _wrap(base, args, directory, stage):
    from brawl.env.match import MatchTrainingEnv, ReplayGymWrapper
    from brawl.pipelines.primitive27.training import metadata
    match = MatchTrainingEnv(base, mode='training_room', max_steps=args.window_steps,
                             matches_path=directory/('windows.csv' if stage == 'train' else f'{stage}_windows.csv'),
                             steps_path=directory/f'{stage}_transitions.csv')
    match.usage = stage
    wrapped = match
    if args.world_replay:
        from brawl.world_model.replay_log import WorldReplayWriter
        writer = WorldReplayWriter(args.world_replay, phase=f'primitive27_{stage}',
            metadata={'policy_contract': metadata(), 'run_directory': str(directory),
                      'collection_includes_discarded_setup': stage in ('collect', 'qualify')})
        wrapped = ReplayGymWrapper(match, writer)
        print(f'World replay: {writer.dir}', flush=True)
    return wrapped, match


def qualify(model, env, args, directory):
    from brawl.pipelines.primitive27.training import PrimitiveControls, PrimitiveTeacher
    from brawl.pipelines.semantic57.teacher import TeacherDecision
    from brawl.pipelines.semantic57.collect import collect_attempt
    class PolicyController:
        def action(self, obs):
            if not PrimitiveControls.ready(obs):
                return TeacherDecision(0, 'unobserved')
            action, _ = model.predict(obs, deterministic=True)
            # Family is diagnostic only. No teacher action substitutes for policy.
            return TeacherDecision(int(action), PrimitiveTeacher().action(obs).family)

    report = dict(passed=True, drills={}, interpretation='Deterministic drills, not match wins')
    for drill in ('weapon', 'recovery', 'combat'):
        attempts = []
        for _ in range(args.live_check_attempts):
            result = collect_attempt(env, drill, controls=PrimitiveControls(),
                teacher_factory=PolicyController, setup_steps=args.setup_steps,
                max_steps=args.collection_steps, hold_steps=args.weapon_hold_steps)
            attempts.append(dict(accepted=result.accepted, reason=result.reason, steps=result.steps,
                pickup_requests=result.pickup_requests, peak_hold_steps=result.peak_hold_steps,
                attack_presses=result.attack_presses, damage_dealt=result.damage_dealt))
            print(f'[live BC check: {drill}] {attempts[-1]}', flush=True)
            if result.accepted:
                break
        report['drills'][drill] = attempts
        report['passed'] &= any(row['accepted'] for row in attempts)
        _json(directory/'live_gate.json', report)
    return report


def _smoke(args, directory):
    import gymnasium as gym
    import numpy as np
    from brawl.pipelines.primitive27 import training as backend
    class FakeEnv(gym.Env):
        observation_space = _space_env().observation_space
        action_space = _space_env().action_space
        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(backend.OBSERVATION_DIM, np.float32), {}
        def step(self, action):
            return np.zeros(backend.OBSERVATION_DIM, np.float32), 0., False, False, {}
        def _release_all_inputs(self):
            pass
    env = FakeEnv()
    model = backend.build_model(env, n_steps=8, batch_size=4, n_epochs=1)
    write_model_config(directory, model)
    model.learn(16)
    backend.save_model(model, directory/'smoke.zip')
    backend.load_model(directory/'smoke.zip')
    _json(directory/'status.json', dict(status='smoke_pass', synthetic=True, live_qualified=False))


def main(argv=None):
    args, experiment = parse_experiment_args(parser(), argv)
    args.experimental_controls = True  # Shared scalar validation; not a semantic57 opt-in.
    _validate(args)
    if args.live_check_attempts < 1:
        raise SystemExit('--live-check-attempts must be positive')
    if args.no_prompt and args.stage in ('all', 'train'):
        raise SystemExit('Live qualification requires operator confirmation; omit --no-prompt')
    if args.stage == 'train' and args.demos is None:
        raise SystemExit('--stage train requires --demos for the BC/retention check')
    live = args.stage in ('all', 'collect', 'train', 'evaluate')
    if live:
        import brawl.capture_first  # noqa: F401 -- must precede torch, env and SB3.
    import torch
    torch.set_num_threads(1)
    from brawl.pipelines.primitive27 import training as backend
    from brawl.pipelines.semantic57.collect import collect_demonstrations

    data = backend.load_demos(args.demos) if args.demos else None
    model = backend.load_model(args.resume) if args.resume else None
    if args.experiment and model is not None:
        validate_model_components(model)
    directory = args.run_dir or (LEGACY_RUNS / 'primitive27') / (time.strftime('%Y%m%d-%H%M%S')+'_'+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True, exist_ok=False)
    _json(directory/'config.json', vars(args) | dict(policy_contract=backend.metadata(),
        device='cpu', python=sys.version, torch=torch.__version__,
        input_sha256={name: _sha256(path) for name, path in [('demos', args.demos), ('resume', args.resume)] if path},
        objective='training_room_stock_exchanges_not_match_wins'))
    write_resolved_experiment(directory, args, experiment, model=model)
    print(f'Run directory: {directory.resolve()}', flush=True)
    try:
        if args.stage == 'smoke':
            _smoke(args, directory)
            return
        if args.stage in ('all', 'collect'):
            _prompt(args, 'Collection: Wu Shang vs moving NON-ATTACKING Medium Mordex; Small Brawlhaven, continuous respawns.')
            env, _ = _wrap(_make_live(args), args, directory, 'collect')
            try:
                data, report = collect_demonstrations(env, controls=backend.PrimitiveControls(),
                    episodes=args.episodes_per_drill, attempts_factor=args.attempts_factor,
                    setup_steps=args.setup_steps, max_steps=args.collection_steps,
                    hold_steps=args.weapon_hold_steps, events_path=directory/'collection_events.jsonl')
                _json(directory/'collection_summary.json', report)
                if len(data.actions):
                    backend.save_demos(directory/('demos.npz' if report['complete'] else 'partial_demos.npz'), data, report)
                if report.get('interrupted'):
                    raise KeyboardInterrupt
                if not report['complete']:
                    raise RuntimeError('Collection quotas not met; STOP before BC/PPO')
            finally:
                env.close()
        if args.stage in ('all', 'pretrain'):
            model = backend.build_model(_space_env(), n_steps=args.n_steps, batch_size=args.batch_size,
                learning_rate=args.learning_rate, ent_coef=args.ent_coef, seed=args.seed)
            write_model_config(directory, model)
            report = backend.pretrain(model, data, epochs=args.bc_epochs, batch_size=args.batch_size,
                learning_rate=args.bc_learning_rate, seed=args.seed, output=directory)
            backend.save_model(model, directory/'bc_candidate.zip', training_stage='bc_candidate')
            if not report['passed']:
                raise RuntimeError(f'BC gate failed: {report["failures"]}; see bc_gate.json. STOP before live PPO')
            backend.save_model(model, directory/'bc_init.zip', training_stage='bc_offline_pass')
        if args.stage in ('all', 'train'):
            _, val = backend.split_episodes(data, seed=args.seed)
            report = backend.agreement(model, data, val)
            _json(directory/'bc_gate.json', report)
            if not report['passed']:
                raise RuntimeError('BC gate failed for this checkpoint; STOP before live PPO')
            _prompt(args, 'Deterministic qualification: keep Mordex NON-ATTACKING. Watch attacks, pickup/hold and offstage recovery.')
            env, _ = _wrap(_make_live(args), args, directory, 'qualify')
            try:
                report = qualify(model, env, args, directory)
            finally:
                env.close()
            if not report['passed']:
                raise RuntimeError('Live BC gate failed; see live_gate.json. STOP before PPO')
            answer = input('Inputs are released. Did attacks, pickup and recovery visibly execute correctly? Type YES to continue: ')
            _json(directory/'operator_gate.json', dict(confirmed=answer.strip() == 'YES',
                 source_sha256=_sha256(args.resume or directory/'bc_init.zip')))
            if answer.strip() != 'YES':
                raise RuntimeError('Control qualification not confirmed; STOP before PPO')
            _prompt(args, 'Now configure ATTACKING Medium Mordex. PPO uses measured stock/damage reward; no menu automation.')
        if args.stage in ('all', 'train', 'evaluate'):
            if args.stage == 'evaluate':
                _prompt(args, 'Deterministic evaluation: configure ATTACKING Medium Mordex.')
            env, match = _wrap(_make_live(args), args, directory, 'train')
            try:
                source = args.resume or directory/'bc_init.zip'
                model = backend.load_model(source, env, n_steps=args.n_steps, batch_size=args.batch_size,
                    learning_rate=args.learning_rate, ent_coef=args.ent_coef, seed=args.seed)
                write_model_config(directory, model)
                _train(model, env, match, args, directory, data, backend=backend)
            finally:
                env.close()
        _json(directory/'status.json', dict(status='completed', stage=args.stage))
    except KeyboardInterrupt:
        _json(directory/'status.json', dict(status='interrupted', stage=args.stage))
        raise SystemExit(130) from None
    except Exception as exc:
        _json(directory/'status.json', dict(status='stopped', stage=args.stage, error=str(exc)))
        raise


if __name__ == '__main__':
    main()
