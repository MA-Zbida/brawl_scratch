"""Collect semantic demos -> masked BC -> CPU PPO on live stock exchanges.

python -m train.semantic_pipeline --help
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

# Direct-script and module entry points share the same imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.pipeline_cli import parser as shared_parser, _validate
from train.pipeline_runner import TrainingHooks, run_training
from train.run_utils import write_json as _json, sha256 as _sha256, prompt as _prompt


def parser(*, primitive=False):
    p = shared_parser(primitive=primitive)
    p.description = __doc__
    return p


def _train(model, wrapped, match, args, directory, data=None, *, backend=None):
    """Compatibility wrapper for existing semantic and primitive callers."""
    if backend is None:
        from train.semantic_contract import metadata
        from train.semantic_bc import agreement
        from train.semantic_demos import split_episodes
        hooks = TrainingHooks(metadata, agreement, split_episodes)
    else:
        from train.primitive_retention import PrimitiveRetentionGuard
        hooks = TrainingHooks(backend.metadata, backend.agreement, backend.split_episodes,
                              PrimitiveRetentionGuard, True)
    return run_training(model, wrapped, match, args, directory, data, hooks=hooks)


def _space_env():
    """An offline contract, never a silent fallback for live training."""
    import gymnasium as gym
    import numpy as np
    from train.semantic_contract import ACTION_DIM, OBSERVATION_DIM

    class ContractEnv(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (OBSERVATION_DIM,), np.float32)
        action_space = gym.spaces.Discrete(ACTION_DIM)

        def reset(self, *, seed=None, options=None):
            raise RuntimeError('Offline BC space cannot collect rollouts')

        def step(self, action):
            raise RuntimeError('Offline BC space cannot collect rollouts')

    return ContractEnv()


def _make_live(args):
    from env import EnvConfig
    from reward.match_objective import NoMatchReward
    from train.semantic_env import SemanticBrawlEnv
    return SemanticBrawlEnv(reward_provider=NoMatchReward(), config=EnvConfig(
        terminate_on_stock_out=False, max_episode_steps=0,
        action_repeat_steps=1, action_repeat_min_steps=1, action_repeat_max_steps=1,
        weapon_classifier_enabled=True,
        weapon_classifier_checkpoint=str(args.weapon_classifier_checkpoint),
    ))


def _wrap(base, args, directory, stage):
    from train.match_env import MatchTrainingEnv, ReplayGymWrapper
    from train.semantic_contract import metadata
    match = MatchTrainingEnv(base, mode='training_room', max_steps=args.window_steps,
                             matches_path=directory / ('windows.csv' if stage == 'train' else f'{stage}_windows.csv'),
                             steps_path=directory / f'{stage}_transitions.csv')
    match.usage = stage if stage != 'train' else 'train'
    wrapped = match
    if args.world_replay:
        from world_model.replay_log import WorldReplayWriter
        writer = WorldReplayWriter(args.world_replay, phase=f'semantic57_{stage}',
                                   metadata={'policy_contract': metadata(), 'run_directory': str(directory),
                                             'collection_includes_discarded_setup': stage == 'collect'})
        wrapped = ReplayGymWrapper(match, writer)
        print(f'World replay: {writer.dir}', flush=True)
    return wrapped, match




def _smoke(args, directory):
    import gymnasium as gym
    import numpy as np
    from train.semantic_contract import ACTION_DIM, OBSERVATION_DIM
    from train.semantic_demos import DemoBatch, save_demos, load_demos
    from train.semantic_bc import pretrain
    from train.semantic_policy import build_model, save_model, load_model

    class FakeEnv(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (OBSERVATION_DIM,), np.float32)
        action_space = gym.spaces.Discrete(ACTION_DIM)

        def obs(self):
            obs = np.zeros(OBSERVATION_DIM, np.float32)
            obs[0] = self.tick % 2
            obs[-ACTION_DIM:] = 1
            return obs

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.tick = 0
            return self.obs(), {}

        def step(self, action):
            self.tick += 1
            return self.obs(), 0., False, False, {'op_delta_damage': float(int(action) == 1)}

        def _release_all_inputs(self):
            pass

    base = FakeEnv()
    obs, _ = base.reset()
    data = DemoBatch(np.stack([obs]*32), np.ones(32, np.int64),
                     np.repeat(np.arange(4), 8), np.array(['movement']*32))
    save_demos(directory / 'demos.npz', data, {'synthetic': True})
    data = load_demos(directory / 'demos.npz')
    args.n_steps, args.batch_size, args.timesteps = 8, 4, 16
    args.checkpoint_every, args.eval_every, args.eval_episodes, args.window_steps = 8, 8, 1, 8
    args.hours = 0.
    model = build_model(_space_env(), n_steps=8, batch_size=4, n_epochs=1)
    pretrain(model, data, epochs=1, batch_size=8, output=directory)
    save_model(model, directory / 'bc_init.zip')
    wrapped, match = _wrap(base, args, directory, 'train')
    model = load_model(directory / 'bc_init.zip', wrapped)
    _train(model, wrapped, match, args, directory, data)
    load_model(directory / 'latest.zip')


def main(argv=None):
    args = parser().parse_args(argv)
    _validate(args)
    live = args.stage in ('all', 'collect', 'train', 'evaluate')
    if live:
        # This MUST precede every import of torch, SB3, ultralytics or env.
        import capture_first  # noqa: F401
    import torch
    torch.set_num_threads(1)
    from train.semantic_contract import metadata
    from train.semantic_policy import build_model, load_model, save_model
    from train.semantic_demos import load_demos, save_demos
    from train.semantic_bc import pretrain
    from train.semantic_collect import collect_demonstrations

    # Validate inputs before touching live controls or creating output artifacts.
    data = load_demos(args.demos) if args.demos else None
    model = load_model(args.resume) if args.resume else None
    directory = args.run_dir or Path('train/models/semantic57') / (time.strftime('%Y%m%d-%H%M%S') + '_' + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True, exist_ok=False)
    hashes = {name: _sha256(path)
              for name, path in (('demos', args.demos), ('resume', args.resume)) if path}
    _json(directory / 'config.json', vars(args) | {'policy_contract': metadata(), 'input_sha256': hashes,
          'python': sys.version, 'torch': torch.__version__, 'device': 'cpu',
          'objective': 'training_room_stock_exchanges_not_match_wins', 'game_inputs_verified': False})
    print(f'Run directory: {directory.resolve()}', flush=True)
    try:
        if args.stage == 'smoke':
            _smoke(args, directory)
        else:
            if args.stage in ('all', 'collect'):
                _prompt(args, 'Collection: Wu Shang vs moving/non-attacking Mordex, Small Brawlhaven, continuous respawns.')
                wrapped, _ = _wrap(_make_live(args), args, directory, 'collect')
                try:
                    data, summary = collect_demonstrations(wrapped, episodes=args.episodes_per_drill,
                        attempts_factor=args.attempts_factor, setup_steps=args.setup_steps,
                        max_steps=args.collection_steps, hold_steps=args.weapon_hold_steps,
                        events_path=directory / 'collection_events.jsonl')
                    _json(directory / 'collection_summary.json', summary)
                    if len(data.actions):
                        path = directory / ('demos.npz' if summary['complete'] else 'partial_demos.npz')
                        save_demos(path, data, summary)
                    if summary.get('interrupted'):
                        raise KeyboardInterrupt
                    if not summary['complete']:
                        raise RuntimeError('Collection quotas not met; saved partial evidence, STOP before BC/PPO')
                finally:
                    wrapped.close()
            if args.stage in ('all', 'pretrain'):
                model = build_model(_space_env(), n_steps=args.n_steps, batch_size=args.batch_size,
                                    learning_rate=args.learning_rate, ent_coef=args.ent_coef, seed=args.seed)
                pretrain(model, data, epochs=args.bc_epochs, batch_size=args.batch_size,
                         learning_rate=args.bc_learning_rate, seed=args.seed, output=directory)
                save_model(model, directory / 'bc_init.zip', training_stage='semantic_bc')
            if args.stage in ('all', 'train', 'evaluate'):
                _prompt(args, 'Live PPO/evaluation: configure attacking Medium Mordex in the training room. No menu automation.')
                wrapped, match = _wrap(_make_live(args), args, directory, 'train')
                # Reload applies explicit runtime PPO overrides, including rollout
                # buffer size; old serialized SB3 settings must not win silently.
                source = args.resume or directory / 'bc_init.zip'
                try:
                    model = load_model(source, wrapped, n_steps=args.n_steps, batch_size=args.batch_size,
                                       learning_rate=args.learning_rate, ent_coef=args.ent_coef, seed=args.seed)
                    _train(model, wrapped, match, args, directory, data)
                finally:
                    wrapped.close()
        _json(directory / 'status.json', {'status': 'smoke_pass' if args.stage == 'smoke' else 'completed',
                                         'stage': args.stage, 'game_inputs_verified': False})
    except KeyboardInterrupt:
        _json(directory / 'status.json', {'status': 'interrupted', 'stage': args.stage})
        print('Interrupted: inputs released; completed checkpoints remain in the run directory.')
        raise SystemExit(130) from None
    except Exception as exc:
        _json(directory / 'status.json', {'status': 'failed', 'stage': args.stage, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
