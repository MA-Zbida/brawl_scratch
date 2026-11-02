"""Round A: aligned demos -> architecture-specific BC -> timed PPO benchmark.

This entry point is deliberately separate from legacy checkpoint loaders. Live
stages import capture_first before any module that can import torch.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import uuid


from brawl.pipelines.round_a.config import CANDIDATES, Protocol, checkpoint_contract, validate_dataset_contract
from brawl.training.run_utils import prompt, sha256, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=('collect', 'pretrain', 'train', 'evaluate', 'smoke', 'preflight'))
    p.add_argument('--candidate', choices=CANDIDATES, default='mlp')
    p.add_argument('--protocol', type=Path, default=Path('experiments/round_a/protocol_v1.json'))
    p.add_argument('--set', action='append', default=[], metavar='PROTOCOL_KEY=JSON_VALUE')
    p.add_argument('--run-dir', type=Path, help='New directory only')
    p.add_argument('--demos', type=Path, help='Aligned dataset directory, not a legacy NPZ')
    start = p.add_mutually_exclusive_group()
    start.add_argument('--resume', type=Path, help='Typed Round A checkpoint; exact protocol resume')
    start.add_argument('--init-bc', type=Path, help='Zero-step Round A BC initializer; permits a new run/evaluation schedule')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--episodes-per-drill', type=int, default=30)
    p.add_argument('--attempts-factor', type=int, default=4)
    p.add_argument('--delay', type=float, default=5.)
    p.add_argument('--weapon-classifier-checkpoint', type=Path,
                   default=Path('train/models/weapon_classifier_r18_live_v1.pt'))
    p.add_argument('--world-replay', type=Path, help='Opt-in structured dynamics replay; never training eval data')
    p.add_argument('--profile-timing', action='store_true',
                   help='Train stage: save qualification_timing.json even if the live gate fails; no PPO profiling')
    p.add_argument('--eval-video-seconds', type=float, default=0.,
                   help='Opt-in: record the first N seconds of each evaluation block (0 disables, max 600)')
    return p


def validate(args):
    protocol = Protocol.load(args.protocol)
    updates = {}
    for entry in args.set:
        name, sep, value = entry.partition('=')
        if not sep or name not in protocol.as_dict():
            raise ValueError(f'Unknown protocol override: {entry}')
        updates[name] = json.loads(value)
    protocol = replace(protocol, **updates)
    if args.seed < 0 or args.episodes_per_drill < 5 or args.attempts_factor < 1:
        raise ValueError('Invalid seed/collection budgets')
    if not 0 <= args.delay <= 60:
        raise ValueError('delay must be between 0 and 60')
    if not 0 <= args.eval_video_seconds <= 600:
        raise ValueError('eval-video-seconds must be between 0 and 600')
    if args.stage in ('pretrain', 'train', 'preflight') and args.demos is None:
        raise ValueError(f'{args.stage} requires --demos')
    if args.stage == 'train' and args.resume is None and args.init_bc is None:
        raise ValueError('train requires --resume or --init-bc')
    if args.stage == 'evaluate' and args.resume is None:
        raise ValueError('evaluate requires --resume')
    if args.init_bc and args.stage != 'train':
        raise ValueError('--init-bc only applies to train')
    if args.resume and args.stage not in ('train', 'evaluate'):
        raise ValueError('--resume only applies to train/evaluate')
    if args.stage == 'collect' and args.candidate != 'mlp':
        raise ValueError('Collect once with default candidate; every arm consumes this same dataset')
    for path in (args.protocol, args.resume, args.init_bc):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    checkpoint = args.init_bc or args.resume
    if checkpoint:
        # Reject incompatible input contracts before acquiring capture or CUDA.
        from brawl.training.checkpoints import read_checkpoint_metadata
        payload = read_checkpoint_metadata(checkpoint)
        checkpoint_contract(payload, protocol, args.candidate, initialize_bc=bool(args.init_bc))
        steps = payload.get('environment_steps', 0)
        if args.stage == 'train' and (steps >= protocol.rollout_budget or steps % protocol.n_steps):
            raise ValueError('Use a completed-rollout checkpoint with remaining training budget')
    return protocol


def provenance(args, protocol, directory):
    from brawl.paths import REPO_ROOT as root, source_files
    sources = source_files()
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'stable-baselines3', 'sb3-contrib', 'gymnasium', 'numpy')}
    artifacts = {}
    for name, path in [('resume', args.resume), ('initializer', args.init_bc),
                       ('demos', args.demos / 'dataset.json' if args.demos else None),
                       ('classifier', args.weapon_classifier_checkpoint), ('detector', root / 'assets/yolo/best.engine')]:
        if path is not None:
            artifacts[name] = dict(path=str(path.resolve()), sha256=sha256(path) if path.is_file() else None)
    write_json(directory / 'manifest.json', dict(schema_version=1, protocol=protocol.as_dict(),
        arguments=vars(args), versions=versions, artifacts=artifacts,
        source_hashes={p.relative_to(root).as_posix(): sha256(p) for p in sources if p.is_file()},
        mode='training_room_stock_exchanges_not_match_wins', manual_bot_settings=True))


def make_live(args, protocol, directory, stage):
    from brawl.pipelines.primitive27.run import _make_live
    from brawl.env.match import MatchTrainingEnv, ReplayGymWrapper
    from brawl.pipelines.round_a.data import ObservationAdapter
    base = _make_live(args)
    match = MatchTrainingEnv(base, mode='training_room', max_steps=protocol.window_steps,
        matches_path=directory / f'{stage}_windows.csv', steps_path=directory / f'{stage}_transitions.csv')
    match.usage = stage
    wrapped = match
    if args.world_replay:
        from brawl.world_model.replay_log import WorldReplayWriter
        writer = WorldReplayWriter(args.world_replay, phase=f'round_a_{stage}',
            metadata={'candidate': args.candidate, 'run_directory': str(directory),
                      'protocol': protocol.as_dict(), 'contains_evaluation_and_update_gap_tags': True})
        wrapped = ReplayGymWrapper(match, writer)
        print(f'World replay: {writer.dir}', flush=True)
    return ObservationAdapter(wrapped, args.candidate, width=protocol.image_width,
                              height=protocol.image_height, hz=protocol.control_hz,
                              prepare_images=(protocol.execution_mode == 'fixed_cadence'
                                              or stage == 'collect' or args.candidate == 'multi')), match


def space_env(candidate, protocol):
    import gymnasium as gym
    import numpy as np
    class Spaces(gym.Env):
        action_space = gym.spaces.Discrete(27)
        observation_space = gym.spaces.Box(-np.inf, np.inf, (146,), np.float32)
        if candidate == 'multi':
            observation_space = gym.spaces.Dict(structured=observation_space,
                frames=gym.spaces.Box(0, 255, (12, protocol.image_height, protocol.image_width), np.uint8))
    return Spaces()


def model_options(args, protocol):
    return dict(n_steps=protocol.n_steps, batch_size=protocol.batch_size, n_epochs=protocol.n_epochs,
                learning_rate=protocol.learning_rate, ent_coef=protocol.ent_coef, seed=args.seed, device=args.device)


def smoke(args, protocol, directory):
    import gymnasium as gym
    import numpy as np
    from brawl.pipelines.round_a.models import build_model, save_model, load_model, Predictor
    class Synthetic(gym.Env):
        action_space = gym.spaces.Discrete(27)
        observation_space = space_env(args.candidate, protocol).observation_space
        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.count = 0
            return self.observation_space.sample(), {}
        def step(self, action):
            self.count += 1
            return self.observation_space.sample(), float(action == 0), False, self.count >= 8, {}
    env = Synthetic()
    model = build_model(args.candidate, env, n_steps=8, batch_size=4, n_epochs=1, device=args.device)
    model.learn(16)
    contract = dict(candidate=args.candidate, **protocol.as_dict())
    save_model(model, directory / 'smoke.zip', contract)
    restored = load_model(directory / 'smoke.zip', args.candidate, env, device=args.device)
    Predictor(restored).predict(env.reset()[0], start=True)
    write_json(directory / 'status.json', dict(status='synthetic_pass', live_qualified=False))


def preflight(args, protocol, data, directory):
    """Offline feasibility, never claims that the complete live loop meets cadence."""
    import numpy as np
    import torch
    from brawl.pipelines.round_a.models import build_model, Predictor
    report = {}
    for candidate in CANDIDATES:
        model = build_model(candidate, space_env(candidate, protocol), **model_options(args, protocol))
        episode = data.episode(data.manifest['episodes'][0])
        obs = data.observations(episode, np.array([0]), candidate)
        obs = {k: v[0] for k, v in obs.items()} if isinstance(obs, dict) else obs[0]
        predictor = Predictor(model)
        timings = []
        for i in range(110):
            if args.device == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            predictor.predict(obs)
            if args.device == 'cuda': torch.cuda.synchronize()
            if i >= 10: timings.append(time.perf_counter()-start)
        report[candidate] = dict(parameters=sum(p.numel() for p in model.policy.parameters()),
            inference_p50_s=float(np.median(timings)), inference_p95_s=float(np.quantile(timings, .95)),
            excludes_capture_detection=True)
        del model
    import cv2
    sample = data.episode(data.manifest['episodes'][0])['frames'][0]
    cv2.imwrite(str(directory / 'resolution_preview.png'), cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
    write_json(directory / 'preflight.json', report)
    print(json.dumps(report, indent=2))


def main(argv=None):
    args = parser().parse_args(argv)
    protocol = validate(args)
    live = args.stage in ('collect', 'train', 'evaluate')
    if live:
        import brawl.capture_first  # noqa: F401 -- DXGI before CUDA.
    import torch
    torch.set_num_threads(1)
    from brawl.pipelines.round_a.data import Dataset
    from brawl.pipelines.round_a.models import build_model, load_model, save_model
    from brawl.pipelines.round_a.bc import pretrain, evaluate_imitation
    data = Dataset(args.demos) if args.demos else None
    if data is not None:
        if data.manifest.get('complete') is False:
            raise ValueError('Incomplete collection: do not pretrain or compare candidates')
        validate_dataset_contract(data.manifest, protocol)
        if {entry['drill'] for entry in data.manifest['episodes']} != {'weapon', 'recovery', 'combat'}:
            raise ValueError('Round A requires the same three drills for every candidate')
    checkpoint = args.init_bc or args.resume
    model = load_model(checkpoint, args.candidate, device=args.device) if checkpoint else None
    if model is not None:
        contract = checkpoint_contract(model.checkpoint_metadata, protocol, args.candidate,
                                       initialize_bc=bool(args.init_bc))
        if args.init_bc and (model.num_timesteps != 0 or model._n_updates != 0):
            raise ValueError('--init-bc cannot reset an already trained PPO model')
        if data is not None and contract.get('dataset_sha256') != sha256(args.demos / 'dataset.json'):
            raise ValueError('Checkpoint and demonstrations differ')
        if args.stage == 'train' and model.seed != args.seed:
            raise ValueError('Training seed differs from BC checkpoint; pretrain each declared seed explicitly')
    else:
        contract = dict(candidate=args.candidate, **protocol.as_dict(),
                        dataset_sha256=sha256(args.demos / 'dataset.json') if args.demos else None)
    directory = args.run_dir or Path('train/models/round_a') / (time.strftime('%Y%m%d-%H%M%S')+'_'+args.candidate+'_'+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True, exist_ok=False)
    provenance(args, protocol, directory)
    if args.init_bc:
        write_json(directory / 'initialization.json', dict(
            source=str(args.init_bc.resolve()), sha256=sha256(args.init_bc),
            source_contract=model.checkpoint_metadata['round_a'], destination_contract=contract,
            source_training_stage=model.checkpoint_metadata['training_stage'],
            source_environment_steps=int(model.num_timesteps),
            interpretation='Reuse fixed BC weights; no PPO interactions reset or discarded.'))
    print(f'Round A output: {directory.resolve()}', flush=True)
    args.no_prompt = False
    adapter = None
    try:
        if args.stage == 'smoke':
            smoke(args, protocol, directory)
            return
        if args.stage == 'preflight':
            preflight(args, protocol, data, directory)
        elif args.stage == 'pretrain':
            model = build_model(args.candidate, space_env(args.candidate, protocol), **model_options(args, protocol))
            pretrain(model, data, protocol, directory, seed=args.seed)
            save_model(model, directory / 'bc_init.zip', contract, training_stage='bc_final_epoch')
        elif args.stage == 'collect':
            from brawl.pipelines.round_a.collection import collect
            prompt(args, 'Collect ONCE: Wu Shang vs NON-ATTACKING moving Medium Mordex, Small Brawlhaven.')
            adapter, _ = make_live(args, protocol, directory, 'collect')
            collect(adapter, protocol, directory, episodes=args.episodes_per_drill, attempts_factor=args.attempts_factor)
        else:
            from brawl.pipelines.round_a.runtime import evaluate, run_training
            if args.stage == 'train':
                from brawl.pipelines.round_a.collection import qualify
                prompt(args, 'Qualification: NON-ATTACKING Medium Mordex. Check visual resolution and control timing.')
                adapter, match = make_live(args, protocol, directory, 'qualify')
                from brawl.pipelines.round_a.timing import qualification_timing
                profiling = (qualification_timing(model, adapter, directory / 'qualification_timing.json')
                             if args.profile_timing else nullcontext())
                with profiling:
                    qualification = qualify(model, adapter, protocol, directory)
                if not qualification['passed']:
                    raise RuntimeError('Live qualification/cadence failed; see qualification.json. No PPO started.')
                adapter.close()
                adapter = None
                if input('Were pickup, recovery and attacks visibly correct? Type YES: ').strip() != 'YES':
                    raise RuntimeError('Operator did not approve qualification')
            prompt(args, 'Configure ATTACKING Medium Mordex; continuous respawns, same map and characters.')
            adapter, match = make_live(args, protocol, directory, 'train')
            adapter.reset()
            if args.stage == 'train':
                run_training(model, adapter, match, protocol, directory, contract, data=data,
                             video_seconds=args.eval_video_seconds)
            else:
                evaluate(model, adapter, match, protocol.final_seconds, protocol.eval_block_seconds, directory, 'evaluation',
                         video_seconds=args.eval_video_seconds)
                if data is not None:
                    write_json(directory / 'imitation_test.json', evaluate_imitation(model, data,
                        data.split(protocol.split_seed)['test'], sequence_length=protocol.sequence_length))
        write_json(directory / 'status.json', dict(status='completed', stage=args.stage))
        if model is not None:
            write_json(directory / 'model_config.json', dict(candidate=args.candidate,
                parameters=sum(p.numel() for p in model.policy.parameters()),
                device=str(model.device), n_steps=model.n_steps, batch_size=model.batch_size,
                n_epochs=model.n_epochs, gamma=model.gamma, gae_lambda=model.gae_lambda,
                learning_rate=model.learning_rate, ent_coef=model.ent_coef,
                clip_range=model.clip_range(1.), target_kl=model.target_kl,
                policy_class=f'{type(model.policy).__module__}.{type(model.policy).__name__}',
                policy_kwargs=model.policy_kwargs))
        if args.stage in ('train', 'evaluate', 'pretrain'):
            from brawl.analysis.plot_round_a import summarize
            summarize([directory], directory / 'report')
    except BaseException as exc:
        write_json(directory / 'status.json', dict(status='stopped', error=str(exc), stage=args.stage))
        raise
    finally:
        if adapter is not None:
            adapter.close()


if __name__ == '__main__':
    main()
