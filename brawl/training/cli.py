"""Shared CLI options and validation; no capture or ML imports."""
from __future__ import annotations

import argparse
from pathlib import Path
from brawl.paths import DATA


def parser(*, primitive: bool = False) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('all', 'collect', 'pretrain', 'train', 'evaluate', 'smoke'), default='all')
    p.add_argument('--run-dir', type=Path, help='NEW directory; existing directories are never overwritten')
    p.add_argument('--demos', type=Path, help='New semantic57 archive, not legacy phase demos')
    p.add_argument('--resume', type=Path, help='Typed semantic57 checkpoint for train/evaluate')
    if not primitive:
        p.add_argument('--experimental-controls', action='store_true', help='Opt into candidate semantic controls; NOT a claim of live qualification')
    p.add_argument('--no-prompt', action='store_true', help='Keep the manually configured bot unchanged across stages')
    p.add_argument('--episodes-per-drill', type=int, default=20)
    p.add_argument('--attempts-factor', type=int, default=4)
    p.add_argument('--setup-steps', type=int, default=240)
    p.add_argument('--collection-steps', type=int, default=600)
    p.add_argument('--weapon-hold-steps', type=int, default=40)
    p.add_argument('--bc-epochs', type=int, default=8)
    p.add_argument('--bc-learning-rate', type=float, default=3e-4)
    p.add_argument('--timesteps', type=int, default=500_000, help='Additional PPO transitions; rounded up to complete rollouts')
    p.add_argument('--hours', type=float, default=0., help='Optional wall-clock cap on PPO including evaluations; 0 disables')
    p.add_argument('--n-steps', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--ent-coef', type=float, default=.01)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint-every', type=int, default=10240)
    p.add_argument('--eval-every', type=int, default=51200, help='0 disables scheduled live evaluation')
    p.add_argument('--eval-episodes', type=int, default=3)
    p.add_argument('--window-steps', type=int, default=3000, help='Training-room reporting/evaluation window, not a game reset')
    p.add_argument('--delay', type=float, default=5.)
    p.add_argument('--weapon-classifier-checkpoint', type=Path, default=DATA / 'models' / 'weapon_classifier_r18_live_v1.pt')
    p.add_argument('--world-replay', type=Path, help='Opt-in compressed transitions; no video')
    return p


def _validate(args):
    import math
    for key in ('episodes_per_drill', 'attempts_factor', 'setup_steps', 'collection_steps',
                'weapon_hold_steps', 'bc_epochs', 'timesteps', 'n_steps', 'batch_size',
                'checkpoint_every', 'eval_episodes', 'window_steps'):
        if getattr(args, key) < 1:
            raise SystemExit(f'--{key.replace("_", "-")} must be positive')
    if args.episodes_per_drill < 2 or args.n_steps < 2 or args.batch_size < 2:
        raise SystemExit('At least two episodes, rollout steps and minibatch samples are required')
    if args.n_steps % args.batch_size:
        raise SystemExit('--batch-size must divide --n-steps')
    for key in ('learning_rate', 'bc_learning_rate', 'ent_coef', 'hours', 'delay'):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0:
            raise SystemExit(f'Invalid --{key.replace("_", "-")}')
    if args.learning_rate == 0 or args.bc_learning_rate == 0 or args.eval_every < 0 or args.delay > 60:
        raise SystemExit('Learning rates must be positive, eval interval nonnegative, delay at most 60s')
    live = args.stage in ('all', 'collect', 'train', 'evaluate')
    if live and not args.experimental_controls:
        raise SystemExit('Live semantic57 requires --experimental-controls; first qualify the candidate inputs')
    if args.stage == 'pretrain' and not args.demos:
        raise SystemExit('--stage pretrain requires --demos')
    if args.stage in ('train', 'evaluate') and not args.resume:
        raise SystemExit('--stage train/evaluate requires --resume')
    if args.resume and args.stage not in ('train', 'evaluate'):
        raise SystemExit('--resume is only accepted for train/evaluate')
    if args.demos and args.stage in ('all', 'collect', 'smoke'):
        raise SystemExit('Use --stage pretrain with existing demos; all/collect create fresh demos')
    for path in (args.demos, args.resume, args.weapon_classifier_checkpoint if live else None):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
