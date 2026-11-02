"""Frozen BC/PPO evaluation crossed with logging A/B, at uncapped live cadence.

No training, checkpoint conversion, video encoding, or automatic menu actions.
Minimal blocks retain metrics/trace in RAM and persist them only after playing.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import math
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.run_utils import sha256, write_json


def block_schedule(rounds: int) -> list[tuple[str, str]]:
    if not 1 <= rounds <= 4:
        raise ValueError('rounds must be between 1 and 4')
    # Reversing the second cycle balances early/late positions of all conditions.
    order = [('bc', 'normal'), ('ppo', 'minimal'), ('ppo', 'normal'), ('bc', 'minimal')]
    return [item for cycle in range(rounds) for item in (order if cycle % 2 == 0 else order[::-1])]


def validate_pair(bc: dict, ppo: dict) -> None:
    if bc.get('training_stage') != 'bc_final_epoch' or bc.get('environment_steps') != 0:
        raise ValueError('BC must be a proven zero-step BC initializer')
    if ppo.get('environment_steps', 0) <= 0:
        raise ValueError('PPO must have training interactions')
    # Evaluation does not resume an optimizer or rewrite the source protocol.
    for key in ('candidate', 'image_width', 'image_height', 'sequence_length',
                'dataset_sha256', 'split_seed', 'bc_epochs'):
        if bc['round_a'].get(key) != ppo['round_a'].get(key):
            raise ValueError(f'Checkpoint input mismatch: {key}')


def _play(model, adapter, obs, seconds: float, *, clock, probe=None, progress=None) -> tuple[dict, object]:
    from train.round_a_models import Predictor
    from train.round_a_runtime import SkillMetrics
    predictor, metrics = Predictor(model), SkillMetrics()
    requested, effective = Counter(), Counter()
    row = progress if progress is not None else {}
    row.update(interactions=0, damage_dealt=0., damage_received=0., stocks_taken=0., stocks_lost=0., reward=0.)
    began = previous = clock()
    start = True
    try:
        while clock() - began < seconds:
            if probe is not None:
                probe.decision = row['interactions']
            with probe.measure('decision') if probe is not None else nullcontext():
                with probe.measure('policy_predict') if probe is not None else nullcontext():
                    action = predictor.predict(obs, start=start)
                obs, reward, done, truncated, info = adapter.step(action)
                now = clock()
                with probe.measure('evaluation_accounting') if probe is not None else nullcontext():
                    metrics.observe(obs, info, now - previous)
                    previous = now
                    row['interactions'] += 1
                    row['reward'] += float(reward)
                    requested[action] += 1
                    effective[int(info.get('effective_action', action))] += 1
                    for key, source in [('damage_dealt', 'op_delta_damage'), ('damage_received', 'self_delta_damage'),
                                        ('stocks_taken', 'op_stock_lost_step'), ('stocks_lost', 'self_stock_lost_step')]:
                        row[key] += float(info.get(source, 0.))
                start = bool(done or truncated)
                if start:
                    obs, _ = adapter.reset()
    finally:
        row.update(seconds=clock()-began, requested_actions=dict(requested), effective_actions=dict(effective),
                   **metrics.row())
        row['measured_hz'] = row['interactions'] / max(row['seconds'], 1e-9)
        row['net_stocks_per_minute'] = 60 * (row['stocks_taken'] - row['stocks_lost']) / max(row['seconds'], 1e-9)
    return row, obs


def run_block(model, match, contract: dict, directory: Path, *, mode: str, seconds: float,
              warmup: float = 10., clock=time.perf_counter, shard_size: int = 4096) -> dict:
    from train.diagnostic_timing import TraceProbe, attach_live_probe
    from train.round_a_data import ObservationAdapter
    from train.match_env import ReplayGymWrapper
    from world_model.replay_log import WorldReplayWriter
    if mode not in ('normal', 'minimal') or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Invalid logging mode/block duration')
    if not math.isfinite(warmup) or warmup < 0:
        raise ValueError('Invalid warm-up duration')
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'result.json').exists():
        raise FileExistsError(directory)
    # Reuse ONE live environment/detector; neither block nor reporting window
    # teleports the fighters. Warm-up reacquires observations after disk work.
    match.steps_path = match.matches_path = None
    match.usage = 'diagnostic_warmup'
    adapter = ObservationAdapter(match, contract['candidate'], width=contract['image_width'],
        height=contract['image_height'], hz=0, prepare_images=contract['candidate'] == 'multi')
    writer = None
    probe = TraceProbe(clock=clock)
    result = dict(status='failed', logging=mode, control_hz=0,
                  source_control_hz=contract['control_hz'], training_step=getattr(model, 'num_timesteps', None),
                  warmup_seconds=warmup, mode='training_room_stock_exchanges_not_match_wins')
    try:
        obs, _ = adapter.reset()
        # The neutral bridge is excluded from scoring. It refreshes a cached
        # reporting reset after model loads, saves, or a previous block.
        obs, _, done, truncated, _ = adapter.step(0)
        if done or truncated:
            obs, _ = adapter.reset()
        if warmup:
            _, obs = _play(model, adapter, obs, warmup, clock=clock)
        match.usage = match._window_usage = 'evaluation'
        # Start a clean accounting window without resetting physical game state.
        match.reset()
        if mode == 'normal':
            match.steps_path, match.matches_path = directory / 'transitions.csv', directory / 'windows.csv'
            writer = WorldReplayWriter(directory / 'world_replay', shard_size=shard_size,
                phase='diagnostic_evaluation', metadata={'evaluation_only': True, 'logging': mode})
            adapter.env = ReplayGymWrapper(match, writer)
            obs, _ = adapter.reset()
        adapter.release_inputs()
        adapter.intervals.clear()
        adapter.cadence.clear()
        adapter.overruns = 0
        with probe:
            attach_live_probe(probe, adapter, match, writer)
            row, obs = _play(model, adapter, obs, seconds, clock=clock, probe=probe, progress=result)
            result.update(row, status='completed')
            result['in_loop_replay_shards'] = writer._shard_index if writer is not None else 0
            # Flush the tail explicitly OUTSIDE the measured playing duration,
            # but keep its time visible alongside periodic in-loop shard stalls.
            adapter.release_inputs()
            if writer is not None:
                with probe.measure('post_block_replay_close'):
                    writer.close()
        result['cadence'] = adapter.cadence_report()
    except BaseException as exc:
        result['status'] = 'failed'
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            adapter.release_inputs()
            if writer is not None:
                result.setdefault('in_loop_replay_shards', writer._shard_index)
                writer.close()
                result['replay_shards'] = writer._shard_index
            else:
                result['replay_shards'] = 0
                result['in_loop_replay_shards'] = 0
        finally:
            with probe.measure('post_block_csv_close'):
                for handle, _ in match._handles.values():
                    handle.close()
                match._handles.clear()
            match.steps_path = match.matches_path = None
            probe.save(directory)
            write_json(directory / 'result.json', result)
    return result


def summarize(rows: list[dict]) -> dict:
    groups: dict[tuple[str, str], dict] = {}
    for row in rows:
        if row['status'] != 'completed':
            continue
        key = row['policy'], row['logging']
        group = groups.setdefault(key, dict(policy=key[0], logging=key[1], blocks=0,
            **{k: 0. for k in ('seconds', 'interactions', 'stocks_taken', 'stocks_lost',
                              'damage_dealt', 'damage_received')}))
        group['blocks'] += 1
        for name in ('seconds', 'interactions', 'stocks_taken', 'stocks_lost', 'damage_dealt', 'damage_received'):
            group[name] += row[name]
    for group in groups.values():
        minutes = group['seconds'] / 60
        group.update(measured_hz=group['interactions']/group['seconds'],
            net_stocks_per_minute=(group['stocks_taken']-group['stocks_lost'])/minutes,
            damage_trade_per_minute=(group['damage_dealt']-group['damage_received'])/minutes)
    logging_comparisons, policy_comparisons = [], []
    for policy in ('bc', 'ppo'):
        normal, minimal = groups.get((policy, 'normal')), groups.get((policy, 'minimal'))
        if normal and minimal and normal['measured_hz'] and minimal['measured_hz']:
            logging_comparisons.append(dict(policy=policy,
                normal_minus_minimal_ms_per_decision=1000 / normal['measured_hz'] - 1000 / minimal['measured_hz'],
                minimal_over_normal_hz_ratio=minimal['measured_hz']/normal['measured_hz']))
    for mode in ('normal', 'minimal'):
        bc, ppo = groups.get(('bc', mode)), groups.get(('ppo', mode))
        if bc and ppo:
            policy_comparisons.append(dict(logging=mode,
                ppo_minus_bc_net_stocks_per_minute=ppo['net_stocks_per_minute']-bc['net_stocks_per_minute'],
                ppo_minus_bc_damage_trade_per_minute=ppo['damage_trade_per_minute']-bc['damage_trade_per_minute']))
    return dict(blocks=rows, conditions=list(groups.values()), logging_comparisons=logging_comparisons,
        policy_comparisons=policy_comparisons, proves_ppo_improvement=False,
        interpretation='Screening of these checkpoints, not independent training seeds or match wins. '
        'Compare policies WITHIN each logging condition. No physical reset between blocks; '
        'starting states and thermal/OS load differ. Uncapped cadence changes temporal history exposure '
        'relative to source training. Timing instrumentation is present in both logging conditions.')


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bc', type=Path, required=True)
    p.add_argument('--ppo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New directory only')
    p.add_argument('--block-seconds', type=float, default=300.)
    p.add_argument('--rounds', type=int, default=2, help='Four conditions per round; default 8 blocks = 40 min')
    p.add_argument('--warmup-seconds', type=float, default=10.)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--delay', type=float, default=5.)
    p.add_argument('--weapon-classifier-checkpoint', type=Path,
                   default=Path('train/models/weapon_classifier_r18_live_v1.pt'))
    return p


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    schedule = block_schedule(args.rounds)
    if not math.isfinite(args.block_seconds) or not 1 <= args.block_seconds <= 900:
        raise ValueError('block-seconds must be between 1 and 900')
    for value in (args.warmup_seconds, args.delay):
        if not math.isfinite(value) or not 0 <= value <= 60:
            raise ValueError('warmup/delay must be between 0 and 60')
    from train.checkpoint_roles import read_checkpoint_metadata
    payloads = {key: read_checkpoint_metadata(getattr(args, key)) for key in ('bc', 'ppo')}
    validate_pair(payloads['bc'], payloads['ppo'])
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (args.bc, args.ppo, args.weapon_classifier_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    lineage_path = args.ppo.parent / 'initialization.json'
    lineage = json.loads(lineage_path.read_text()) if lineage_path.exists() else None
    if lineage is not None and lineage.get('sha256') != sha256(args.bc):
        raise ValueError('PPO initialization hash differs from supplied BC checkpoint')
    args.output.mkdir(parents=True)
    manifest = dict(arguments=vars(args), schedule=schedule, control_hz=0, deterministic=True,
        initialization_lineage=lineage, lineage_verified=lineage is not None,
        source_checkpoints={key: dict(path=str(getattr(args, key).resolve()),
                                     sha256=sha256(getattr(args, key)), metadata=payloads[key]) for key in payloads},
        source_hashes={str(path): sha256(path) for folder in ('train', 'control', 'perception', 'reward',
                            'feature_extractor', 'capture', 'world_model') for path in Path(folder).rglob('*.py')},
        artifacts={str(path): sha256(path) for path in (Path('env.py'), Path('config.py'), Path('config.json'),
                     Path('tools/profile_round_a.py'), Path('feature_extractor/yolo/best.engine'),
                     args.weapon_classifier_checkpoint) if path.is_file()},
        limitations='Host timing, not isolated CUDA kernels or physical disk I/O; no video. '
                    'Do not use this evaluation replay for training the compared policies.')
    write_json(args.output / 'manifest.json', manifest)
    rows, match = [], None
    status = dict(status='failed')
    try:
        import capture_first  # noqa: F401 -- DXGI acquisition MUST precede torch.
        import torch
        torch.set_num_threads(1)  # Match train.round_a; default CPU pools distort small-MLP latency.
        from train.round_a_models import load_model
        from train.primitive_pipeline import _make_live
        from train.match_env import MatchTrainingEnv
        models = {key: load_model(getattr(args, key), payloads[key]['round_a']['candidate'], device=args.device)
                  for key in payloads}
        for model in models.values():
            model.policy.set_training_mode(False)
        manifest['runtime'] = dict(torch=torch.__version__, torch_threads=torch.get_num_threads(),
                                   device=args.device)
        import importlib.metadata
        manifest['versions'] = {name: importlib.metadata.version(name) for name in
                               ('stable-baselines3', 'sb3-contrib', 'gymnasium', 'numpy', 'ultralytics')}
        write_json(args.output / 'manifest.json', manifest)
        print('Configure Wu Shang vs ATTACKING Medium Mordex, Small Brawlhaven, continuous respawns. '
              'No video; no training; UNCAPPED. Keep game/power/OneDrive settings unchanged between blocks.')
        input('Press Enter when ready, then focus the game. ')
        time.sleep(args.delay)
        match = MatchTrainingEnv(_make_live(args), mode='training_room', max_steps=1200)
        for index, (policy, mode) in enumerate(schedule):
            directory = args.output / f'{index:02d}_{policy}_{mode}'
            print(f'Block {index+1}/{len(schedule)}: {policy} / {mode}, {args.block_seconds:g}s', flush=True)
            try:
                row = run_block(models[policy], match, payloads[policy]['round_a'], directory,
                                mode=mode, seconds=args.block_seconds, warmup=args.warmup_seconds)
            except BaseException:
                if (directory / 'result.json').exists():
                    rows.append(json.loads((directory / 'result.json').read_text()) | {'policy': policy})
                raise
            row.update(policy=policy, block=index)
            write_json(directory / 'result.json', row)
            rows.append(row)
            write_json(args.output / 'summary.json', summarize(rows))
            print(f"  {row['measured_hz']:.2f} Hz; stocks {row['stocks_taken']:g}-{row['stocks_lost']:g}; "
                  f"trade {row['damage_dealt']-row['damage_received']:+.1f}", flush=True)
        status = dict(status='completed')
    except BaseException as exc:
        status['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            if match is not None:
                match.close()
        finally:
            write_json(args.output / 'summary.json', summarize(rows))
            write_json(args.output / 'status.json', status)


if __name__ == '__main__':
    main()
