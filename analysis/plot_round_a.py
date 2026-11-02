"""Read-only comparison of Round A JSONL evidence; never calls the live game."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def aggregate(rows):
    seconds = sum(row['seconds'] for row in rows)
    sums = {key: sum(row.get(key, 0.) for row in rows) for key in
        ('stocks_taken', 'stocks_lost', 'damage_dealt', 'damage_received', 'recoveries', 'offstage_events',
         'offstage_deaths', 'pickup_opportunities', 'armed_after_opportunity', 'eligible_unarmed_seconds',
         'uncertain_seconds', 'censored_recoveries', 'censored_pickup_opportunities')}
    return dict(seconds=seconds, blocks=len(rows), **sums,
        net_stocks_per_minute=60*(sums['stocks_taken']-sums['stocks_lost'])/max(seconds, 1e-6),
        damage_trade_per_minute=60*(sums['damage_dealt']-sums['damage_received'])/max(seconds, 1e-6))


def summarize(directories, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    time_fig, time_axes = plt.subplots(1, 2, figsize=(12, 4))
    summary, inventory = [], []
    for directory in map(Path, directories):
        manifest = json.loads((directory / 'manifest.json').read_text())
        args = manifest['arguments']
        protocol = manifest.get('protocol', {})
        mode = protocol.get('execution_mode', 'fixed_cadence')
        protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        # Keep failed/unfinished runs visible even when they have no evaluation
        # points. A report is an evidence inventory, not a best-run selector.
        evidence = {}
        for key in ('status', 'qualification', 'initialization', 'training_summary'):
            path = directory / f'{key}.json'
            evidence[key] = json.loads(path.read_text()) if path.exists() else None
        inventory.append(dict(run=str(directory), candidate=args['candidate'], seed=args['seed'],
                              protocol=protocol, protocol_id=protocol_id,
                              artifacts=manifest.get('artifacts', {}), **evidence))
        name = f'{args["candidate"]} seed {args["seed"]} {mode} ({directory.name})'
        grouped = defaultdict(list)
        for row in read_rows(directory / 'evaluations.jsonl'):
            grouped[(row['training_step'], row['label'])].append(row)
        points = []
        timed = []
        for (step, label), rows in sorted(grouped.items()):
            metrics = aggregate(rows)
            summary.append(dict(run=str(directory), candidate=args['candidate'], seed=args['seed'],
                                training_step=step, label=label, execution_mode=mode,
                                protocol_id=protocol_id, cadence=rows[-1].get('cadence'), **metrics))
            points.append((step, metrics))
            if rows[-1].get('elapsed_run_seconds') is not None:
                hours = rows[-1]['elapsed_run_seconds']/3600
                timed.append((hours, metrics))
                summary[-1]['elapsed_run_hours'] = hours
        if points:
            axes[0, 0].plot([s for s, _ in points], [r['net_stocks_per_minute'] for _, r in points], 'o-', label=name)
            axes[0, 1].plot([s for s, _ in points], [r['damage_trade_per_minute'] for _, r in points], 'o-', label=name)
        if timed:
            time_axes[0].plot([s for s, _ in timed], [r['net_stocks_per_minute'] for _, r in timed], 'o-', label=name)
            time_axes[1].plot([s for s, _ in timed], [r['damage_trade_per_minute'] for _, r in timed], 'o-', label=name)
        retention = read_rows(directory / 'retention.jsonl')
        if retention:
            initial = retention[0]['validation']['slices']
            for skill in ('pickup', 'recovery', 'ground_attack'):
                baseline = initial[skill]['accuracy']
                if baseline > 0:
                    axes[1, 0].plot([r['training_step'] for r in retention],
                        [r['validation']['slices'][skill]['accuracy']/baseline for r in retention], label=f'{name}: {skill}')
        bc = read_rows(directory / 'bc_metrics.jsonl')
        if bc:
            axes[1, 1].plot([r['epoch'] for r in bc], [r['validation']['overall_accuracy'] for r in bc], label=name)
    for ax, title, xlabel in zip(axes.flat,
        ('Net stocks / minute (not match wins)', 'Damage trade / minute',
         'Held-out imitation ratio (not live retention)', 'BC validation agreement'),
        ('Training interactions', 'Training interactions', 'Training interactions', 'BC epoch')):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=.25)
        if ax.lines: ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output / 'learning_curves.png', dpi=150)
    plt.close(fig)
    for ax, title in zip(time_axes, ('Net stocks / minute', 'Damage trade / minute')):
        ax.set_title(title)
        ax.set_xlabel('Elapsed run hours (includes updates and evaluation)')
        ax.grid(alpha=.25)
        if ax.lines: ax.legend(fontsize=6)
    time_fig.tight_layout()
    time_fig.savefig(output / 'wall_clock_curves.png', dpi=150)
    plt.close(time_fig)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'run_inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    (output / 'interpretation.txt').write_text(
        'Blocks are correlated gameplay, not independent training seeds. No confidence or superiority claim '
        'is computed from individual frames. Compare final-budget results separately from baseline/interim '
        'checks. Raw opportunity/censor counts are observed sensor events, not privileged game truth. '
        'Deployment runs compare full systems at measured cadence, not isolated algorithms at identical '
        'gameplay exposure. Do not pool pilots, incomplete runs, or different protocols as final-budget results. '
        'The inventory includes every supplied run regardless of outcome.\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New report directory')
    args = parser.parse_args()
    summarize(args.runs, args.output)


if __name__ == '__main__':
    main()
