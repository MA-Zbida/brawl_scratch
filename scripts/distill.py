"""Fold the advised play of past nights back into a new clone, so the next night's gains build on it.

    python scripts/distill.py --name advised_v1 runs/rdqn_medium_v3/default/seed0 runs/advised_medium_night2/default/seed0

One night of the advised clone is one step of policy improvement around a fixed clone; with the clone
fixed, a second night only adds what the clipped advice allows (2026-11-16: +0.26 against +0.38). This
script makes the improvement permanent: behaviour cloning, as for the human (brawl/learn.py), on the
nights' own decisions, kept from the stretches that ended in a stock taken (the advised policy at its
best). The result is a new policy.zip to use as the next night's ``train.rdqn.prior``.

Reads <run>/trace/*.npz; only the decisions played with the advice at full strength are used (the
first hour of each night is the clone alone, or the advice ramping in). Writes runs/bc/<name>/ and
results/bc/<name>.md.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from brawl import actions, learn, observation  # noqa: E402
from brawl.paths import REPO_ROOT, RESULTS, RUNS  # noqa: E402


def night(run: Path, keep: tuple[str, ...]):
    shards = sorted((run / "trace").glob("shard_*.npz"))
    z = [np.load(f) for f in shards]
    step = np.concatenate([x["step"] for x in z])
    obs = np.concatenate([x["obs"] for x in z]).astype(np.float32)
    joint = np.concatenate([x["action"] for x in z])
    events = np.concatenate([x["events"] for x in z])
    action = np.stack([joint // actions.SHAPE[1], joint % actions.SHAPE[1]], 1)
    rows = list(csv.DictReader(open(run / "progress.csv", encoding="utf-8")))
    full = max(float(r["beta"]) for r in rows)
    first = min(int(r["steps"]) for r in rows if float(r["beta"]) >= full) - int(rows[0]["steps"])  # start of full advice
    stretches = [(a, b, o) for a, b, o in learn.stretches(events[:, 2], events[:, 3]) if a >= first and o in keep]
    return obs, action, stretches, len(step), first


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+", help="run folders with a trace/")
    p.add_argument("--name", required=True)
    p.add_argument("--keep", default="won", help="stretches to keep: won, lost, open (comma separated)")
    p.add_argument("--epochs", type=int, default=60)
    args = p.parse_args(argv)
    keep = tuple(args.keep.split(","))
    parts, summary, count = [], [], 0
    for run in args.runs:
        run = run if run.is_absolute() else REPO_ROOT / run
        obs, action, stretches, n, first = night(run, keep)
        kept = sum(b - a for a, b, _ in stretches)
        summary.append(f"{run.relative_to(RUNS)}: {len(stretches)} stretches kept, {kept} of {n - first} advised decisions "
                       f"({kept * 3 / 3600:.0f} of {(n - first) * 3 / 3600:.0f} min)")
        for a, b, _ in stretches:
            prev = np.vstack([np.zeros((1, 2), np.int64), action[a:b - 1]])
            parts.append((obs[a:b], action[a:b], prev, np.full(b - a, count)))
            count += 1
    demos = learn.Demos(*(np.concatenate(x) for x in zip(*parts)), summary)
    print("\n".join(summary), flush=True)
    out = RUNS / "bc" / args.name
    r = learn.clone(demos, out, epochs=args.epochs, net=(256, 256), seed=0)
    v = r["valid"]
    lines = [f"# Distilled clone: {args.name}", "", f"{datetime.now():%Y-%m-%d %H:%M}. scripts/distill.py: behaviour cloning on "
             f"the advised play of past nights (stretches kept: {args.keep}). {r['stretches']} stretches, {r['train_decisions']} "
             f"decisions to train on, {r['valid_decisions']} held out. Best epoch {v['epoch']}.", "",
             "| Measure (held-out stretches) | Value |", "|---|---|",
             *(f"| {label} | {v[key]:.3f} |" for label, key in (("Loss", "loss"), ("Both right", "both"),
                                                               ("Baseline: repeat the previous action", "repeat_previous"),
                                                               ("Right where the action changes", "changed"),
                                                               ("Decisions with a button, played", "press_rate_human"),
                                                               ("Decisions with a button, most likely action", "press_rate_policy"))),
             "", "Nights:", "", *(f"- {s}" for s in summary)]
    path = RESULTS / "bc" / f"{args.name}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[4:12]), f"\n-> {out / 'policy.zip'}\n-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
