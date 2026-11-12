"""Behaviour cloning from the prepared demonstrations (scripts/prepare_demos.py first).

    python scripts/train_bc.py --name medium_v1 --difficulty medium
    python scripts/train_bc.py --name all --keep won,lost,open      # every decision, not only winning stretches

Reads decisions_k<k>.npz of every recording at the named difficulties, keeps the winning stretches
(brawl/learn.py), trains the policy and writes runs/bc/<name>/policy.zip and results/bc/<name>.md.
The numbers say how well the policy predicts you on stretches it never saw; whether it wins is
measured in the game (scripts/play.py).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brawl import learn  # noqa: E402
from brawl.paths import DATA, RESULTS, RUNS  # noqa: E402
from brawl.record import Recording  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--difficulty", default="", help="comma separated bot levels to learn from (default: all)")
    p.add_argument("--keep", default="won", help="stretches to keep: won, lost, open (comma separated)")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--net", default="256,256")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    levels = {x for x in args.difficulty.split(",") if x}
    roots = [r for r in sorted(q for q in (DATA / "demos").iterdir() if (q / "meta.json").exists())
             if not levels or Recording(r).meta["setting"].get("difficulty") in levels]
    demos = learn.load(roots, args.k, tuple(args.keep.split(",")))
    print("\n".join(demos.summary))
    out = RUNS / "bc" / args.name
    r = learn.clone(demos, out, epochs=args.epochs, net=tuple(int(x) for x in args.net.split(",")), seed=args.seed,
                    device=args.device)
    v, t = r["valid"], r["train"]
    lines = [f"# Behaviour cloning: {args.name}", "",
             f"{datetime.now():%Y-%m-%d %H:%M}. Levels: {', '.join(sorted(levels)) or 'all'}; stretches kept: {args.keep}; "
             f"k = {args.k}; network {args.net}; seed {args.seed}. {r['stretches']} stretches, "
             f"{r['train_decisions']} decisions to train on, {r['valid_decisions']} held out (whole stretches). "
             f"Best epoch {v['epoch']}.", "",
             "| Measure | Held out | Train |", "|---|---|---|",
             *(f"| {label} | {v[key]:.3f} | {t[key]:.3f} |" for label, key in (
                 ("Loss (negative log likelihood)", "loss"), ("Direction right", "direction"), ("Button right", "button"),
                 ("Both right", "both"), ("Baseline: repeat the previous action", "repeat_previous"),
                 ("Both right when the human changes action", "changed"), ("New button presses predicted", "new_press_recall"),
                 ("Share of decisions with a button, human", "press_rate_human"),
                 ("Share of decisions with a button, policy's most likely action", "press_rate_policy"))), "",
             "Recordings:", "", *(f"- {s}" for s in demos.summary)]
    path = RESULTS / "bc" / f"{args.name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[4:15]), f"\n-> {out / 'policy.zip'}\n-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
