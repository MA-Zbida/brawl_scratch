"""Run several seeds of one variant back to back, unattended (each with scripts/train.py).

    python scripts/sweep.py --config configs/experiments/s1_ppo.yaml --variant realtime-bc --seeds 1 2 3 4

Click into the game during the first countdown; the later runs start at once, the game still in
front. A finished seed is skipped. The sweep stops when a run does not reach its end (F10, an
error): a stopped run is never resumed, so the next sweep starts that seed afresh.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brawl.config import load_experiment  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--variant", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="default: the experiment's seeds")
    args = p.parse_args(argv)
    experiment = load_experiment(args.config)
    seeds = args.seeds if args.seeds is not None else experiment.seeds
    for n, seed in enumerate(seeds):
        run_dir = experiment.run_dir(args.variant, seed)
        if (run_dir / "policy.zip").exists():
            print(f"seed {seed}: already finished", flush=True)
            continue
        print(f"\n=== {experiment.name} / {args.variant} / seed {seed} ({n + 1} of {len(seeds)}) ===", flush=True)
        code = subprocess.call([sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(args.config),
                                "--variant", args.variant, "--seed", str(seed), "--countdown", "5" if n == 0 else "0"])
        if code != 0 or not (run_dir / "policy.zip").exists():
            print(f"seed {seed} did not reach its end: the sweep stops here", flush=True)
            return 1
    print(f"\ndone. python scripts/analyze.py {experiment.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
