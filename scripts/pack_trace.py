"""Pack a live run's trace (with crops) and the code for the vision value test on Kaggle.

    python scripts/pack_trace.py runs/advised_night4/default/seed0     # -> <BRAWL_DATA>/kaggle/<experiment>_trace.zip

Upload the zip as a Kaggle dataset, add it to a notebook with a GPU (P100 or T4), turn the internet off
or on (nothing is downloaded), and run CELL (printed at the end) in one cell. The laptop cannot run this
test: 10 hours of crops are 13 GB unpacked against 8 GB of RAM. The result comes back as
/kaggle/working/code/results/rdqn/value_bench_vision_<experiment>.md; copy it into results/rdqn/. Only the
trace leaves this machine: vectors, actions, rewards, events and 96 px grey crops of the two fighters.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brawl.paths import DATA  # noqa: E402

CODE = ["configs/base.yaml", "configs/experiments/rdqn_medium_v3.yaml", "scripts/value_bench.py", "scripts/value_bench_vision.py",
        *(p.relative_to(ROOT).as_posix() for d in ("brawl", "brawl/game", "brawl/sense") for p in sorted((ROOT / d).glob("*.py")))]
CELL = '''
import glob, os, shutil, subprocess, sys
code = glob.glob("/kaggle/input/**/code/scripts/value_bench_vision.py", recursive=True)[0].split("/scripts/")[0]
shutil.copytree(code, "/kaggle/working/code", dirs_exist_ok=True, ignore=shutil.ignore_patterns("runs"))  # the result is written beside the code; the trace is read where it is
run = glob.glob(f"{code}/runs/*/default/seed0")[0]
env = {**os.environ, "BRAWL_DATA": "/tmp/brawl_data", "YOLO_AUTOINSTALL": "False"}   # crops unpacked to /tmp (13 GB)
with open("/kaggle/working/log.txt", "w") as log:
    p = subprocess.run([sys.executable, "scripts/value_bench_vision.py", "--trace", run, "--folds", "2"],
                       cwd="/kaggle/working/code", env=env, stdout=log, stderr=subprocess.STDOUT)
print("exit", p.returncode)
print(open("/kaggle/working/log.txt").read()[-3000:])
'''


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="runs/<experiment>/<variant>/seed<k>, its trace saved with crops")
    args = p.parse_args(argv)
    run = (args.run if args.run.is_absolute() else ROOT / args.run).resolve()
    shards = sorted((run / "trace").glob("shard_*.npz"))
    assert shards, f"no trace in {run}"
    out = DATA / "kaggle" / f"{run.parts[-3]}_trace.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    rel = run.relative_to(ROOT).as_posix()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:  # the shards are compressed already
        for f in CODE:
            z.write(ROOT / f, f"code/{f}")
        z.write(run / "progress.csv", f"code/{rel}/progress.csv")
        for i, f in enumerate(shards):
            z.write(f, f"code/{rel}/trace/{f.name}")
            if i % 20 == 0:
                print(f"  {i + 1}/{len(shards)} shards", flush=True)
    print(f"-> {out} ({out.stat().st_size / 2**30:.1f} GB)\n\nKaggle cell:\n{CELL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
