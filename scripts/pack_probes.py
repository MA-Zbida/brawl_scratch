"""Pack the probes for a bigger GPU (Kaggle): the code, and each session's decisions and crops.

    python scripts/prepare_demos.py                       # new recordings -> decisions
    python scripts/prepare_crops.py --difficulty medium   # -> crops
    python scripts/pack_probes.py                         # -> <BRAWL_DATA>/probes/brawl_probe_pack.zip

Upload the zip as a Kaggle dataset, add it to a notebook with two T4 GPUs, and run CELL (printed at
the end). The results come back in /kaggle/working/code: results/probes/vision.md and
runs/probes/vision/folds_*.json; copy both into this repository. Recordings themselves (frames,
stills) never leave this machine: only 96 px grey crops of the fighters and the vectors do.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brawl.paths import DATA  # noqa: E402
from brawl.record import Recording, is_demo  # noqa: E402

CODE = ("brawl/__init__.py", "brawl/paths.py", "brawl/actions.py", "brawl/stage.py", "brawl/observation.py",
        "brawl/ledger.py", "brawl/record.py", "brawl/game/__init__.py", "brawl/game/keys.py", "brawl/sense/__init__.py",
        "brawl/sense/hud.py", "brawl/sense/perception.py", "brawl/sense/track.py", "brawl/sense/truth.py",
        "scripts/probe_vision.py")
CELL = '''
import glob, os, shutil, subprocess, sys, time
pack = glob.glob("/kaggle/input/**/code/scripts/probe_vision.py", recursive=True)[0].split("/code/")[0]
shutil.copytree(f"{pack}/code", "/kaggle/working/code", dirs_exist_ok=True)   # results are written beside the code
env = {**os.environ, "BRAWL_DATA": pack}
jobs = [subprocess.Popen([sys.executable, "scripts/probe_vision.py", "--part", f"{g}/2", "--gpu", str(g), "--tag", f"gpu{g}",
                          "--hours", "9.5"], cwd="/kaggle/working/code", env=env,
                         stdout=open(f"/kaggle/working/log_{g}.txt", "w"), stderr=subprocess.STDOUT) for g in range(2)]
end = time.time() + 10.25 * 3600   # Kaggle ends a notebook at 12 h: each job stops itself at 9.5 h, this is the second guard
for j in jobs:
    try:
        print(j.wait(timeout=max(1, end - time.time())))
    except subprocess.TimeoutExpired:
        j.kill()
        print("stopped by the notebook's own limit; finished runs are kept")
subprocess.run([sys.executable, "scripts/probe_vision.py", "--report"], cwd="/kaggle/working/code", env=env)
'''


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--difficulty", default="medium")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--size", type=int, default=96)
    args = p.parse_args(argv)
    levels = set(args.difficulty.split(","))
    names = (f"decisions_k{args.k}.npz", f"crops_k{args.k}_{args.size}.npy", "meta.json")
    roots = [q for q in sorted((DATA / "demos").iterdir()) if all((q / n).exists() for n in names)
             and is_demo(Recording(q).meta) and Recording(q).meta["setting"].get("difficulty") in levels]
    out = DATA / "probes" / "brawl_probe_pack.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    decisions = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as z:  # crops barely compress; stored is fast
        for rel in CODE:
            z.write(ROOT / rel, f"code/{rel}")
        for root in roots:
            decisions += Recording(root).meta["rows"] // args.k
            for name in names:
                z.write(root / name, f"demos/{root.name}/{name}")
    print(f"{len(roots)} sessions, {decisions} decisions ({decisions * args.k / 3600:.0f} min) -> {out} "
          f"({out.stat().st_size / 1e6:.0f} MB)\n\nKaggle notebook cell (GPU T4 x2):\n{CELL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
