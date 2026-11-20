"""Pack the fighter-state network's data for Kaggle: the night traces once, the hand labels each round.

    python scripts/pack_states.py            # -> <BRAWL_DATA>/kaggle/states_labels.zip (and states_pretrain/ once)
    python scripts/pack_states.py --force    # rebuild states_pretrain/ too
    python scripts/pack_states.py --labels-only

Two Kaggle datasets:
- **states_pretrain/** (about 3 GB, upload once: drag the folder, or all its files, into a new dataset):
  every 4th trace shard of nights 4 and 5, as recorded (both fighters' crops, observations, events);
  scripts/train_states.py computes the free targets from them. The files are hard links to the traces, so
  the folder takes no extra disk; one flat folder, the night in each name.
- **states_labels.zip** (small, upload again whenever there are more labels): the code and your labelled
  clips (the labelled frame and the 5 before it; "unclear" left out).

On Kaggle (GPU), with both datasets added, run CELL (printed at the end) in one cell. The result comes back
as /kaggle/working/results/states/fighter_state.md and /kaggle/working/states/{pretrained.pt, state_net.pt}.
For a later round, add the earlier notebook's output as a third input: the cell then reuses pretrained.pt.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from brawl.paths import DATA  # noqa: E402
import label_states  # noqa: E402
import train_states  # noqa: E402

RUNS = label_states.RUNS
EVERY = 4
CELL = '''
import glob, os, subprocess, sys
code = glob.glob("/kaggle/input/**/states_code/train_states.py", recursive=True)[0]
labels = glob.glob("/kaggle/input/**/clips_labelled.npz", recursive=True)[0]
shards = glob.glob("/kaggle/input/**/*shard_*.npz", recursive=True)
pretrain = [os.path.commonpath([os.path.dirname(s) for s in shards])] if shards else []
encoder = glob.glob("/kaggle/input/**/states/pretrained.pt", recursive=True)
args = [sys.executable, code, "--labels", labels, "--out", "/kaggle/working"]
args += ["--encoder", encoder[0]] if encoder else ["--pretrain", pretrain[0]] if pretrain else []  # neither: no pretraining
with open("/kaggle/working/log.txt", "w") as log:
    p = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT)
print("exit", p.returncode)
print(open("/kaggle/working/log.txt").read()[-6000:])
'''


def labelled(clips: Path, labels: Path) -> dict:
    """The labelled clips, as train_states.py reads them: the labelled frame and the T - 1 before it."""
    z = np.load(clips)
    done = {k: v for k, v in label_states.read_labels(labels).items() if v != "unclear"}
    ids = np.array(sorted(done), int)
    before = int(z["before"])
    first = before - (train_states.T - 1)
    assert first >= 0, "the clips hold fewer frames before the labelled one than the network reads"
    return {"crops": z["crops"][ids, first:before + 1], "label": np.array([train_states.STATES.index(done[k]) for k in ids], np.int8),
            "run": z["run"][ids], "step": z["step"][ids], "clip": ids}


def link_shards(runs: list[Path], folder: Path, every: int = EVERY) -> list[Path]:
    """Every ``every``-th trace shard of each run, hard-linked into one flat folder as <night>_shard_<step>.npz
    (no copy: the traces and the links share the same bytes on disk)."""
    import os
    import shutil

    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    out, size = [], 0
    for run in runs:
        for s in sorted((run / "trace").glob("shard_*.npz"))[::every]:
            dest = folder / f"{run.parts[-3]}_{s.name}"
            os.link(s, dest)  # fails loudly off the traces' disk rather than copying gigabytes
            out.append(dest)
            size += s.stat().st_size
    print(f"-> {folder}: {len(out)} shards, {size / 2**30:.1f} GB to upload (hard links: no extra disk)")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="rebuild states_pretrain/")
    p.add_argument("--labels-only", action="store_true", help="only states_labels.zip")
    args = p.parse_args(argv)
    out = DATA / "kaggle"
    out.mkdir(parents=True, exist_ok=True)
    pre = out / "states_pretrain"
    if args.labels_only:
        print("labels only: states_pretrain/ not built")
    elif args.force or not pre.exists():
        link_shards(RUNS, pre)
    else:
        print(f"{pre} exists (--force to rebuild)")
    data = labelled(label_states.OUT / "clips.npz", label_states.OUT / "labels.csv")
    lab = out / "states_labels.zip"
    tmp = out / "clips_labelled.npz"
    np.savez_compressed(tmp, **data)
    with zipfile.ZipFile(lab, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "scripts" / "train_states.py", "states_code/train_states.py")
        z.write(tmp, "clips_labelled.npz")
    tmp.unlink()
    counts = np.bincount(data["label"], minlength=len(train_states.STATES))
    print(f"-> {lab} ({lab.stat().st_size / 2**20:.0f} MB): {len(data['label'])} labels, " +
          ", ".join(f"{s} {c}" for s, c in zip(train_states.STATES, counts)) + f"\n\nKaggle cell:\n{CELL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
