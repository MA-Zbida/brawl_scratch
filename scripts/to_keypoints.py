"""Convert the Roboflow detection export into a keypoint-detection import (docs/labelling.md).

Every box gets one point, pre-placed at the box centre. For a fighter the point is moved by hand to
the centre of the torso (its position); for a weapon it stays at the centre. Images are the lossless
stills, named <recording>__<row>.png, in their train/valid/test splits; classes take the spelling
of docs/labelling.md.

    python scripts/to_keypoints.py      # -> <BRAWL_DATA>/labelling/keypoint_import/

Upload that folder to the Roboflow keypoint-detection project, keeping the existing splits.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brawl.paths import DATA  # noqa: E402
from brawl.sense.truth import CLASSES, read_export  # noqa: E402

W, H = 1920, 1080


def line(cls: str, b) -> str:
    cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
    return (f"{CLASSES.index(cls)} {cx / W:.6f} {cy / H:.6f} {(b.x1 - b.x0) / W:.6f} {(b.y1 - b.y0) / H:.6f} "
            f"{cx / W:.6f} {cy / H:.6f} 2")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", type=Path, default=DATA / "labelling" / "export")
    p.add_argument("--demos", type=Path, default=DATA / "demos")
    p.add_argument("--out", type=Path, default=DATA / "labelling" / "keypoint_import")
    args = p.parse_args(argv)
    if args.out.exists():
        shutil.rmtree(args.out)
    counts: dict[str, int] = {}
    for frame in read_export(args.export, args.demos):
        images, labels = args.out / frame.split / "images", args.out / frame.split / "labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        stem = f"{frame.recording}__{frame.row:07d}"
        try:
            os.link(frame.still, images / f"{stem}.png")
        except OSError:
            shutil.copy2(frame.still, images / f"{stem}.png")
        (labels / f"{stem}.txt").write_text("".join(line(b.cls, b) + "\n" for b in frame.boxes), encoding="utf-8")
        counts[frame.split] = counts.get(frame.split, 0) + 1
    (args.out / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(CLASSES)}\nnames: {list(CLASSES)}\nkpt_shape: [1, 3]\nflip_idx: [0]\n", encoding="utf-8")
    print(f"{counts} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
