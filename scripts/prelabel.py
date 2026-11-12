"""Draft labels for new frames with the trained pose detector, to be corrected in Roboflow.

    python scripts/prelabel.py <recording> [<recording> ...] --model runs/detector/<name>/weights/best.pt

The detector drafts every label of docs/labelling.md: box, class and torso point. From each
recording's full-resolution stills, half of the ``--count`` frames are those the detector is least
sure of (a fighter missing, or its box at low confidence): correcting them teaches it the most.
The other half are spread evenly over the rest, so ordinary frames stay represented.

Writes <BRAWL_DATA>/labelling/<date_time>/ with images/, labels/ and data.yaml in YOLO keypoint
format: upload that folder to the Roboflow keypoint project (Train/Valid), correct every image by
the rules, and export as before. Images are named <recording>__<row>.png, so each label traces
back to its recording and row.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from brawl.paths import DATA  # noqa: E402
from brawl.sense.truth import CLASSES, FIGHTERS, Box, identity  # noqa: E402


def line(b: Box, width: int, height: int) -> str:
    """One YOLO keypoint label: class, box, point (a weapon's point is its box centre)."""
    (cx, cy), (px, py) = b.centre, b.position if identity(b.cls) else b.centre
    return (f"{CLASSES.index(b.cls)} {cx / width:.6f} {cy / height:.6f} {(b.x1 - b.x0) / width:.6f} "
            f"{(b.y1 - b.y0) / height:.6f} {px / width:.6f} {py / height:.6f} 2")


def sureness(boxes: list[Box]) -> float:
    """The lower confidence of the two fighters' boxes. A missing fighter counts as 0.3: often it
    is truly absent (KO'd, about 3 s each time), so a doubtful box ranks as less sure than no box."""
    scores = {identity(b.cls): b.score for b in boxes if identity(b.cls)}
    return min(scores.get(who, 0.3) for who in FIGHTERS)


def pick(sure: list[float], count: int) -> list[int]:
    """Indices of ``count`` frames: the least sure half, and an even spread over the others."""
    if count <= 0 or count >= len(sure):
        return list(range(len(sure)))
    hard = sorted(np.argsort(sure, kind="stable")[:count // 2].tolist())
    rest = [i for i in range(len(sure)) if i not in set(hard)]
    spread = [rest[i] for i in np.linspace(0, len(rest) - 1, count - len(hard)).round().astype(int)]
    return sorted(hard + spread)


def main(argv=None) -> int:
    from datetime import datetime

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recordings", type=Path, nargs="+", help="recording folders (their stills/ are used)")
    p.add_argument("--model", type=Path, required=True, help="weights of the trained pose detector")
    p.add_argument("--count", type=int, default=100, help="frames per recording (0 = all)")
    p.add_argument("--conf", type=float, default=0.10, help="low, so a doubtful box is drafted, not missing")
    p.add_argument("--out", type=Path, default=None, help="default: <BRAWL_DATA>/labelling/<date_time>")
    args = p.parse_args(argv)

    import cv2

    from brawl.sense.detectors import Yolo

    detector = Yolo(args.model, conf=args.conf)
    out = args.out or DATA / "labelling" / datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / "images").mkdir(parents=True, exist_ok=False)
    (out / "labels").mkdir(parents=True)
    total = 0
    for recording in args.recordings:
        stills = sorted((recording / "stills").glob("*.png"))
        if not stills:
            print(f"no stills in {recording}")
            return 1
        drafts = []
        for i, path in enumerate(stills):
            drafts.append(detector(cv2.imread(str(path))))
            print(f"\r{recording.name}: {i + 1}/{len(stills)}", end="", flush=True)
        sure = [sureness(d) for d in drafts]
        chosen = pick(sure, args.count)
        for i in chosen:
            name = f"{recording.name}__{stills[i].stem}"
            shutil.copy2(stills[i], out / "images" / f"{name}.png")
            (out / "labels" / f"{name}.txt").write_text("".join(line(b, 1920, 1080) + "\n" for b in drafts[i]),
                                                        encoding="utf-8")
        total += len(chosen)
        print(f"\n  {len(chosen)} of {len(stills)} stills; least sure fighter box: "
              f"median {np.median([sure[i] for i in chosen]):.2f} (all stills {np.median(sure):.2f})")
    (out / "data.yaml").write_text(f"train: images\nval: images\nnc: {len(CLASSES)}\nnames: {list(CLASSES)}\n"
                                   "kpt_shape: [1, 3]\nflip_idx: [0]\n", encoding="utf-8")
    print(f"{total} frames -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
