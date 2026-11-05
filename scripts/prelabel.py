"""Draft fighter boxes for labelling in Roboflow (research/DESIGN.md section 9, gate P1).

The ground truth is two boxes per frame, ``wu_shang`` and ``mordex``. Drawing them from nothing
is slow, so this runs the old detector (assets/yolo/best.pt: classes character, indicator_self,
weapon) on full-resolution stills and writes its boxes as drafts. The character under the
``indicator_self`` triangle is Wu Shang; any other character is Mordex. You upload the folder to
Roboflow, correct the drafts there (zoom, shortcuts), and export the result (YOLO format, no
augmentation, no resize) for perception to be measured against.

    python scripts/prelabel.py data/demos/<recording> --count 150

Writes <recording>/roboflow/{images,labels}/ and data.yaml: drag that folder into a new Roboflow
object-detection project. Drafts are deliberately generous (low confidence): deleting a wrong box
is quicker than drawing a missing one.

Box rule when correcting: the fighter's body, head to feet, tight; no effects or weapon swing;
partly off screen, box the visible part; not visible at all, no box; a menu on screen, delete the
image.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from brawl.paths import ASSETS  # noqa: E402

CLASSES = ("wu_shang", "mordex")


def assign(characters: list[list[float]], indicators: list[list[float]]) -> list[tuple[str, list[float]]]:
    """Name the character boxes. Wu Shang is the character whose top is nearest below the
    indicator triangle (it floats just above the agent's head); the most confident other one is
    Mordex; the rest are duplicates and dropped."""
    named = [("mordex", box) for box in characters]
    if indicators and characters:
        ix0, iy0, ix1, iy1 = max(indicators, key=lambda b: b[4] if len(b) > 4 else 0)[:4]
        cx, bottom = (ix0 + ix1) / 2, iy1

        def distance(box: list[float]) -> float:
            x0, y0, x1, _ = box[:4]
            horizontal = 0.0 if x0 <= cx <= x1 else min(abs(cx - x0), abs(cx - x1))
            return horizontal + abs(y0 - bottom)

        agent = min(range(len(characters)), key=lambda i: distance(characters[i]))
        named[agent] = ("wu_shang", characters[agent])
    # Only two fighters exist: keep the most confident other character, drop duplicates.
    others = [n for n in named if n[0] == "mordex"]
    best = [max(others, key=lambda n: n[1][4] if len(n[1]) > 4 else 0)] if others else []
    return [n for n in named if n[0] == "wu_shang"] + best


def yolo_line(cls: str, box: list[float], width: int, height: int) -> str:
    x0, y0, x1, y1 = box[:4]
    return (f"{CLASSES.index(cls)} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
            f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}")


def pick(files: list[Path], count: int) -> list[Path]:
    """``count`` frames spread evenly over the recording."""
    if count <= 0 or count >= len(files):
        return files
    return [files[i] for i in np.linspace(0, len(files) - 1, count).round().astype(int)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording", type=Path, help="a recording folder (uses its stills/) or a folder of PNGs")
    p.add_argument("--count", type=int, default=150, help="frames to draft, spread over the recording (0 = all)")
    p.add_argument("--conf", type=float, default=0.10, help="detector confidence for drafts")
    p.add_argument("--model", type=Path, default=ASSETS / "yolo" / "best.pt")
    args = p.parse_args(argv)
    source = args.recording / "stills" if (args.recording / "stills").is_dir() else args.recording
    files = pick(sorted(source.glob("*.png")), args.count)
    if not files:
        print(f"no PNG frames in {source}")
        return 1
    out = args.recording / "roboflow"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    import os

    os.environ.setdefault("YOLO_AUTOINSTALL", "False")  # never let ultralytics install packages
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(args.model), task="detect")
    counts = {0: 0, 1: 0, 2: 0, "more": 0, "no_indicator": 0}
    for i, path in enumerate(files):
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        result = model.predict(image, imgsz=960, conf=args.conf, verbose=False, device="cpu")[0]
        boxes = [[*b, s] for b, s in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())]
        classes = [model.names[int(c)] for c in result.boxes.cls.tolist()]
        characters = [b for b, c in zip(boxes, classes) if c == "character"]
        indicators = [b for b, c in zip(boxes, classes) if c == "indicator_self"]
        named = assign(characters, indicators)
        counts[len(named) if len(named) <= 2 else "more"] += 1
        counts["no_indicator"] += int(not indicators)
        shutil.copy2(path, out / "images" / path.name)
        (out / "labels" / f"{path.stem}.txt").write_text(
            "".join(yolo_line(c, b, w, h) + "\n" for c, b in named), encoding="utf-8")
        print(f"\r{i + 1}/{len(files)}", end="", flush=True)
    (out / "data.yaml").write_text(f"names: {list(CLASSES)}\nnc: {len(CLASSES)}\n"
                                   "train: images\nval: images\n", encoding="utf-8")
    print(f"\n{len(files)} frames -> {out}")
    print(f"drafts with 0 / 1 / 2 / more fighters: {counts[0]} / {counts[1]} / {counts[2]} / {counts['more']}; "
          f"no indicator found: {counts['no_indicator']}")
    print("Next: Roboflow > new object-detection project > Upload > drag this folder; correct; "
          "export as YOLO (no augmentation, no resize).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
