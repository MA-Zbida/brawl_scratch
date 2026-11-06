"""Draft boxes for the ground-truth set, to be corrected in Roboflow (docs/labelling.md).

The ground truth has three classes: ``wu_shang``, ``mordex`` (body boxes) and ``weapon`` (weapons
not held by a fighter). Drawing them from nothing is slow, so this runs the old detector
(assets/yolo/best.pt: character, indicator_self, weapon) on full-resolution stills and writes its
boxes as drafts. The character under the ``indicator_self`` triangle is Wu Shang; the most
confident other character is Mordex. Drafts are generous (low confidence): deleting a wrong box
is quicker than drawing a missing one.

    python scripts/prelabel.py data/demos/<easy recording> data/demos/<hard recording> --count 75

Writes data/labelling/<date_time>/ with images/, labels/ and data.yaml: upload that whole folder
to a Roboflow object-detection project, correct every image by the rules in docs/labelling.md,
export YOLO with no preprocessing and no augmentation, and unzip the export there as export/.
Images are named <recording>__<row>.png, so each label traces back to its recording and row.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from brawl.paths import ASSETS, DATA  # noqa: E402

CLASSES = ("wu_shang", "mordex", "weapon")


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


def free_weapons(weapons: list[list[float]], fighters: list[list[float]]) -> list[list[float]]:
    """Weapon detections whose centre is not inside a fighter box (a held weapon is not labelled)."""
    def inside(w: list[float], f: list[float]) -> bool:
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        return f[0] <= cx <= f[2] and f[1] <= cy <= f[3]

    return [w for w in weapons if not any(inside(w, f) for f in fighters)]


def main(argv=None) -> int:
    from datetime import datetime

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recordings", type=Path, nargs="+", help="recording folders (their stills/ are used)")
    p.add_argument("--count", type=int, default=75, help="frames per recording, spread evenly in time (0 = all)")
    p.add_argument("--conf", type=float, default=0.10, help="detector confidence for drafts")
    p.add_argument("--model", type=Path, default=ASSETS / "yolo" / "best.pt")
    p.add_argument("--out", type=Path, default=None, help="default: data/labelling/<date_time>")
    args = p.parse_args(argv)
    chosen: list[tuple[str, Path]] = []
    for recording in args.recordings:
        source = recording / "stills" if (recording / "stills").is_dir() else recording
        files = pick(sorted(source.glob("*.png")), args.count)
        if not files:
            print(f"no PNG stills in {source}")
            return 1
        chosen += [(f"{recording.name}__{f.stem}", f) for f in files]
    out = args.out or DATA / "labelling" / datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / "images").mkdir(parents=True, exist_ok=False)
    (out / "labels").mkdir(parents=True)

    import os

    os.environ.setdefault("YOLO_AUTOINSTALL", "False")  # never let ultralytics install packages
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(args.model), task="detect")
    counts = {"fighters": [0, 0, 0], "no_indicator": 0, "weapons": 0}
    for i, (name, path) in enumerate(chosen):
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        result = model.predict(image, imgsz=960, conf=args.conf, verbose=False, device="cpu")[0]
        boxes = [[*b, s] for b, s in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())]
        classes = [model.names[int(c)] for c in result.boxes.cls.tolist()]
        of = {k: [b for b, c in zip(boxes, classes) if c == k] for k in ("character", "indicator_self", "weapon")}
        named = assign(of["character"], of["indicator_self"])
        named += [("weapon", b) for b in free_weapons(of["weapon"], [b for _, b in named])]
        fighters = sum(c != "weapon" for c, _ in named)
        counts["fighters"][fighters] += 1
        counts["no_indicator"] += int(not of["indicator_self"])
        counts["weapons"] += len(named) - fighters
        shutil.copy2(path, out / "images" / f"{name}.png")
        (out / "labels" / f"{name}.txt").write_text(
            "".join(yolo_line(c, b, w, h) + "\n" for c, b in named), encoding="utf-8")
        print(f"\r{i + 1}/{len(chosen)}", end="", flush=True)
    (out / "data.yaml").write_text(f"names: {list(CLASSES)}\nnc: {len(CLASSES)}\n"
                                   "train: images\nval: images\n", encoding="utf-8")
    f0, f1, f2 = counts["fighters"]
    print(f"\n{len(chosen)} frames -> {out}")
    print(f"drafts with 0 / 1 / 2 fighters: {f0} / {f1} / {f2}; no indicator found: {counts['no_indicator']}; "
          f"free weapons drafted: {counts['weapons']}")
    print("Next (docs/labelling.md): Roboflow object-detection project with classes wu_shang, mordex, weapon; "
          "upload this folder; correct every image; export YOLO, no preprocessing, no augmentation, into export/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
