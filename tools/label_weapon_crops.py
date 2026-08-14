#!/usr/bin/env python
"""Hand-label character crops as armed / unarmed, for the weapon-state classifier.

Weapon state is currently inferred from the agent's own actions: pressing pickup
sets the flag, pressing it again or dying clears it. That is only sound if pickup
and death are the *only* transitions, and they are not -- knockback disarms you.
A live match showed 43 damage taken with a single drop transition recorded, so the
flag goes stale. And for the opponent there is no action stream to infer from at
all, which leaves `opponent_weapon_type` a placeholder.

A classifier over the character crop measures the thing directly instead, for both
fighters, with no assumptions about how the weapon was lost.

Workflow: the script writes the current crop to a fixed path (default `cropped.png`)
and waits for a keypress. Keep that file open in the editor -- it refreshes in place
as you label, so you never touch the file picker.

    0  unarmed        s  skip this crop
    1  armed          u  undo the previous label
                      q  save and quit

Progress is written after every label, so quitting at any point loses nothing and
re-running resumes exactly where you stopped.

Output, ready for `torchvision.datasets.ImageFolder`::

    <out>/unarmed/<image>__<n>.png
    <out>/armed/<image>__<n>.png
    <out>/labels.jsonl        one record per decision, with source box

Crops are drawn from the YOLO dataset in image order *shuffled* under a fixed seed:
consecutive frames of one match are near-duplicates, so labelling them in sequence
would spend an hour on a handful of distinct situations.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

#: YOLO class id for a fighter in this dataset (nc=3: character, indicator_self, weapon).
CHARACTER_CLASS = 0

LABEL_NAMES = {0: "unarmed", 1: "armed"}


@dataclass(frozen=True)
class Crop:
    image_path: Path
    index: int
    cx: float
    cy: float
    w: float
    h: float

    @property
    def key(self) -> str:
        return f"{self.image_path.stem}__{self.index}"


#: Roboflow export naming: `<source>_png.rf.<hash>.jpg`. Augmented copies of one
#: source frame share everything before `.rf.`.
_RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+\..*$", re.IGNORECASE)


def source_key(image_path: Path) -> str:
    """Strip the Roboflow hash so augmented copies of one frame group together."""
    return _RF_SUFFIX.sub("", image_path.name)


def noise_level(cv2, image_path: Path) -> float:
    """Residual left by a 3x3 median filter -- an augmentation-noise estimate.

    Measured on this dataset: every group of three train copies contains exactly
    one image scoring 0.00 (the un-augmented original) and two scoring 4-19. The
    valid and test splits score 0.00 throughout, confirming they are not augmented.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return float("inf")
    import numpy as _np

    residual = _np.abs(image.astype(_np.float32) - cv2.medianBlur(image, 3).astype(_np.float32))
    return float(_np.median(residual))


def select_originals(cv2, images: Sequence[Path], cache_path: Optional[Path] = None) -> list[Path]:
    """Keep one image per source frame: the least-noisy, i.e. the un-augmented one.

    Heavy augmentation makes a crop genuinely unlabelable -- colour-shifted noise
    over a 70x180 sprite hides whether a weapon is held. Labelling those wastes the
    only expensive resource here, which is human attention.
    """
    cached: dict[str, float] = {}
    if cache_path and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cached = {}

    groups: dict[str, list[Path]] = {}
    for path in images:
        groups.setdefault(source_key(path), []).append(path)

    chosen: list[Path] = []
    for key in sorted(groups):
        members = groups[key]
        if len(members) == 1:
            chosen.append(members[0])
            continue
        scored = []
        for path in members:
            name = path.name
            if name not in cached:
                cached[name] = noise_level(cv2, path)
            scored.append((cached[name], name, path))
        scored.sort()
        chosen.append(scored[0][2])

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached), encoding="utf-8")
    return chosen


def iter_crops(dataset_root: Path, splits: Sequence[str],
               keep: Optional[set] = None) -> Iterator[Crop]:
    for split in splits:
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.glob("*.*")):
            if keep is not None and image_path not in keep:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            try:
                lines = label_path.read_text(encoding="utf-8").strip().splitlines()
            except Exception:
                continue
            n = 0
            for line in lines:
                parts = line.split()
                if len(parts) < 5 or int(float(parts[0])) != CHARACTER_CLASS:
                    continue
                cx, cy, w, h = (float(v) for v in parts[1:5])
                yield Crop(image_path, n, cx, cy, w, h)
                n += 1


def crop_pixels(image, box: Crop, pad: float):
    """Cut the padded box out of the frame, clamped to the image.

    Padding is generous by default: a held weapon frequently extends well past the
    fighter's own bounding box, and a tight crop would hide the very thing being
    labelled.
    """
    height, width = image.shape[:2]
    half_w = (box.w * (1.0 + pad)) / 2.0
    half_h = (box.h * (1.0 + pad)) / 2.0
    x0 = max(0, int((box.cx - half_w) * width))
    x1 = min(width, int((box.cx + half_w) * width))
    y0 = max(0, int((box.cy - half_h) * height))
    y1 = min(height, int((box.cy + half_h) * height))
    if x1 <= x0 or y1 <= y0:
        return None
    return image[y0:y1, x0:x1]


def write_preview(cv2, patch, path: Path, scale: int) -> None:
    """Write the preview atomically so the editor never reads a half-written file."""
    height, width = patch.shape[:2]
    big = cv2.resize(patch, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
    tmp = path.with_suffix(".tmp.png")
    cv2.imwrite(str(tmp), big)
    os.replace(tmp, path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=Path("brawlhalla-yolo-data"))
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    p.add_argument("--out", type=Path, default=Path("data/weapon_crops"))
    p.add_argument("--preview", type=Path, default=Path("cropped.png"),
                   help="Fixed path rewritten for each crop. Keep it open in the editor.")
    p.add_argument("--pad", type=float, default=0.45,
                   help="Fractional padding around the box; a held weapon extends past it.")
    p.add_argument("--scale", type=int, default=4, help="Preview upscale factor.")
    p.add_argument("--limit", type=int, default=0, help="Stop after N new labels (0 = no limit).")
    p.add_argument("--include-augmented", action="store_true",
                   help="Also offer Roboflow augmented copies. Off by default: heavy "
                        "noise over a small sprite makes weapon state unreadable.")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    import cv2

    if not args.dataset.is_dir():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    out: Path = args.out
    for name in LABEL_NAMES.values():
        (out / name).mkdir(parents=True, exist_ok=True)
    ledger = out / "labels.jsonl"

    done: dict[str, int] = {}
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("label") is None:
                done[rec["key"]] = -1          # explicitly skipped
            else:
                done[rec["key"]] = int(rec["label"])

    keep = None
    if not args.include_augmented:
        every = [p for s_ in args.splits for p in sorted((args.dataset / s_ / 'images').glob('*.*'))
                 if (args.dataset / s_ / 'images').is_dir()]
        print(f"filtering augmented copies out of {len(every)} images...")
        originals = select_originals(cv2, every, cache_path=out / 'noise_cache.json')
        keep = set(originals)
        print(f"  kept {len(keep)} un-augmented source frames")
    crops = list(iter_crops(args.dataset, args.splits, keep=keep))
    random.Random(int(args.seed)).shuffle(crops)
    todo = [c for c in crops if c.key not in done]

    labelled = sum(1 for v in done.values() if v >= 0)
    counts = {0: sum(1 for v in done.values() if v == 0), 1: sum(1 for v in done.values() if v == 1)}

    print("=" * 68)
    print(f"{len(crops)} character crops in {args.dataset}")
    print(f"already labelled: {labelled}  (unarmed {counts[0]}, armed {counts[1]})"
          f"   skipped: {sum(1 for v in done.values() if v < 0)}")
    print(f"remaining: {len(todo)}")
    print(f"\nOpen {args.preview} in your editor -- it refreshes in place.")
    print("  0 = unarmed   1 = armed   s = skip   u = undo   q = save and quit")
    print("=" * 68)

    if not todo:
        print("Nothing left to label.")
        return 0

    history: list[Crop] = []
    session = 0
    i = 0

    def record(crop: Crop, label: Optional[int], patch) -> None:
        if label is not None:
            dest = out / LABEL_NAMES[label] / f"{crop.key}.png"
            cv2.imwrite(str(dest), patch)
        with ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "key": crop.key,
                "label": label,
                "image": str(crop.image_path),
                "box": [crop.cx, crop.cy, crop.w, crop.h],
            }) + "\n")

    while i < len(todo):
        crop = todo[i]
        image = cv2.imread(str(crop.image_path))
        if image is None:
            i += 1
            continue
        patch = crop_pixels(image, crop, float(args.pad))
        if patch is None or patch.size == 0:
            i += 1
            continue

        write_preview(cv2, patch, args.preview, max(1, int(args.scale)))
        prompt = (f"[{session + 1}{'/' + str(args.limit) if args.limit else ''}] "
                  f"{crop.key}  (u {counts[0]} / a {counts[1]})  0/1/s/u/q > ")
        try:
            answer = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nStopping.")
            break

        if answer == "q":
            break
        if answer == "u":
            if history:
                undone = history.pop()
                # Rewrite the ledger without the undone record, and drop its file.
                kept = [ln for ln in ledger.read_text(encoding="utf-8").splitlines()
                        if json.loads(ln)["key"] != undone.key]
                ledger.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
                for name in LABEL_NAMES.values():
                    stale = out / name / f"{undone.key}.png"
                    if stale.exists():
                        stale.unlink()
                        counts[0 if name == "unarmed" else 1] -= 1
                i -= 1
                session = max(0, session - 1)
                print(f"  undid {undone.key}")
            else:
                print("  nothing to undo")
            continue
        if answer == "s":
            record(crop, None, patch)
            history.append(crop)
            i += 1
            continue
        if answer not in ("0", "1"):
            print("  expected 0, 1, s, u or q")
            continue

        label = int(answer)
        record(crop, label, patch)
        counts[label] += 1
        history.append(crop)
        session += 1
        i += 1
        if args.limit and session >= int(args.limit):
            print(f"\nReached --limit {args.limit}.")
            break

    total = counts[0] + counts[1]
    print("\n" + "=" * 68)
    print(f"labelled this session: {session}")
    print(f"totals: unarmed {counts[0]}, armed {counts[1]}, total {total}")
    if total:
        minority = min(counts.values()) / total
        print(f"class balance: {minority:.1%} minority class")
        if minority < 0.25:
            print("  Imbalanced. A classifier will lean on the prior; label more of the")
            print("  rarer class before training, or weight the loss.")
    print(f"crops under {out}/, ledger at {ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
