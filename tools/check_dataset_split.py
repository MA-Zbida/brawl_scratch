#!/usr/bin/env python
"""Check a Roboflow-exported dataset for split leakage and augmentation inflation.

Roboflow names exported files ``<source-stem>_<ext>.rf.<hash>.<ext>``, so every augmented
copy of a frame keeps its source stem and differs only in the hash. That makes two things
directly measurable:

**Leakage.** If augmentation ran before the split, augmented variants of one source frame
land in different splits. Validation then measures memorisation of a frame the model
already trained on, and every metric derived from it is optimistic. This is the failure
mode that matters, because it is invisible in the numbers -- it makes them *better*.

**Inflation.** How many copies of each source frame exist per split. Val and test should
normally be un-augmented (multiplier 1.0); a multiplier above 1 there means the model is
being scored on synthetic images rather than on frames resembling live gameplay.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}

# "myframe_png.rf.0123abcd....png" -> "myframe"
_ROBOFLOW = re.compile(r"^(?P<stem>.+?)_(?:png|jpg|jpeg|bmp)\.rf\.[0-9a-f]+$", re.IGNORECASE)


def source_stem(path: Path) -> str:
    """Strip the Roboflow suffix to recover the original frame identity."""
    m = _ROBOFLOW.match(path.stem)
    return m.group("stem") if m else path.stem


def collect(root: Path, splits: Sequence[str], images_subdir: str) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for split in splits:
        d = root / split / images_subdir
        found[split] = sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT) if d.exists() else []
    return found


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="Dataset root containing the split directories")
    p.add_argument("--splits", type=str, default="train,valid,test")
    p.add_argument("--images-subdir", type=str, default="images")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if not args.root.exists():
        print(f"No such directory: {args.root}", file=sys.stderr)
        return 1

    images = collect(args.root, splits, args.images_subdir)
    if not any(images.values()):
        print(f"No images found under {args.root}", file=sys.stderr)
        return 1

    print("=" * 72)
    print("DATASET SPLIT CHECK")
    print("=" * 72)

    sources: dict[str, set[str]] = collections.defaultdict(set)
    print(f"{'split':<10}{'files':>8}{'sources':>10}{'copies/source':>16}{'augmented':>12}")
    for split in splits:
        stems = [source_stem(p) for p in images[split]]
        uniq = set(stems)
        for s in uniq:
            sources[s].add(split)
        n, u = len(stems), len(uniq)
        mult = (n / u) if u else 0.0
        print(f"{split:<10}{n:>8}{u:>10}{mult:>16.2f}{'YES' if mult > 1.05 else 'no':>12}")

    # ── leakage ──────────────────────────────────────────────────────────────
    leaked = {s: sorted(sp) for s, sp in sources.items() if len(sp) > 1}
    print(f"\nsource frames appearing in more than one split: {len(leaked)}")
    for s, sp in list(leaked.items())[:20]:
        print(f"  {s}  ->  {', '.join(sp)}")
    if len(leaked) > 20:
        print(f"  ... and {len(leaked) - 20} more")

    print()
    if leaked:
        print("*** LEAKAGE ***")
        print("The same source frame appears in multiple splits, so validation is scored on")
        print("frames the model trained on. Every metric from that split is optimistic.")
        print("Fix: in Roboflow, split FIRST, then apply augmentation to train only.")
    else:
        print("No source frame spans two splits.")

    # ── augmented eval splits ────────────────────────────────────────────────
    warned = False
    for split in splits[1:]:
        stems = [source_stem(p) for p in images[split]]
        uniq = set(stems)
        if uniq and len(stems) / len(uniq) > 1.05:
            warned = True
            print(f"\nNOTE: '{split}' is augmented ({len(stems)/len(uniq):.2f} copies per source).")
            print("      It is being scored on synthetic images rather than on frames that")
            print("      resemble live gameplay. Prefer un-augmented val/test.")
    if not warned:
        print("Evaluation splits are un-augmented.")

    if any(source_stem(p) == p.stem for split in splits for p in images[split][:5]):
        print("\n(Filenames do not look Roboflow-exported; leakage grouping may be unreliable.)")

    return 2 if leaked else 0


if __name__ == "__main__":
    raise SystemExit(main())
