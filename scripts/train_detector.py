"""Train the detector on the labelled frames (docs/labelling.md); score it with gate P1 afterwards.

The detector is a pose model with one keypoint: box, class and point in one pass. A fighter's point
is the centre of its torso (its position); every other box's point is written at its centre.

Locally (builds the dataset from the Roboflow export and the lossless stills, then trains):

    python scripts/train_detector.py train                        # held weapon as classes (nine)
    python scripts/train_detector.py train --variant identity     # who only; held weapon by a crop classifier

On a bigger GPU (Kaggle: a 4 GB laptop GPU fits one 1280 px image per batch):

    python scripts/train_detector.py pack     # -> <BRAWL_DATA>/detector/brawl_detector_pack.zip

Upload the zip as a Kaggle dataset; in a GPU notebook (internet on):

    !pip install -q ultralytics==8.3.92
    !cd /kaggle/input/<dataset>/code && python scripts/train_detector.py train \\
        --pack /kaggle/input/<dataset> --out /kaggle/working --variant armed

and download /kaggle/working/runs/<name>/weights/best.pt into runs/detector/<name>/weights/.

The pack holds the training and validation frames with all nine classes, the code, and the
pretrained weights; the test split never leaves this machine. Augmentation is only what the game
can show: horizontal flips and small brightness changes; no rotation, no scale change, no mosaic
(the camera is fixed, so a fighter is always the same size).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brawl.sense.truth import CLASSES, Box, identity, read_yolo  # noqa: E402

VARIANTS = {
    "identity": ("wu_shang", "mordex", "weapon_spawn", "weapon_thrown", "offscreen"),
    "armed": CLASSES,
}
SIZE = (1920, 1080)
VALID_EVERY = 5  # the best epoch is chosen on validation frames: they must cover every session
CODE = ("brawl/__init__.py", "brawl/paths.py", "brawl/sense/__init__.py", "brawl/sense/truth.py",
        "brawl/sense/detectors.py", "scripts/train_detector.py")
SETTINGS = dict(fliplr=0.5, flipud=0.0, degrees=0.0, shear=0.0, perspective=0.0, scale=0.0, translate=0.05,
                mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.1,
                patience=40, seed=0, plots=True)


def frames_from_export(export: Path, demos: Path) -> list[tuple[str, str, Path, list[Box]]]:
    from brawl.sense.truth import problems, read_export

    frames = sorted(read_export(export, demos, splits=("train", "valid")), key=lambda f: (f.recording, f.row))
    for problem in problems(frames):
        print(f"label problem, fix in Roboflow: {problem}")
    validated = {f.recording for f in frames if f.split == "valid"}
    seen: dict[str, int] = {}
    out = []
    for f in frames:  # a session with no validation frames gives every fifth of its own (VALID_EVERY)
        seen[f.recording] = seen.get(f.recording, 0) + 1
        split = "valid" if f.recording not in validated and seen[f.recording] % VALID_EVERY == 0 else f.split
        out.append((split, f"{f.recording}__{f.row:07d}", f.still, f.boxes))
    return out


def frames_from_pack(pack: Path) -> list[tuple[str, str, Path, list[Box]]]:
    frames = []
    for split in ("train", "valid"):
        for image in sorted((pack / split / "images").glob("*.png")):
            boxes = read_yolo(pack / split / "labels" / f"{image.stem}.txt", list(CLASSES), *SIZE)
            frames.append((split, image.stem, image, boxes))
    return frames


def build(frames, variant: str, out: Path) -> dict[str, int]:
    """Write <out>/{train,valid}/{images,labels} and data.yaml for one variant; images are linked."""
    names = VARIANTS[variant]
    if out.exists():
        shutil.rmtree(out)
    counts: dict[str, int] = {}
    for split, stem, image, boxes in frames:
        images, labels = out / split / "images", out / split / "labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        try:
            (os.symlink if os.name != "nt" else os.link)(image, images / f"{stem}.png")
        except OSError:
            shutil.copy2(image, images / f"{stem}.png")
        lines = []
        for b in boxes:
            cls = (identity(b.cls) or b.cls) if variant == "identity" else b.cls
            px, py = b.position if identity(b.cls) else b.centre  # docs/labelling.md rule 10b
            lines.append(f"{names.index(cls)} {(b.x0 + b.x1) / 2 / SIZE[0]:.6f} {(b.y0 + b.y1) / 2 / SIZE[1]:.6f} "
                         f"{(b.x1 - b.x0) / SIZE[0]:.6f} {(b.y1 - b.y0) / SIZE[1]:.6f} "
                         f"{px / SIZE[0]:.6f} {py / SIZE[1]:.6f} 2")
        (labels / f"{stem}.txt").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        counts[split] = counts.get(split, 0) + 1
    (out / "data.yaml").write_text(f"path: {out.as_posix()}\ntrain: train/images\nval: valid/images\n"
                                   f"names: {list(names)}\nkpt_shape: [1, 3]\nflip_idx: [0]\n",
                                   encoding="utf-8")
    return counts


def pack(export: Path, demos: Path, out: Path, pretrained: Path) -> Path:
    """One zip: the frames with all nine classes, the code, the pretrained weights."""
    staging = out / "pack"
    counts = build(frames_from_export(export, demos), "armed", staging)
    zip_path = out / "brawl_detector_pack.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as z:  # PNGs do not compress
        for f in sorted(staging.rglob("*")):
            if f.is_file() and f.name != "data.yaml":
                z.write(f, f.relative_to(staging).as_posix())
        for rel in CODE:
            z.write(ROOT / rel, f"code/{rel}")
        if pretrained.exists():
            z.write(pretrained, pretrained.name)
    print(f"pack: {counts} -> {zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB)")
    return zip_path


def train(frames, variant: str, out: Path, model: str, imgsz: int, epochs: int, batch: float, device: str) -> Path:
    if device.isdigit():  # pin the GPU before anything starts CUDA, or every process lands on GPU 0
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        device = "0"
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    from ultralytics import YOLO

    from brawl.sense.detectors import cuda_nms_fallback

    name = f"{variant}_{Path(model).stem}_{imgsz}"
    data = out / "data" / name  # one dataset folder per run: two runs started together must not share it
    print(f"dataset {data}: {build(frames, variant, data)}")
    cuda_nms_fallback()
    YOLO(model).train(data=str(data / "data.yaml"), imgsz=imgsz, epochs=epochs,
                      batch=batch if batch < 1 else int(batch), device=device, workers=2,
                      project=str(out / "runs"), name=name, exist_ok=True, **SETTINGS)
    weights = out / "runs" / name / "weights" / "best.pt"
    print(f"\nweights: {weights}")
    return weights


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("train", "pack"))
    p.add_argument("--variant", choices=sorted(VARIANTS), default="armed")
    p.add_argument("--model", default="yolo11s-pose.pt", help="pretrained starting point (a pose model)")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=float, default=0.7, help="images per batch, or a fraction of GPU memory")
    p.add_argument("--device", default="0")
    p.add_argument("--pack", type=Path, default=None, help="train from an unpacked pack instead of the export")
    p.add_argument("--out", type=Path, default=None, help="default: <BRAWL_DATA>/detector (runs go to runs/)")
    args = p.parse_args(argv)

    if args.pack is not None:  # on the training machine: everything comes from the pack
        model = args.pack / args.model if (args.pack / args.model).exists() else args.model
        train(frames_from_pack(args.pack), args.variant, args.out or Path.cwd(), str(model),
              args.imgsz, args.epochs, args.batch, args.device)
        return 0
    from brawl.paths import DATA, RUNS

    export, demos = DATA / "labelling" / "export_kp_final", DATA / "demos"
    out = args.out or DATA / "detector"
    if args.command == "pack":
        pack(export, demos, out, DATA / args.model)
        return 0
    weights = train(frames_from_export(export, demos), args.variant, out, args.model, args.imgsz,
                    args.epochs, args.batch, args.device)
    dest = RUNS / "detector" / weights.parents[1].name / "weights"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, dest / "best.pt")
    print(f"next: python scripts/gate_p1.py {dest / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
