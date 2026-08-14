#!/usr/bin/env python
"""Train the armed / unarmed classifier over character crops.

Weapon state is currently inferred from the agent's own key presses, which is only
correct if pickup and death are the sole transitions -- knockback disarms break it,
and for the opponent there is no action stream at all. This measures it instead.

Two choices here are driven by the size of the dataset rather than convention:

**Grouped cross-validation, not a single hold-out.** At ~300 crops a 20% validation
split is ~60 samples, so an accuracy estimate carries roughly +/-6% of noise -- wide
enough that you could not tell 90% from 96%. K-fold uses every sample for both
training and evaluation and reports the spread, and it yields an out-of-fold
prediction for *every* crop, so the confusion matrix covers the whole dataset and
the misclassified list is complete rather than a sample.

**Folds are split by source frame.** Both fighters in one frame share a background,
a moment and a compression artefact. Putting one in train and the other in
validation leaks, and the reported accuracy would flatter the model.

Augmentation matters more than capacity here. The crops come from clean annotated
frames, but at inference the model sees live DXCam captures, so the training-time
jitter deliberately includes the noise and brightness variation that separates the
two. Horizontal flips are safe and free: weapon state is mirror-invariant, and the
observation pipeline canonicalises by mirroring anyway.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: Must match the padding used when the crops were cut, or the model sees a
#: different framing at inference than it trained on. Recorded in the checkpoint.
DEFAULT_PAD = 0.45

CLASSES = ("unarmed", "armed")


def load_dataset(root: Path) -> tuple[list[Path], np.ndarray, np.ndarray]:
    """Return crop paths, labels, and a group id per crop (the source frame)."""
    paths: list[Path] = []
    labels: list[int] = []
    for index, name in enumerate(CLASSES):
        for path in sorted((root / name).glob("*.png")):
            paths.append(path)
            labels.append(index)
    if not paths:
        raise FileNotFoundError(f"No crops under {root}/{{{','.join(CLASSES)}}}")

    # `<frame>__<n>.png` -- strip the character index to recover the source frame.
    frames = sorted({p.stem.rsplit("__", 1)[0] for p in paths})
    frame_id = {name: i for i, name in enumerate(frames)}
    groups = np.asarray([frame_id[p.stem.rsplit("__", 1)[0]] for p in paths])
    return paths, np.asarray(labels, dtype=np.int64), groups


def build_model(torch, nn, backbone: str = "small"):
    """Build the classifier.

    `resnet18` fine-tunes ImageNet features. With only a few hundred crops that is
    normally a large win over training from scratch: the early filters that separate
    a held sword from an empty hand are edge and texture detectors the pretrained
    network already has, and 300 samples is nowhere near enough to learn them.
    """
    if backbone == "resnet18":
        from torchvision import models

        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        net.fc = nn.Linear(net.fc.in_features, 2)
        return net

    def block(cin: int, cout: int):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        block(3, 24), block(24, 48), block(48, 96),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(96, 2),
    )


def augment(torch, batch: "torch.Tensor") -> "torch.Tensor":
    """Flip, brightness/contrast jitter and noise, applied on-device per batch."""
    n = batch.shape[0]
    flip = torch.rand(n, device=batch.device) < 0.5
    batch = torch.where(flip.view(-1, 1, 1, 1), batch.flip(-1), batch)

    gain = 1.0 + (torch.rand(n, 1, 1, 1, device=batch.device) - 0.5) * 0.5
    bias = (torch.rand(n, 1, 1, 1, device=batch.device) - 0.5) * 0.3
    batch = batch * gain + bias
    # Live captures are noisier than annotated frames; train through it.
    batch = batch + torch.randn_like(batch) * 0.05
    return batch.clamp(0.0, 1.0)


def load_images(cv2, paths: Sequence[Path], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    out = np.zeros((len(paths), height, width, 3), dtype=np.float32)
    for i, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            continue
        out[i] = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) / 255.0
    return out.transpose(0, 3, 1, 2)          # NCHW


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/weapon_crops"))
    p.add_argument("--out", type=Path, default=Path("train/models/weapon_classifier.pt"))
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=96, help="Median crop aspect is ~1.65 h/w.")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backbone", choices=["small", "resnet18"], default="small")
    p.add_argument("--pad", type=float, default=DEFAULT_PAD)
    return p.parse_args(argv)


def _train_fold(torch, nn, x_tr, y_tr, x_va, y_va, args, device, weight):
    model = build_model(torch, nn, args.backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(args.epochs))
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    n = x_tr.shape[0]
    for _ in range(int(args.epochs)):
        model.train()
        order = torch.randperm(n, device=device)
        for start in range(0, n, int(args.batch_size)):
            idx = order[start: start + int(args.batch_size)]
            xb = augment(torch, x_tr[idx])
            loss = loss_fn(model(xb), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_va)
        probs = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)
    return model, pred.cpu().numpy(), probs.cpu().numpy()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    import cv2
    import torch
    from torch import nn

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    paths, labels, groups = load_dataset(args.data)
    counts = Counter(labels.tolist())
    print("=" * 70)
    print(f"{len(paths)} crops from {len(set(groups.tolist()))} source frames")
    print(f"  unarmed {counts.get(0, 0)}   armed {counts.get(1, 0)}"
          f"   minority {min(counts.values()) / len(labels):.1%}")
    print(f"  input {args.width}x{args.height}   device {device}   backbone {args.backbone}")
    print("=" * 70)

    images = load_images(cv2, paths, (int(args.width), int(args.height)))
    x_all = torch.from_numpy(images).to(device)
    y_all = torch.from_numpy(labels).to(device)

    freq = np.bincount(labels, minlength=2).astype(np.float32)
    weight = torch.from_numpy((freq.sum() / (2.0 * np.maximum(freq, 1))).astype(np.float32)).to(device)

    # Grouped folds: every crop from one frame lands in the same fold.
    unique = np.unique(groups)
    rng = np.random.default_rng(int(args.seed))
    rng.shuffle(unique)
    fold_of_group = {g: i % max(2, int(args.folds)) for i, g in enumerate(unique)}
    fold = np.asarray([fold_of_group[g] for g in groups])

    oof_pred = np.full(len(labels), -1, dtype=np.int64)
    oof_prob = np.zeros(len(labels), dtype=np.float32)
    accuracies: list[float] = []

    for k in range(max(2, int(args.folds))):
        va = fold == k
        tr = ~va
        if va.sum() == 0 or tr.sum() == 0:
            continue
        _model, pred, prob = _train_fold(
            torch, nn, x_all[tr], y_all[tr], x_all[va], y_all[va], args, device, weight)
        oof_pred[va] = pred
        oof_prob[va] = prob
        acc = float((pred == labels[va]).mean())
        accuracies.append(acc)
        print(f"  fold {k}: train {int(tr.sum()):4d}  val {int(va.sum()):3d}  acc {acc:.3f}")

    scored = oof_pred >= 0
    overall = float((oof_pred[scored] == labels[scored]).mean())
    print("\n" + "=" * 70)
    print(f"cross-validated accuracy: {np.mean(accuracies):.3f} +/- {np.std(accuracies):.3f}"
          f"   (out-of-fold over all {int(scored.sum())} crops: {overall:.3f})")

    print("\nconfusion matrix (rows = truth, cols = predicted):")
    print(f"  {'':<10}{'unarmed':>10}{'armed':>10}")
    for t, name in enumerate(CLASSES):
        row = [int(((labels == t) & (oof_pred == p) & scored).sum()) for p in range(2)]
        print(f"  {name:<10}{row[0]:>10}{row[1]:>10}")

    for t, name in enumerate(CLASSES):
        tp = int(((labels == t) & (oof_pred == t)).sum())
        fp = int(((labels != t) & (oof_pred == t)).sum())
        fn = int(((labels == t) & (oof_pred != t)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        print(f"  {name:<10} precision {prec:.3f}  recall {rec:.3f}")

    wrong = np.flatnonzero(scored & (oof_pred != labels))
    if wrong.size:
        print(f"\n{wrong.size} misclassified crops (inspect these before labelling more):")
        for i in wrong[np.argsort(-np.abs(oof_prob[wrong] - 0.5))][:15]:
            print(f"  {CLASSES[labels[i]]:>8} -> {CLASSES[oof_pred[i]]:<8} p(armed)={oof_prob[i]:.2f}  {paths[i].name}")

    # Final model on everything: the fold models measured, this one ships.
    print("\ntraining final model on all crops...")
    model, _, _ = _train_fold(torch, nn, x_all, y_all, x_all[:1], y_all[:1], args, device, weight)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "classes": list(CLASSES),
        # The inference-time contract. Cropping or resizing differently than this
        # silently degrades the model with no error anywhere.
        "input_width": int(args.width),
        "input_height": int(args.height),
        "crop_pad": float(args.pad),
        "cv_accuracy": float(np.mean(accuracies)),
        "cv_std": float(np.std(accuracies)),
        "n_train": int(len(labels)),
        "backbone": str(args.backbone),
    }, args.out)
    print(f"saved {args.out}")

    if np.mean(accuracies) < 0.90:
        print("\nUnder 90%: label more crops, and look at the misclassified list first --")
        print("if they cluster on one situation, targeted labelling beats volume.")
    else:
        print(f"\n{np.mean(accuracies):.1%} cross-validated. Check the misclassified list; if those")
        print("cases matter in play, label more of them specifically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
