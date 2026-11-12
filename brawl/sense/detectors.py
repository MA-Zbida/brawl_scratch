"""Detector candidates for gate P1, each returning boxes in the labelled classes (brawl.sense.truth).

``reports_held`` says whether a candidate reads the held weapon; when it does not, its fighter boxes
carry the ``*_unarmed`` class only as a placeholder and P1 does not score the held weapon.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from brawl.paths import ASSETS
from brawl.sense.truth import Box, identity


def assign(characters: list[list[float]], indicators: list[list[float]]) -> list[tuple[str, list[float]]]:
    """Name the old detector's ``character`` boxes ([x0, y0, x1, y1, score]). Wu Shang is the
    character whose top is nearest below the indicator triangle (it floats just above the agent's
    head); the most confident other one is Mordex; the rest are duplicates and dropped. Without a
    triangle, no character is taken for the agent."""
    named = [("mordex_unarmed", box) for box in characters]
    if indicators and characters:
        ix0, iy0, ix1, iy1 = max(indicators, key=lambda b: b[4] if len(b) > 4 else 0)[:4]
        cx, bottom = (ix0 + ix1) / 2, iy1

        def distance(box: list[float]) -> float:
            x0, y0, x1, _ = box[:4]
            horizontal = 0.0 if x0 <= cx <= x1 else min(abs(cx - x0), abs(cx - x1))
            return horizontal + abs(y0 - bottom)

        agent = min(range(len(characters)), key=lambda i: distance(characters[i]))
        named[agent] = ("wu_shang_unarmed", characters[agent])
    others = [n for n in named if n[0] == "mordex_unarmed"]
    best = [max(others, key=lambda n: n[1][4] if len(n[1]) > 4 else 0)] if others else []
    return [n for n in named if n[0] == "wu_shang_unarmed"] + best


class OldYolo:
    """The detector of the earlier work (assets/yolo/best.pt: character, indicator_self, weapon).
    Its ``weapon`` class only ever boxed spawn markers; it has no thrown weapons or held weapon."""

    name = "old_yolo"
    reports_held = False

    def __init__(self, model: Path = ASSETS / "yolo" / "best.pt", conf: float = 0.25, imgsz: int = 960,
                 device: str = "cpu"):
        os.environ.setdefault("YOLO_AUTOINSTALL", "False")  # never let ultralytics install packages
        from ultralytics import YOLO

        self.model = YOLO(str(model), task="detect")
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def raw(self, image: np.ndarray) -> dict[str, list[list[float]]]:
        result = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, verbose=False, device=self.device)[0]
        boxes = [[*b, s] for b, s in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())]
        classes = [self.model.names[int(c)] for c in result.boxes.cls.tolist()]
        return {k: [b for b, c in zip(boxes, classes) if c == k] for k in ("character", "indicator_self", "weapon")}

    def __call__(self, image: np.ndarray) -> list[Box]:
        of = self.raw(image)
        named = assign(of["character"], of["indicator_self"]) + [("weapon_spawn", b) for b in of["weapon"]]
        return [Box(c, *b[:4], score=b[4]) for c, b in named]


def cuda_nms_fallback() -> None:
    """Run torchvision's NMS on the CPU when its CUDA kernel is missing (a CPU-only torchvision
    beside a CUDA torch); everything else stays on the GPU. A no-op when CUDA NMS works."""
    import torch
    import torchvision

    if not torch.cuda.is_available() or getattr(torchvision.ops.nms, "cpu_fallback", False):
        return
    try:
        torchvision.ops.nms(torch.zeros((1, 4), device="cuda"), torch.zeros(1, device="cuda"), 0.5)
        return
    except (NotImplementedError, RuntimeError):
        original = torchvision.ops.nms

    def nms(boxes, scores, iou_threshold):
        return original(boxes.cpu(), scores.cpu(), iou_threshold).to(boxes.device)

    nms.cpu_fallback = True
    torchvision.ops.nms = nms


class Yolo:
    """A detector trained on the labels (scripts/train_detector.py). Trained on identity classes
    (``wu_shang``, ``mordex``), it does not read the held weapon; trained on the armed classes, it does.
    A pose model also gives each box its point (the torso, for a fighter). Weights are a ``.pt``, or
    the TensorRT ``.engine`` built from it (scripts/export_detector.py), which fixes the input size."""

    reports_held = False

    def __init__(self, weights: Path, conf: float = 0.25, imgsz: int | None = None, device: str | None = None):
        os.environ.setdefault("YOLO_AUTOINSTALL", "False")
        import torch
        from ultralytics import YOLO

        cuda_nms_fallback()
        weights = Path(weights)
        self.name = weights.parents[1].name if weights.parent.name == "weights" else weights.stem
        if weights.suffix == ".engine":  # ultralytics writes its metadata first: 4-byte length, then JSON
            import json

            with open(weights, "rb") as f:
                meta = json.loads(f.read(int.from_bytes(f.read(4), "little")))
            self.model = YOLO(str(weights), task=meta["task"])
            self.names = {int(k): v for k, v in meta["names"].items()}
            self.imgsz = tuple(meta["imgsz"])
            self.name += "_trt"
        else:
            self.model = YOLO(str(weights))
            self.names = {int(k): v for k, v in self.model.names.items()}
            self.imgsz = imgsz or int(self.model.overrides.get("imgsz", 1280))
        self.reports_held = any(n.endswith(("_spear", "_scythe", "_gauntlets")) for n in self.names.values())
        self.conf = conf
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _cls(self, name: str) -> str:
        return f"{name}_unarmed" if name in ("wu_shang", "mordex") else name

    def __call__(self, image: np.ndarray) -> list[Box]:
        result = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, verbose=False, device=self.device)[0]
        points = (result.keypoints.xy[:, 0].tolist() if result.keypoints is not None and len(result.boxes)
                  else [None] * len(result.boxes))  # a frame with no detections has no keypoint axis
        boxes = [Box(self._cls(self.names[int(c)]), *b, score=s, point=tuple(k) if k else None)
                 for b, s, c, k in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                       result.boxes.cls.tolist(), points)]
        # Only two fighters exist: keep the most confident box of each identity.
        best: dict[str, Box] = {}
        for b in boxes:
            who = identity(b.cls)
            if who and (who not in best or b.score > best[who].score):
                best[who] = b
        return list(best.values()) + [b for b in boxes if identity(b.cls) is None]
