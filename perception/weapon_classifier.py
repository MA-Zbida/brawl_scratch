"""Armed / unarmed classification from a detected character box.

Weapon state has two sources and neither is sufficient alone:

* **Action inference** knows the agent pressed pickup. It is correct until a
  knockback disarms the fighter, which it cannot see at all, and it says nothing
  whatsoever about the opponent.
* **This classifier** sees the result. Cross-validated at 95.7% on 300 hand-labelled
  crops, it works identically for both fighters because it only looks at pixels.

They fail in different directions, so `WeaponStateFusion` below keeps both and
treats a confident disagreement as evidence of the event action-inference is blind
to: a weapon lost to knockback.

**Cadence.** Inference costs ~4.4 ms for two crops, which is 21% of a 21 ms step --
too much to pay every frame. Weapon state is also one of the slowest-moving
variables in the observation: a live match showed two pickups and one drop across
1537 steps. So the classifier runs every `interval` frames and holds its result in
between, which costs ~1.1 ms per frame at the default and delays noticing a disarm
by under 100 ms, against a status quo of never noticing.

**The crop must match training.** The checkpoint records the input size and the
padding used when the crops were cut. Cropping differently at inference degrades
the model silently -- no error, just worse predictions -- so those values are read
from the checkpoint rather than configured here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

#: Below this confidence the classifier abstains rather than overwrite a prior.
DEFAULT_MIN_CONFIDENCE = 0.70


@dataclass
class WeaponPrediction:
    armed: bool
    confidence: float
    #: False when the crop was unusable (off-screen, degenerate) or the model was
    #: not run on this frame. Callers must not treat an abstention as "unarmed".
    valid: bool = True


class WeaponClassifier:
    """Runs the trained crop classifier on demand, honouring the checkpoint contract."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "",
        interval: int = 4,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        import torch

        from train.train_weapon_classifier import build_model

        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"weapon classifier checkpoint not found: {path}")
        blob = torch.load(str(path), map_location="cpu", weights_only=False)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.width = int(blob["input_width"])
        self.height = int(blob["input_height"])
        self.pad = float(blob["crop_pad"])
        self.classes = list(blob.get("classes", ["unarmed", "armed"]))
        self.cv_accuracy = float(blob.get("cv_accuracy", 0.0))
        self.interval = max(1, int(interval))
        self.min_confidence = float(min_confidence)

        class _Args:
            backbone = str(blob.get("backbone", "small"))

        self._torch = torch
        self.model = build_model(torch, torch.nn, _Args.backbone).to(self.device).eval()
        self.model.load_state_dict(blob["state_dict"])

        self._frame = 0
        self._cache: dict[str, WeaponPrediction] = {}

    # ── cropping ───────────────────────────────────────────────────────────

    def crop(self, frame: np.ndarray, cx: float, cy: float, w: float, h: float) -> Optional[np.ndarray]:
        """Cut a normalised box out of the frame using the training-time padding."""
        height, width = frame.shape[:2]
        half_w = (w * (1.0 + self.pad)) / 2.0
        half_h = (h * (1.0 + self.pad)) / 2.0
        x0 = max(0, int((cx - half_w) * width))
        x1 = min(width, int((cx + half_w) * width))
        y0 = max(0, int((cy - half_h) * height))
        y1 = min(height, int((cy + half_h) * height))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        return frame[y0:y1, x0:x1]

    def predict_batch(self, patches: list[np.ndarray]) -> list[WeaponPrediction]:
        """Classify several crops in one forward pass."""
        import cv2

        if not patches:
            return []
        batch = np.zeros((len(patches), self.height, self.width, 3), dtype=np.float32)
        for i, patch in enumerate(patches):
            batch[i] = cv2.resize(patch, (self.width, self.height), interpolation=cv2.INTER_AREA) / 255.0

        torch = self._torch
        tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[:, 1].cpu().numpy()

        return [
            WeaponPrediction(armed=bool(p >= 0.5), confidence=float(max(p, 1.0 - p)), valid=True)
            for p in probs
        ]

    def should_run(self) -> bool:
        return self._frame % self.interval == 0

    def update(self, frame: Optional[np.ndarray], boxes: dict[str, tuple[float, float, float, float]]) -> dict[str, WeaponPrediction]:
        """Classify the given boxes, or return the held result on skipped frames.

        `boxes` maps a name ("player", "opponent") to a normalised (cx, cy, w, h).
        """
        run = self.should_run()
        self._frame += 1
        if not run or frame is None or not boxes:
            return dict(self._cache)

        names: list[str] = []
        patches: list[np.ndarray] = []
        for name, box in boxes.items():
            patch = self.crop(frame, *box)
            if patch is None:
                self._cache[name] = WeaponPrediction(False, 0.0, valid=False)
                continue
            names.append(name)
            patches.append(patch)

        for name, prediction in zip(names, self.predict_batch(patches)):
            self._cache[name] = prediction
        return dict(self._cache)


class WeaponStateFusion:
    """Combine action-inferred weapon state with the classifier.

    Action inference is authoritative about *intent* -- it knows a pickup was
    pressed -- and blind to knockback disarms. The classifier is authoritative
    about *appearance* and has a few percent of error. Neither should simply
    override the other, so:

    * the classifier only overrides when it is confident (`min_confidence`), which
      keeps its residual error from flickering the flag every few frames;
    * a confident "unarmed" against an action-inferred "armed" is reported as a
      **disarm event**. That is the knockback drop neither source detects alone,
      and it is worth surfacing rather than silently absorbing.
    """

    def __init__(self, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> None:
        self.min_confidence = float(min_confidence)
        self.disarm_events = 0

    def resolve(self, action_inferred: bool, prediction: Optional[WeaponPrediction]) -> tuple[bool, bool]:
        """Return `(armed, disarm_detected)`."""
        if prediction is None or not prediction.valid:
            return bool(action_inferred), False
        if prediction.confidence < self.min_confidence:
            return bool(action_inferred), False

        armed = bool(prediction.armed)
        disarm = bool(action_inferred and not armed)
        if disarm:
            self.disarm_events += 1
        return armed, disarm
