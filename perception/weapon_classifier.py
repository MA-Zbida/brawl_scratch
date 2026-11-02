"""Armed / unarmed classification from a detected character box.

Weapon state has two sources and neither is sufficient alone:

* **Action inference** records executed pickup/throw intent. Pickup remains
  provisional until confirmed; an ignored input is not proof of possession.
  It cannot see knockback disarms and says nothing about the opponent.
* **This classifier** sees the result. Cross-validated at 95.7% on 300 hand-labelled
  crops, it works identically for both fighters because it only looks at pixels.

They fail in different directions, so `WeaponStateFusion` below keeps both and
treats sustained confident disagreement after confirmed possession as a possible
disarm, without turning isolated prediction errors into events.

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

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

#: Below this confidence the classifier abstains rather than overwrite a prior.
from perception.weapon_evidence import DEFAULT_MIN_CONFIDENCE


@dataclass
class WeaponPrediction:
    armed: bool
    confidence: float
    #: False when the crop was unusable (off-screen, degenerate) or the model was
    #: not run on this frame. Callers must not treat an abstention as "unarmed".
    valid: bool = True
    #: True only on a frame where the classifier actually evaluated this crop.
    #: Held predictions are diagnostic history, not new evidence capable of
    #: overriding a more recent pickup/throw inference.
    fresh: bool = True


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
        # Diagnostics only; normal training performs no crop copies or disk I/O.
        self.prediction_observer: Optional[Callable] = None

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
        if frame is None:
            self._cache.clear()
            return {}
        # A returning fighter must be re-observed, not inherit a crop from before
        # disappearance/respawn. Prune even when this is a cadence-skipped frame.
        self._cache = {name: prediction for name, prediction in self._cache.items() if name in boxes}
        if not run or not boxes:
            return {name: replace(prediction, fresh=False) for name, prediction in self._cache.items()}

        names: list[str] = []
        patches: list[np.ndarray] = []
        refreshed: set[str] = set()
        for name, box in boxes.items():
            patch = self.crop(frame, *box)
            if patch is None:
                self._cache[name] = WeaponPrediction(False, 0.0, valid=False, fresh=True)
                refreshed.add(name)
                continue
            names.append(name)
            patches.append(patch)

        observer = getattr(self, "prediction_observer", None)
        for name, patch, prediction in zip(names, patches, self.predict_batch(patches)):
            self._cache[name] = prediction
            refreshed.add(name)
            if observer is not None:
                observer(name, boxes[name], patch, prediction)
        return {
            name: prediction if name in refreshed else replace(prediction, fresh=False)
            for name, prediction in self._cache.items()
        }


class WeaponStateFusion:
    """Immediate action intent, bounded confirmation, debounced visual correction.

    Counters count independent classifier refreshes, never cached predictions.
    A rejected pickup is not a disarm: possession must first be established.
    """

    def __init__(self, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> None:
        self.min_confidence = float(min_confidence)
        if not 0 <= self.min_confidence <= 1:
            raise ValueError('Invalid weapon confidence threshold')
        self.disarm_events = 0
        self.reset()

    def reset(self) -> None:
        self.confirmed = False
        self.pickup_pending = 0
        self.throw_guard = 0
        self.positive_refreshes = self.negative_refreshes = 0
        self.evidence_age = 0

    def observe_action(self, *, pickup: bool, dropped: bool) -> None:
        """Called AFTER fusion of the capture that predates this executed input."""
        if pickup or dropped:
            self.reset()
            # Three classifier periods to establish a pickup, two to let a
            # throw leave the crop. Neither intent can latch state forever.
            self.pickup_pending = 12 if pickup and not dropped else 0
            self.throw_guard = 8 if dropped else 0

    def resolve(self, action_inferred: bool, prediction: Optional[WeaponPrediction]) -> tuple[bool, bool]:
        """Return `(armed, disarm_detected)`."""
        was_pending = self.pickup_pending > 0
        self.pickup_pending = max(0, self.pickup_pending - 1)
        self.evidence_age += 1
        if self.evidence_age > 8:
            self.positive_refreshes = self.negative_refreshes = 0
        if self.throw_guard:
            self.throw_guard -= 1
            return False, False
        reliable = (prediction is not None and prediction.valid and prediction.fresh
                    and self.min_confidence <= prediction.confidence <= 1.)
        if reliable:
            self.evidence_age = 0
            if prediction.armed:
                self.positive_refreshes += 1
                self.negative_refreshes = 0
                if self.positive_refreshes >= 2:
                    self.confirmed = True
                    self.pickup_pending = 0
                    return True, False
            else:
                self.negative_refreshes += 1
                self.positive_refreshes = 0
                if self.negative_refreshes >= 3 and not self.pickup_pending:
                    disarm = bool(action_inferred and self.confirmed)
                    self.disarm_events += int(disarm)
                    self.confirmed = False
                    return False, disarm
        if was_pending and not self.pickup_pending and not self.confirmed:
            return False, False
        return bool(action_inferred), False
