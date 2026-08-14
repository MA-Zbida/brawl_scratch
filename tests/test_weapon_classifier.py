"""Tests for weapon-state classification and its fusion with action inference.

The failure modes worth guarding are silent ones: cropping differently at inference
than at training, treating an abstention as "unarmed", and letting the classifier's
residual error flicker a flag that action inference already had right.
"""

from __future__ import annotations

import numpy as np

from perception.weapon_classifier import WeaponPrediction, WeaponStateFusion


class _FakeClassifier:
    """Exercises the cadence and cropping logic without loading a checkpoint."""

    def __init__(self, interval: int = 4, pad: float = 0.45) -> None:
        self.interval = interval
        self.pad = pad
        self.width, self.height = 64, 96
        self._frame = 0
        self._cache: dict[str, WeaponPrediction] = {}
        self.runs = 0

    crop = staticmethod(lambda *a, **k: None)

    def should_run(self) -> bool:
        return self._frame % self.interval == 0


def test_fusion_keeps_action_inference_when_the_classifier_abstains() -> None:
    """An invalid crop is not evidence of anything -- it must not clear the flag."""
    fusion = WeaponStateFusion()

    armed, disarm = fusion.resolve(True, WeaponPrediction(False, 0.0, valid=False))
    assert armed is True and disarm is False

    armed, disarm = fusion.resolve(True, None)
    assert armed is True and disarm is False


def test_low_confidence_predictions_do_not_override() -> None:
    """At ~96% accuracy the residual error would flicker the flag every few frames."""
    fusion = WeaponStateFusion(min_confidence=0.70)
    armed, disarm = fusion.resolve(True, WeaponPrediction(armed=False, confidence=0.55))
    assert armed is True, "an unconfident prediction overrode action inference"
    assert disarm is False
    assert fusion.disarm_events == 0


def test_confident_disagreement_is_reported_as_a_disarm() -> None:
    """Action-inferred armed + confidently-unseen weapon = the knockback drop.

    Neither source detects this alone: action inference never saw a drop input, and
    the classifier alone cannot tell a knockback disarm from never having picked up.
    """
    fusion = WeaponStateFusion(min_confidence=0.70)

    armed, disarm = fusion.resolve(True, WeaponPrediction(armed=False, confidence=0.95))
    assert armed is False
    assert disarm is True
    assert fusion.disarm_events == 1


def test_agreement_produces_no_disarm_event() -> None:
    fusion = WeaponStateFusion(min_confidence=0.70)
    for inferred, predicted in ((True, True), (False, False)):
        armed, disarm = fusion.resolve(inferred, WeaponPrediction(armed=predicted, confidence=0.99))
        assert armed is predicted
        assert disarm is False
    assert fusion.disarm_events == 0


def test_classifier_finding_a_weapon_action_inference_missed_is_not_a_disarm() -> None:
    """Picking a weapon off the ground without the modelled input is the other way round."""
    fusion = WeaponStateFusion(min_confidence=0.70)
    armed, disarm = fusion.resolve(False, WeaponPrediction(armed=True, confidence=0.99))
    assert armed is True
    assert disarm is False


def test_cadence_runs_on_the_expected_frames() -> None:
    """Weapon state moves ~3 times in 1500 steps; per-frame inference is 21% of a step."""
    fake = _FakeClassifier(interval=4)
    ran = []
    for step in range(12):
        ran.append(fake.should_run())
        fake._frame += 1
    assert ran == [True, False, False, False] * 3


def test_crop_uses_training_padding_and_clamps_to_the_frame() -> None:
    """Cropping differently than training degrades the model with no error anywhere."""
    from perception.weapon_classifier import WeaponClassifier

    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    # Call the unbound method so no checkpoint is needed.
    patch = WeaponClassifier.crop.__get__(_FakeClassifier(pad=0.0))(frame, 0.5, 0.5, 0.2, 0.4)
    assert patch.shape[1] == int(0.2 * 300)
    assert patch.shape[0] == int(0.4 * 200)

    padded = WeaponClassifier.crop.__get__(_FakeClassifier(pad=0.45))(frame, 0.5, 0.5, 0.2, 0.4)
    assert padded.shape[1] > patch.shape[1], "padding must widen the crop"


def test_degenerate_box_yields_no_crop(tmp_path) -> None:
    from perception.weapon_classifier import WeaponClassifier

    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    tiny = WeaponClassifier.crop.__get__(_FakeClassifier(pad=0.0))(frame, 0.5, 0.5, 0.001, 0.001)
    assert tiny is None, "a sub-pixel box must abstain rather than return garbage"
