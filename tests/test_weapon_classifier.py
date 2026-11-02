"""Tests for weapon-state classification and its fusion with action inference.

The failure modes worth guarding are silent ones: cropping differently at inference
than at training, treating an abstention as "unarmed", and letting the classifier's
residual error flicker a flag that action inference already had right.
"""

from __future__ import annotations

import numpy as np
import pytest

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


def test_sustained_disagreement_after_confirmed_possession_is_a_disarm() -> None:
    """A one-frame classification error must not become a disarm event."""
    fusion = WeaponStateFusion(min_confidence=0.70)
    for _ in range(2):
        fusion.resolve(True, WeaponPrediction(True, .99))
    for _ in range(3):
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
    fusion.resolve(False, WeaponPrediction(armed=True, confidence=0.99))
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


def test_held_classifier_prediction_is_not_reported_as_fresh() -> None:
    """A cached armed result remains observable, but is not new fusion evidence."""
    from perception.weapon_classifier import WeaponClassifier

    classifier = WeaponClassifier.__new__(WeaponClassifier)
    classifier.interval = 4
    classifier._frame = 1  # cadence skips this update
    classifier._cache = {
        "player": WeaponPrediction(armed=True, confidence=0.99, valid=True, fresh=True)
    }

    result = classifier.update(
        np.zeros((32, 32, 3), dtype=np.uint8),
        {"player": (0.5, 0.5, 0.2, 0.2)},
    )

    assert result["player"].armed is True
    assert result["player"].valid is True
    assert result["player"].fresh is False


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


@pytest.mark.parametrize('intent', [True, False])
def test_cached_visual_state_cannot_undo_a_new_pickup_or_throw(intent):
    fusion = WeaponStateFusion()
    cached = WeaponPrediction(armed=not intent, confidence=.99, fresh=False)
    assert fusion.resolve(intent, cached) == (intent, False)
    assert fusion.disarm_events == 0


def test_lost_fighter_invalidates_cached_visual_state_even_on_skipped_frame():
    from perception.weapon_classifier import WeaponClassifier
    classifier = WeaponClassifier.__new__(WeaponClassifier)
    classifier.interval, classifier._frame = 4, 1
    classifier._cache = {'player': WeaponPrediction(True, .99)}
    assert classifier.update(np.zeros((32, 32, 3), np.uint8), {}) == {}
    assert classifier.update(np.zeros((32, 32, 3), np.uint8), {'player': (.5, .5, .2, .2)}) == {}


def test_missing_frame_cannot_return_a_valid_weapon_prediction():
    from perception.weapon_classifier import WeaponClassifier
    classifier = WeaponClassifier.__new__(WeaponClassifier)
    classifier.interval, classifier._frame = 4, 1
    classifier._cache = {'player': WeaponPrediction(True, .99)}
    assert classifier.update(None, {'player': (.5, .5, .2, .2)}) == {}


def test_refresh_prunes_absent_fighters_and_retains_invalid_crop_status():
    from perception.weapon_classifier import WeaponClassifier
    classifier = WeaponClassifier.__new__(WeaponClassifier)
    classifier.interval, classifier._frame, classifier.pad = 4, 0, 0.
    classifier._cache = {'absent': WeaponPrediction(True, .99)}
    classifier.predict_batch = lambda patches: [WeaponPrediction(True, .98) for _ in patches]
    boxes = {'player': (.5, .5, .4, .4), 'opponent': (.2, .2, .001, .001)}
    frame = np.zeros((64, 64, 3), np.uint8)
    fresh = classifier.update(frame, boxes)
    assert set(fresh) == {'player', 'opponent'}
    assert fresh['player'].armed and fresh['player'].valid and fresh['player'].fresh
    assert not fresh['opponent'].valid and fresh['opponent'].fresh
    held = classifier.update(frame, boxes)
    assert held['player'].armed and not held['player'].fresh
    assert not held['opponent'].valid and not held['opponent'].fresh


def test_pre_input_capture_cannot_undo_current_pickup_intent():
    from types import SimpleNamespace
    from action_space import Action
    from feature_extractor.memory.state_spec import StateSpec
    from test_observation_window import make_env, det
    env, extractor = make_env(action_repeat_steps=1)
    env.weapon_classifier = SimpleNamespace(update=lambda frame, boxes: {
        'player': WeaponPrediction(False, .99, fresh=True)})
    extractor.detections = [det('character', .3, .6), det('character', .7, .6),
                           det('indicator_self', .3, .53, .02, .02),
                           det('weapon', .3, .64, .04, .04)]
    try:
        env.reset()
        obs, _, _, _, info = env.step(Action.PICKUP)
        assert info['weapon_pickup_action'] == 1.
        assert StateSpec.get(obs, 'player_has_weapon') == 1.
        # Sustained later evidence can correct a pickup the game ignored.
        for _ in range(12):
            obs, _, _, _, _ = env.step(Action.NOOP)
        assert StateSpec.get(obs, 'player_has_weapon') == 0.
    finally:
        env.close()


def test_missing_detection_is_not_a_fresh_classifier_crop():
    from test_observation_window import make_env, det
    env, extractor = make_env()
    extractor.detections = [det('character', .3, .6), det('indicator_self', .3, .53, .02, .02)]
    try:
        env.reset()
        assert 'player' in env._weapon_classifier_boxes()
        env.memory.update_from_detections([])
        assert 'player' not in env._weapon_classifier_boxes()
    finally:
        env.close()


def test_unobserved_identity_cannot_supply_weapon_crops():
    from test_observation_window import make_env, det
    env, extractor = make_env()
    extractor.detections = [det('character', .3, .6), det('character', .7, .6),
                           det('indicator_self', .3, .53, .02, .02)]
    try:
        env.reset()
        assert set(env._weapon_classifier_boxes()) == {'player', 'opponent'}
        env.memory.update_from_detections(extractor.detections[:2])
        assert not env.memory.identity_observed
        assert env._weapon_classifier_boxes() == {}
    finally:
        env.close()


def test_crop_observer_gets_exact_input_only_when_inference_runs():
    from perception.weapon_classifier import WeaponClassifier
    classifier = WeaponClassifier.__new__(WeaponClassifier)
    classifier.interval, classifier._frame, classifier.pad = 4, 0, 0.
    classifier._cache = {}
    classifier.predict_batch = lambda patches: [WeaponPrediction(True, .98) for _ in patches]
    seen = []
    classifier.prediction_observer = lambda *args: seen.append(args)
    frame = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    box = (.5, .5, .4, .4)
    classifier.update(frame, {'player': box})
    classifier.update(frame, {'player': box})
    assert len(seen) == 1
    name, recorded_box, patch, prediction = seen[0]
    assert name == 'player' and recorded_box == box
    np.testing.assert_array_equal(patch, classifier.crop(frame, *box))
    assert prediction.armed and prediction.fresh


def test_inference_tensor_matches_training_image_preprocessing(tmp_path):
    import cv2
    import torch
    from perception.weapon_classifier import WeaponClassifier
    from train.train_weapon_classifier import load_images

    class InspectModel(torch.nn.Module):
        def forward(self, batch):
            self.batch = batch.clone()
            return torch.tensor([[0., 2.]], dtype=torch.float32).repeat(len(batch), 1)

    classifier = WeaponClassifier.__new__(WeaponClassifier)
    classifier.width, classifier.height = 96, 144
    classifier._torch, classifier.device = torch, torch.device('cpu')
    classifier.model = InspectModel().eval()
    patch = np.random.default_rng(4).integers(0, 256, (63, 41, 3), dtype=np.uint8)
    path = tmp_path / 'native.png'
    assert cv2.imwrite(str(path), patch)
    expected = load_images(cv2, [path], (96, 144))
    assert classifier.predict_batch([]) == []
    predictions = classifier.predict_batch([patch])
    np.testing.assert_array_equal(classifier.model.batch.numpy(), expected)
    assert predictions[0].armed
    assert predictions[0].confidence == pytest.approx(.880797, abs=1e-6)


def test_inference_crop_matches_labelling_crop():
    from pathlib import Path
    from perception.weapon_classifier import WeaponClassifier
    from tools.label_weapon_crops import Crop, crop_pixels
    frame = np.random.default_rng(2).integers(0, 256, (108, 192, 3), dtype=np.uint8)
    for box in [(.5, .5, .1, .2), (.01, .99, .1, .2)]:
        expected = crop_pixels(frame, Crop(Path('unused.png'), 0, *box), .45)
        actual = WeaponClassifier.crop.__get__(_FakeClassifier())(frame, *box)
        np.testing.assert_array_equal(actual, expected)
