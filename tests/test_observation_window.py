"""Observation assembly: layout, temporal window, and end-to-end canonicalisation.

The layout contract is that the first ``StateSpec.dim()`` entries are the complete
current frame, so ``StateSpec.get(obs, name)`` keeps working no matter how deep the
history window is. Everything after is stacked dynamic blocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import BrawlDeepEnv, EnvConfig, NullInputController
from feature_extractor.memory.state_spec import StateSpec


class StubFrames:
    def get_frame(self):
        return np.zeros((1080, 1920, 3), dtype=np.uint8)

    def close(self):
        pass


class StubExtractor:
    def __init__(self):
        self.detections: list[dict] = []

    def predict(self, frame):
        return list(self.detections)


def det(class_name: str, x: float, y: float, w: float = 0.05, h: float = 0.09, conf: float = 0.9) -> dict:
    return {"class_name": class_name, "bbox": [x, y, w, h], "confidence": conf}


def make_env(**config_kwargs):
    extractor = StubExtractor()
    config = EnvConfig(ui_regions=None, **config_kwargs)
    env = BrawlDeepEnv(
        extractor=extractor,
        frame_provider=StubFrames(),
        input_controller=NullInputController(),
        stocks_health_provider=None,
        config=config,
    )
    return env, extractor


# ── layout ──────────────────────────────────────────────────────────────────

def test_observation_width_matches_spec():
    env, _ = make_env(history_offsets=(2, 4, 8))
    expected = StateSpec.dim() + 3 * StateSpec.dynamic_dim()
    assert env.observation_space.shape == (expected,)
    assert len(env.get_observation_spec()) == expected


def test_current_frame_occupies_the_leading_slots():
    """StateSpec.get must stay valid on the full stacked observation."""
    env, extractor = make_env(history_offsets=(2, 4))
    extractor.detections = [
        det("character", 0.30, 0.60),
        det("character", 0.70, 0.60),
        det("indicator_self", 0.30, 0.53, 0.02, 0.02),
    ]
    obs, _ = env.reset()

    core = obs[: StateSpec.dim()]
    assert StateSpec.get(obs, "player_x") == pytest.approx(core[StateSpec.index("player_x")])
    assert StateSpec.get(obs, "rel_dx") > 0.0


def test_history_window_can_be_disabled():
    env, _ = make_env(history_offsets=())
    assert env.observation_space.shape == (StateSpec.dim(),)


# ── temporal window ─────────────────────────────────────────────────────────

def test_history_is_padded_with_the_first_frame():
    """Before enough frames exist, history repeats the oldest available frame."""
    env, extractor = make_env(history_offsets=(2,))
    extractor.detections = [
        det("character", 0.40, 0.60),
        det("character", 0.60, 0.60),
        det("indicator_self", 0.40, 0.53, 0.02, 0.02),
    ]
    obs, _ = env.reset()

    dynamic_dim = StateSpec.dynamic_dim()
    current = obs[:dynamic_dim]
    past = obs[StateSpec.dim() : StateSpec.dim() + dynamic_dim]
    np.testing.assert_allclose(current, past, atol=1e-6)


def test_history_retains_an_older_frame():
    """After the agent moves, the history slice must differ from the present."""
    env, extractor = make_env(history_offsets=(2,), action_repeat_steps=1)
    extractor.detections = [
        det("character", 0.30, 0.60),
        det("character", 0.80, 0.60),
        det("indicator_self", 0.30, 0.53, 0.02, 0.02),
    ]
    env.reset()

    for x in (0.35, 0.40, 0.45):
        extractor.detections = [
            det("character", x, 0.60),
            det("character", 0.80, 0.60),
            det("indicator_self", x, 0.53, 0.02, 0.02),
        ]
        obs, *_ = env.step(0)

    dynamic_dim = StateSpec.dynamic_dim()
    current_x = obs[StateSpec.index("player_x")]
    past_x = obs[StateSpec.dim() + StateSpec.index("player_x")]
    assert current_x != pytest.approx(past_x), "history slice should lag the current frame"


# ── canonicalisation, end to end ────────────────────────────────────────────

def _observe(env, extractor, player_x, opponent_x):
    extractor.detections = [
        det("character", player_x, 0.60),
        det("character", opponent_x, 0.60),
        det("indicator_self", player_x, 0.53, 0.02, 0.02),
    ]
    obs, info = env.reset()
    return obs, info


def test_reflected_scenes_yield_identical_observations():
    """The whole point: one skill instead of two."""
    from feature_extractor.memory.canonicalize import STAGE_CENTER_X

    env, extractor = make_env(history_offsets=(2,))
    reflect = lambda x: (2.0 * STAGE_CENTER_X) - x

    facing_right, _ = _observe(env, extractor, player_x=0.40, opponent_x=0.60)
    facing_right = facing_right.copy()
    facing_left, _ = _observe(env, extractor, player_x=reflect(0.40), opponent_x=reflect(0.60))

    # No flag to exclude: the mirror bit deliberately never enters the observation,
    # so the two canonical views must match exactly.
    np.testing.assert_allclose(facing_right, facing_left, atol=1e-6)


def test_canonical_frame_keeps_the_opponent_on_one_side():
    env, extractor = make_env()
    for player_x, opponent_x in ((0.40, 0.60), (0.60, 0.40)):
        obs, _ = _observe(env, extractor, player_x, opponent_x)
        assert StateSpec.get(obs, "rel_dx") > 0.0


def test_mirror_flag_travels_in_info_not_the_observation():
    """Handing the flag to the policy would undo the symmetry canonicalisation buys."""
    env, extractor = make_env()

    assert "canon_mirrored" not in StateSpec.names()

    _, info_right = _observe(env, extractor, 0.40, 0.60)
    _, info_left = _observe(env, extractor, 0.60, 0.40)
    assert info_right["canon_mirrored"] == 0.0
    assert info_left["canon_mirrored"] == 1.0


def test_canonical_action_reaches_the_game_as_the_right_physical_key():
    """MOVE_TOWARD is physical right normally and physical left when mirrored."""
    from action_space import KEY_LEFT, KEY_RIGHT, Action, to_keys

    env, extractor = make_env()

    _observe(env, extractor, player_x=0.60, opponent_x=0.40)
    assert env._mirrored is True
    held, _ = to_keys(int(Action.MOVE_TOWARD), mirrored=env._mirrored)
    assert KEY_LEFT in held

    _observe(env, extractor, player_x=0.40, opponent_x=0.60)
    assert env._mirrored is False
    held, _ = to_keys(int(Action.MOVE_TOWARD), mirrored=env._mirrored)
    assert KEY_RIGHT in held


def test_canonicalisation_can_be_switched_off():
    env, extractor = make_env(canonicalize_observation=False)
    obs, info = _observe(env, extractor, player_x=0.60, opponent_x=0.40)
    assert info["canon_mirrored"] == 0.0
    assert StateSpec.get(obs, "rel_dx") < 0.0


# ── silhouette features ─────────────────────────────────────────────────────

def test_bbox_extent_is_recorded():
    """Box dimensions carry animation signal the centre point discards."""
    env, extractor = make_env()
    extractor.detections = [
        det("character", 0.40, 0.60, w=0.11, h=0.07),
        det("character", 0.60, 0.60, w=0.05, h=0.09),
        det("indicator_self", 0.40, 0.53, 0.02, 0.02),
    ]
    obs, _ = env.reset()

    assert StateSpec.get(obs, "player_w") == pytest.approx(0.11)
    assert StateSpec.get(obs, "player_h") == pytest.approx(0.07)
    assert StateSpec.get(obs, "opponent_w") == pytest.approx(0.05)


def test_identity_provenance_is_observable():
    """A stale identity must be visible to the policy, not silently assumed."""
    env, extractor = make_env()

    obs, _ = _observe(env, extractor, player_x=0.40, opponent_x=0.60)
    assert StateSpec.get(obs, "identity_observed") == 1.0

    extractor.detections = [det("character", 0.41, 0.60), det("character", 0.60, 0.60)]
    obs, *_ = env.step(0)
    assert StateSpec.get(obs, "identity_observed") == 0.0


# ── detector input path ─────────────────────────────────────────────────────

def test_detector_receives_the_raw_frame_by_default():
    """No pre-resize: Ultralytics letterboxes internally, so doing it here too
    resamples the frame twice and costs the most expensive interpolation OpenCV
    has on a full 1920x1080 image."""
    from feature_extractor.yolo.extract import Extract
    import inspect

    params = inspect.signature(Extract.__init__).parameters
    assert params["infer_width"].default == 0
    assert params["infer_height"].default == 0


def test_env_does_not_pre_resize_by_default():
    config = EnvConfig()
    assert config.yolo_infer_width == 0
    assert config.yolo_infer_height == 0
    assert config.yolo_imgsz == 960, "must match the detector's training resolution"


def test_ui_probes_read_the_raw_frame_not_the_detector_input():
    """Stock and damage pixel coordinates are calibrated against the full-resolution
    capture, so they must never see a resized frame."""
    import numpy as np

    seen = {}

    class RecordingProbe:
        def __call__(self, frame, detections):
            seen["shape"] = None if frame is None else frame.shape
            return 3.0, 3.0, 351.0, 351.0

        def reset(self, preserve_match_state=True):
            pass

    extractor = StubExtractor()
    env = BrawlDeepEnv(
        extractor=extractor,
        frame_provider=StubFrames(),
        input_controller=NullInputController(),
        stocks_health_provider=RecordingProbe(),
        config=EnvConfig(),
    )
    env.reset()

    assert seen["shape"] == (1080, 1920, 3), (
        f"UI probes saw {seen['shape']}; they must receive the full-resolution capture"
    )
