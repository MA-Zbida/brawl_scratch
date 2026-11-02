from __future__ import annotations

import numpy as np
import pytest

from brawl.control.input import NullInputController
from brawl.control.semantic_actions import BY_NAME, ControlContext
from brawl.env.live import EnvConfig
from brawl.state.state_spec import StateSpec
from brawl.pipelines.semantic57.contract import ACTION_DIM, OBSERVATION_DIM
from brawl.pipelines.semantic57.env import SemanticBrawlEnv


class Frames:
    def get_frame(self):
        return np.zeros((1080, 1920, 3), np.uint8)


class Extractor:
    def predict(self, frame):
        return [
            {"class_name": "character", "bbox": [.7, .6, .05, .09], "confidence": .99},
            {"class_name": "character", "bbox": [.3, .6, .05, .09], "confidence": .99},
            {"class_name": "indicator_self", "bbox": [.7, .53, .02, .02], "confidence": .99},
        ]


def make_env():
    return SemanticBrawlEnv(frame_provider=Frames(), extractor=Extractor(),
                            input_controller=NullInputController(),
                            config=EnvConfig(ui_regions=None, action_repeat_steps=1))


def test_live_step_uses_semantic_id_and_records_physical_components(monkeypatch):
    clock = [1.]
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])
    env = make_env()
    monkeypatch.setattr(env, "control_context", lambda: ControlContext(grounded=False, mirrored=env._mirrored))
    obs, info = env.reset()
    assert obs.shape == (OBSERVATION_DIM,)
    assert env.action_space.n == ACTION_DIM
    action = BY_NAME["RECOVERY_TOWARD"].id
    clock[0] += .03
    obs, _, _, _, info = env.step(action)
    assert info["effective_action"] == action
    assert info["semantic_execution_status"] == "started"
    assert info["input_taps_pressed"] == ["num6"]
    assert StateSpec.get(obs, "prev_hdir") == 1.  # Physical left canonicalises once.
    assert obs[-ACTION_DIM:].tolist() == [1.] + [0.] * (ACTION_DIM - 1)
    clock[0] += .03
    assert env.step(0)[-1]["semantic_execution_status"] == "released"
    clock[0] += .03
    assert env.step(0)[-1]["semantic_execution_status"] == "completed"
    env.close()


def test_invalid_action_is_not_silently_substituted(monkeypatch):
    env = make_env()
    monkeypatch.setattr(env, "control_context", lambda: ControlContext(grounded=True, armed=True))
    env.reset()
    with pytest.raises(ValueError, match="masked"):
        env.step(BY_NAME["PICKUP"].id)
    with pytest.raises(ValueError, match="action"):
        env.step(999)
    env.close()


def test_none_capture_returns_noop_only_mask():
    env = make_env()
    env.reset()
    env.frame_provider.get_frame = lambda: None
    obs, _, _, _, info = env.step(0)
    assert obs[-ACTION_DIM:].tolist() == [1.] + [0.] * (ACTION_DIM - 1)
    env.close()


def test_close_clears_pending_sequence_even_when_capture_close_fails():
    env = make_env()
    env.executor.submit('RECOVERY_TOWARD', ControlContext(grounded=False), now=1.)
    def broken_close():
        raise RuntimeError('capture close')
    env.frame_provider.close = broken_close
    with pytest.raises(RuntimeError, match='capture close'):
        env.close()
    assert not env.executor.busy
    assert not env.input_controller.taps


def test_debug_names_cover_full_semantic_observation():
    env = make_env()
    assert len(env.get_observation_spec()) == OBSERVATION_DIM
    assert len(set(env.get_observation_spec())) == OBSERVATION_DIM
    env.close()
