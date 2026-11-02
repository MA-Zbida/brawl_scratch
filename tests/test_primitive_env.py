from __future__ import annotations

import numpy as np
import pytest

from brawl.control.action_space import Action
from brawl.control.input import NullInputController
from brawl.env.live import EnvConfig
from brawl.pipelines.primitive27.env import PrimitiveBrawlEnv
from brawl.pipelines.primitive27.training import OBSERVATION_DIM, PrimitiveControls


class Frames:
    def get_frame(self):
        return np.zeros((1080, 1920, 3), np.uint8)


class Extractor:
    def predict(self, frame):
        return [
            {'class_name': 'character', 'bbox': [.7, .6, .05, .09], 'confidence': .99},
            {'class_name': 'character', 'bbox': [.3, .6, .05, .09], 'confidence': .99},
            {'class_name': 'indicator_self', 'bbox': [.7, .53, .02, .02], 'confidence': .99},
        ]


def make_env(**config):
    return PrimitiveBrawlEnv(frame_provider=Frames(), extractor=Extractor(),
        input_controller=NullInputController(), config=EnvConfig(ui_regions=None, **config))


def test_primitive_environment_keeps_original_spaces_and_emits_input():
    env = make_env(action_repeat_steps=1)
    obs, _ = env.reset()
    assert obs.shape == (OBSERVATION_DIM,) and env.action_space.n == 27
    env.memory.identity_observed = True
    env.memory.player.exists = True
    env.memory.player.missing_frames = 0
    env.memory.player_respawn_timer = 0
    assert env._apply_action(Action.LIGHT_TOWARD) == {'num4'}
    assert not env.input_blocked
    obs, _, _, _, info = env.step(Action.MOVE_TOWARD)
    assert obs.shape == (OBSERVATION_DIM,)
    assert 'input_execution_blocked' in info
    env.close()


def test_missing_frame_never_becomes_a_valid_demo_label():
    env = make_env(action_repeat_steps=1)
    env.reset()
    env.frame_provider.get_frame = lambda: None
    _, _, _, _, info = env.step(Action.LIGHT_TOWARD)
    assert info['input_execution_blocked']
    assert not PrimitiveControls.recordable(Action.LIGHT_TOWARD, info)
    env.close()


def test_error_and_close_release_pending_taps(monkeypatch):
    env = make_env(action_repeat_steps=1)
    env.reset()
    def broken():
        raise RuntimeError('capture failure')
    monkeypatch.setattr(env.frame_provider, 'get_frame', broken)
    with pytest.raises(RuntimeError, match='capture failure'):
        env.step(Action.JUMP)
    assert not env._tap_sequencer.held
    env.close()


def test_changed_history_is_not_silently_loaded():
    with pytest.raises(ValueError, match='history'):
        make_env(action_repeat_steps=1, history_offsets=(1,))
