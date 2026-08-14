from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from action_space import Action
from tools import debug_observation_overlay as overlay


@pytest.mark.parametrize(
    ("pressed", "mirrored", "expected"),
    [
        (set(), False, Action.NOOP),
        ({"d"}, False, Action.MOVE_TOWARD),
        ({"d"}, True, Action.MOVE_AWAY),
        ({"a", "space"}, False, Action.JUMP_AWAY),
        ({"d", "e"}, False, Action.DODGE_TOWARD),
        ({"a", "w", "e"}, False, Action.DODGE_UP_AWAY),
        ({"d", 75}, False, Action.LIGHT_TOWARD),
        ({"s", 77}, False, Action.HEAVY_DOWN),
        ({"d", 76}, False, Action.PICKUP),
    ],
)
def test_pressed_keys_map_to_one_canonical_action(
    pressed: set[str | int],
    mirrored: bool,
    expected: Action,
) -> None:
    assert overlay._action_from_pressed_keys(pressed, mirrored=mirrored) == int(expected)


def test_dodge_takes_precedence_over_simultaneous_attack() -> None:
    pressed = {"d", "e", 75}

    assert overlay._action_from_pressed_keys(pressed, mirrored=False) == int(Action.DODGE_TOWARD)


def test_read_keyboard_action_returns_categorical_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(overlay.keyboard, "is_pressed", lambda key: key in {"a", 77})

    action = overlay.read_keyboard_action(mirrored=True)

    assert action == int(Action.HEAVY_TOWARD)
    assert isinstance(action, int)


def test_observation_panel_accepts_and_names_categorical_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered_text: list[str] = []
    monkeypatch.setattr(overlay, "_format_success_lines", lambda info, spec: ([], False))
    monkeypatch.setattr(overlay, "_format_base_obs_lines", lambda obs: [])
    monkeypatch.setattr(
        overlay.cv2,
        "putText",
        lambda _panel, text, *_args, **_kwargs: rendered_text.append(text),
    )
    monkeypatch.setattr(overlay.cv2, "line", lambda *_args, **_kwargs: None)

    overlay.draw_obs_panel(
        panel_width=640,
        panel_height=360,
        obs=np.zeros((1,), dtype=np.float32),
        step_idx=1,
        reward=0.0,
        action=int(Action.DODGE_UP_TOWARD),
        info={},
        spec=SimpleNamespace(),
        font_scale=0.5,
        line_height=20,
    )

    assert "action: DODGE_UP_TOWARD (14)" in rendered_text


def test_success_panel_exposes_recovery_sequence_state() -> None:
    spec = SimpleNamespace(name="stage1_recovery_mastery", success_threshold=0.08)
    info = {
        "stage_name": spec.name,
        "goal_success": 0.0,
        "terminal_success": 0.0,
        "goal_error": 0.02,
        "sequential_goal_enabled": 1.0,
        "sequential_phase": 2,
        "sequential_step1_completed": 1.0,
        "stage_feature_names": [],
        "raw_goal_feats": np.zeros((0,), dtype=np.float32),
        "goal_target": np.zeros((0,), dtype=np.float32),
        "goal_mask": np.zeros((0,), dtype=np.float32),
    }

    lines, success = overlay._format_success_lines(info, spec)

    assert success is False
    assert "sequence: ARMED_RETURN (phase=2)" in lines
    assert "terminal_success: 0" in lines
