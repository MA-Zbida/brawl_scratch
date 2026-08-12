"""Tap-type inputs must be held long enough for the game to sample them.

Brawlhalla polls input at ~60 Hz. A keyDown/keyUp pair emitted inside a single
control step lasts far less than one poll interval and is frequently dropped, so
tap keys are held across their latch window instead of pulsed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from action_space import Action
from env import EnvConfig


class RecordingController:
    """Records the key_down / key_up / set_pressed calls the env emits."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.held: set[str] = set()

    def set_pressed(self, keys) -> None:
        self.events.append(("set_pressed", ",".join(sorted(keys))))

    def key_down(self, key: str) -> None:
        self.events.append(("down", key))
        self.held.add(key)

    def key_up(self, key: str) -> None:
        self.events.append(("up", key))
        self.held.discard(key)

    def tap(self, keys) -> None:  # pragma: no cover - must not be used by the loop
        self.events.append(("tap", ",".join(sorted(keys))))

    def reset(self) -> None:
        self.events.append(("reset", ""))
        self.held.clear()


def _env(tap_latch_steps: int = 1):
    """Build a BrawlDeepEnv with every external dependency stubbed out."""
    from env import BrawlDeepEnv

    env = BrawlDeepEnv.__new__(BrawlDeepEnv)
    env.config = EnvConfig(tap_latch_steps=tap_latch_steps)
    env.input_controller = RecordingController()
    env._mirrored = False
    from action_space import TAP_KEYS
    env._tap_latch_remaining = {key: 0 for key in sorted(TAP_KEYS)}
    return env


def test_tap_is_held_not_pulsed():
    """A single attack input holds the key down, rather than press-releasing it."""
    env = _env()
    ctl = env.input_controller

    env._apply_action(int(Action.PICKUP), emit_tap_actions=True)

    assert ("down", "num5") in ctl.events
    assert ("up", "num5") not in ctl.events
    assert "num5" in ctl.held, "NUM5 must still be held after the step that requested it"
    assert not any(kind == "tap" for kind, _ in ctl.events), "control loop must not pulse taps"


def test_tap_releases_after_latch_window():
    """The key is released once the latch expires, not before."""
    env = _env(tap_latch_steps=2)
    ctl = env.input_controller

    env._apply_action(int(Action.LIGHT_NEUTRAL), emit_tap_actions=True)
    assert "num4" in ctl.held

    # Second step of the window: still held, no new press needed.
    env._apply_action(int(Action.NOOP), emit_tap_actions=True)
    assert "num4" in ctl.held, "key must stay down for the whole latch window"

    # Window expired.
    env._apply_action(int(Action.NOOP), emit_tap_actions=True)
    assert "num4" not in ctl.held
    assert ("up", "num4") in ctl.events


def test_hold_window_spans_at_least_one_full_step():
    """Down and up never land in the same step, at the minimum latch setting."""
    env = _env(tap_latch_steps=1)
    ctl = env.input_controller

    env._apply_action(int(Action.JUMP), emit_tap_actions=True)
    down_idx = ctl.events.index(("down", "space"))
    assert ("up", "space") not in ctl.events[down_idx:]

    env._apply_action(int(Action.NOOP), emit_tap_actions=True)
    assert ("up", "space") in ctl.events


def test_release_all_inputs_clears_latched_taps():
    """Losing control must not leave a tap key physically held down."""
    env = _env(tap_latch_steps=8)
    ctl = env.input_controller

    env._apply_action(int(Action.HEAVY_NEUTRAL), emit_tap_actions=True)
    assert "num6" in ctl.held

    env._release_all_inputs()
    assert "num6" not in ctl.held
    assert all(v == 0 for v in env._tap_latch_remaining.values())


@pytest.mark.parametrize(
    "action,expected",
    [
        (Action.LIGHT_NEUTRAL, "num4"),
        (Action.HEAVY_NEUTRAL, "num6"),
        (Action.PICKUP, "num5"),
        (Action.JUMP, "space"),
        (Action.DODGE_SPOT, "e"),
    ],
)
def test_action_maps_to_expected_key(action, expected):
    env = _env()
    env._apply_action(int(action), emit_tap_actions=True)
    assert expected in env.input_controller.held


def test_null_controller_supports_key_down_up():
    """NullInputController must satisfy the same interface as the real one."""
    from env import NullInputController

    controller = NullInputController()
    controller.key_down("num5")
    controller.key_up("num5")
    controller.set_pressed({"a"})
    controller.reset()


def test_inputs_are_released_before_the_optimiser_runs():
    """The env is not stepped during a PPO update, but the game keeps running.

    Without this, the last action's keys stay physically held for the whole
    optimiser pass -- seconds of holding a direction while the agent cannot react.
    """
    from train.llc_stage_common import ReleaseInputsOnRolloutEnd

    calls = []

    class FakeVecEnv:
        def env_method(self, name, *args, **kwargs):
            calls.append(name)
            return [None]

    class FakeModel:
        # BaseCallback.training_env is a read-only property over model.get_env().
        def get_env(self):
            return FakeVecEnv()

    cb = ReleaseInputsOnRolloutEnd()
    cb.model = FakeModel()
    cb._on_rollout_end()

    assert calls == ["_release_all_inputs"]


def test_release_failure_does_not_break_training():
    """A stuck key is bad; a crashed multi-hour run is worse."""
    from train.llc_stage_common import ReleaseInputsOnRolloutEnd

    class ExplodingVecEnv:
        def env_method(self, *args, **kwargs):
            raise RuntimeError("vec env does not support env_method")

    class FakeModel:
        def get_env(self):
            return ExplodingVecEnv()

    cb = ReleaseInputsOnRolloutEnd()
    cb.model = FakeModel()
    cb._on_rollout_end()   # must not raise
