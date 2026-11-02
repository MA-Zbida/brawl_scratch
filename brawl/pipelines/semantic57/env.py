"""Opt-in live environment for semantic57; legacy BrawlDeepEnv stays discrete27."""
from __future__ import annotations

import time

import gymnasium as gym
import numpy as np

from brawl.control.action_space import Components
from brawl.control.semantic_actions import ACTIONS, ControlContext
from brawl.control.semantic_executor import SemanticExecutor, ExecutionResult
from brawl.env.live import BrawlDeepEnv
from brawl.pipelines.semantic57.contract import ACTION_DIM, BASE_DIM, HISTORY_OFFSETS, OBSERVATION_DIM


class _ObservedInput:
    def __init__(self, controller):
        self.controller = controller
        self.held: set[str] = set()
        self.taps: set[str] = set()
        self.new_presses: set[str] = set()

    def set_pressed(self, keys):
        self.controller.set_pressed(keys)
        self.held = set(keys)

    def key_down(self, key):
        self.controller.key_down(key)
        self.taps.add(key)
        self.new_presses.add(key)

    def key_up(self, key):
        self.controller.key_up(key)
        self.taps.discard(key)

    def reset(self):
        self.controller.reset()
        self.held.clear()
        self.taps.clear()


class SemanticBrawlEnv(BrawlDeepEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if tuple(self.config.history_offsets) != HISTORY_OFFSETS or self.config.action_repeat_steps != 1:
            raise ValueError("Semantic environment requires history (2,4,8) and action_repeat_steps=1")
        self.input_controller = _ObservedInput(self.input_controller)
        self.executor = SemanticExecutor(self.input_controller)
        self.action_space = gym.spaces.Discrete(ACTION_DIM)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBSERVATION_DIM,), np.float32)
        self._decision_mask = np.array([True] + [False] * (ACTION_DIM - 1))
        self.execution_result = ExecutionResult(None, "idle")

    def control_context(self) -> ControlContext:
        player = self.memory.player
        controlled = bool(self._last_frame is not None and player.exists
                          and player.missing_frames == 0 and self.memory.identity_observed
                          and self.memory.player_respawn_timer <= 0.)
        return ControlContext(
            grounded=bool(player.grounded) if controlled else None,
            armed=bool(player.weapon_state > .5) if controlled else None,
            # Resource counters are estimates, not an oracle for legal inputs.
            dodge_ready=None, controllable=controlled, mirrored=self._mirrored,
        )

    def _pack(self, base: np.ndarray) -> np.ndarray:
        now = time.perf_counter()
        self._decision_mask = np.asarray(self.executor.available_actions(self.control_context(), now=now), dtype=bool)
        state = self.executor.observation_state(now=now, mirrored=self._mirrored)
        return np.concatenate((base[:BASE_DIM], state, self._decision_mask)).astype(np.float32)

    def _get_obs(self):
        return self._pack(super()._get_obs())

    def action_masks(self):
        return self._decision_mask.copy()

    def get_observation_spec(self) -> list[str]:
        return (super().get_observation_spec()
                + [f'executor_active_{action.name}' for action in ACTIONS]
                + [f'executor_phase_{phase}' for phase in ('idle', 'dodge', 'press', 'release')]
                + ['executor_remaining', 'executor_dodge_guard', 'executor_mirror_changed']
                + [f'action_allowed_{action.name}' for action in ACTIONS])

    def _get_effective_action(self, action):
        array = np.asarray(action)
        if array.size != 1 or not np.isfinite(array).all():
            raise ValueError("Expected one finite semantic action")
        value = float(array.item())
        if value != int(value) or not 0 <= value < ACTION_DIM:
            raise ValueError(f"Invalid semantic action {value}")
        if not self._decision_mask[int(value)]:
            raise ValueError(f"Semantic action {int(value)} was masked in the sampling observation")
        return int(value)

    def _action_components(self, action):
        row = ACTIONS[action]
        return Components(row.hdir, row.vdir, int(row.button == "space"), int(row.button == "e"),
                          int(row.button == "num4"), int(row.button == "num6"), int(row.button == "num5"))

    def _describe_action(self, action):
        return ACTIONS[action].name

    def _record_action(self, action):
        # Memory history is world-frame. Record actual physical levels, then let
        # observation assembly mirror ONCE; pending GC is not an already-hit attack.
        keys = self.input_controller.held | self.input_controller.taps
        self.memory._prev_action = np.asarray([
            int("d" in keys) - int("a" in keys), int("s" in keys) - int("w" in keys),
            int("space" in keys), int("e" in keys), int("num4" in keys),
            int("num6" in keys), int("num5" in keys),
        ], dtype=np.float32)
        self.memory._prev_action_id = action
        self.memory.player.last_action_id = float(action)

    def _apply_action(self, action, *, emit_tap_actions=True):
        self.input_controller.new_presses.clear()
        ctx, now = self.control_context(), time.perf_counter()
        if self.executor.busy:
            self.execution_result = self.executor.advance(ctx, now=now)
        elif ACTIONS[action].button is None:
            # Walking is a held level, not a pulse followed by a forced neutral tick.
            held, _ = ACTIONS[action].keys(mirrored=self._mirrored)
            self.input_controller.set_pressed(held)
            self.execution_result = ExecutionResult(ACTIONS[action].name, "held")
        else:
            self.execution_result = self.executor.submit(ACTIONS[action].name, ctx, now=now)
        return self.input_controller.new_presses.copy()

    def _release_all_inputs(self):
        if hasattr(self, "executor"):
            self.executor.close()
        super()._release_all_inputs()

    def reset(self, **kwargs):
        self.executor.close()
        return super().reset(**kwargs)

    def close(self):
        # Input cleanup must not depend on the capture backend closing cleanly.
        self._release_all_inputs()
        super().close()

    def step(self, action):
        try:
            obs, reward, terminated, truncated, info = super().step(action)
            if self._last_frame is None:
                obs = self._pack(obs)
            self._last_obs = obs.copy()
            info.update(
                input_contract=self.executor.VERSION,
                semantic_execution_status=self.execution_result.status,
                semantic_execution_reason=self.execution_result.reason,
                semantic_active_action=self.execution_result.action,
                action_mask=self.action_masks(),
            )
            return obs, reward, terminated, truncated, info
        except BaseException:
            self._release_all_inputs()
            raise
