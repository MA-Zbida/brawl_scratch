"""Original discrete27 execution, guarded against uncertain player identity."""
from __future__ import annotations

from env import BrawlDeepEnv
from train.primitive_training import HISTORY_OFFSETS


class PrimitiveBrawlEnv(BrawlDeepEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if tuple(self.config.history_offsets) != HISTORY_OFFSETS or self.config.action_repeat_steps != 1:
            self.close()
            raise ValueError('Primitive27 requires history (2,4,8) and action_repeat_steps=1')
        self.input_blocked = False

    def _apply_action(self, action, *, emit_tap_actions=True):
        self.input_blocked = bool(not self.memory.identity_observed or not self.memory.player.exists
            or self.memory.player.missing_frames > 0 or self.memory.player_respawn_timer > 0)
        if self.input_blocked:
            self._release_all_inputs()
            return set()
        return super()._apply_action(action, emit_tap_actions=emit_tap_actions)

    def step(self, action):
        self.input_blocked = False
        try:
            obs, reward, terminated, truncated, info = super().step(action)
            info['input_execution_blocked'] = bool(self.input_blocked or self._last_frame is None)
            return obs, reward, terminated, truncated, info
        except BaseException:
            self._release_all_inputs()
            raise

    def close(self):
        self._release_all_inputs()
        super().close()
