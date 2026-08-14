from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import inspect
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# MUST precede the Extract import below: that pulls in ultralytics -> torch, and
# loading torch's CUDA libraries makes DXGI duplication impossible on this hardware.
# See capture_first's docstring for the measurement.
import capture_first

from capture import DxcamFrameProvider, MssFrameProvider, create_frame_provider
from config import UI_REGIONS
from control.input import KeySet, NullInputController, PyDirectInputController
from reward.providers import NullRewardProvider
from reward.ui_probe import PixelStocksHealthProvider
from action_space import (
    ACTION_DIM,
    TAP_KEYS,
    Action,
    components as action_components,
    describe as describe_action,
    legal_action_mask,
    sanitize,
    to_keys,
)
from feature_extractor.memory.canonicalize import (
    mirror_dynamic_block,
    mirror_state_vector,
    should_mirror,
)
from feature_extractor.memory.structured_memory import Memory
from feature_extractor.memory.state_spec import StateSpec
from feature_extractor.yolo.extract import Extract
from reward.extract_rgb import get_rgb
from reward.rgb_to_dmg import get_dmg
from reward.stock import get_stock


@dataclass
class EnvConfig:
    terminate_on_stock_out: bool = True
    ui_regions: Optional[dict] = field(default_factory=lambda: dict(UI_REGIONS))
    yolo_infer_every_n_steps: int = 1
    yolo_max_det: int = 8
    yolo_conf: float = 0.25
    yolo_verbose: bool = False
    # Optional pre-resize before the detector. 0 disables it, which is the default:
    # Ultralytics letterboxes to `yolo_imgsz` itself, so resizing here first resamples
    # the frame twice for nothing. Set only if a specific pre-scale is needed.
    yolo_infer_width: int = 0
    yolo_infer_height: int = 0
    # Must match the detector's TRAINING resolution. A 960-trained model fed at 640
    # shrinks the self-indicator below what the P2 head learned to find, and nothing
    # in the pipeline reports the mismatch.
    yolo_imgsz: int = 960
    # Temporal window. The policy is a plain MLP, so history is supplied by
    # stacking rather than recurrence: recurrent PPO is markedly more
    # sample-hungry, and this setup is bounded by wall-clock, not by compute.
    # Only the DYNAMIC block is stacked; slow context is carried once.
    history_offsets: tuple[int, ...] = (2, 4, 8)
    # Capture backend. Default is "dxcam" and it RAISES rather than degrading:
    # DXGI duplication sustains 60+ fps while GDI manages roughly 10-30 at 1080p,
    # and this project is bounded by control rate. A silent fallback would quietly
    # halve the step rate that every design decision here is built around, so the
    # GDI path is opt-in only ("mss"), never automatic.
    capture_backend: str = "dxcam"
    # Horizontally mirror the observation so the opponent is always on the same
    # side. Roughly halves the state space the policy must cover.
    canonicalize_observation: bool = True
    canonicalize_deadband: float = 0.03
    profile_step_timing: bool = False
    profile_window_size: int = 120
    emit_detailed_info: bool = False
    action_repeat_steps: int = 1
    action_repeat_min_steps: int = 4
    action_repeat_max_steps: int = 6
    tap_latch_steps: int = 1
    max_episode_steps: int = 0  # truncate after this many steps (0 = no limit)


class BrawlDeepEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        extractor: Optional[Extract] = None,
        frame_provider: Optional[DxcamFrameProvider] = None,
        input_controller: Optional[NullInputController] = None,
        reward_provider: Optional[NullRewardProvider] = None,
        ground_contact_provider: Optional[Callable[..., Tuple[bool, bool]]] = None,
        opponent_dodge_detector: Optional[Callable[..., bool]] = None,
        recovery_provider: Optional[Callable[..., Tuple[Optional[bool], Optional[bool]]]] = None,
        stocks_health_provider: Optional[Callable[..., Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]] = None,
        config: Optional[EnvConfig] = None,
    ):
        super().__init__()

        self.config = config or EnvConfig()

        # ORDER MATTERS: capture must be created BEFORE the detector.
        #
        # On a hybrid-graphics laptop the display output belongs to the Intel iGPU
        # while CUDA runs on the NVIDIA dGPU. Initialising CUDA/TensorRT first binds
        # the process to the NVIDIA adapter, after which duplicating the Intel-owned
        # output fails with DXGI_ERROR_UNSUPPORTED (0x887A0004) -- reported only as
        # "device interface or feature level is not supported", which reads like a
        # driver problem rather than an ordering one. Grabbing the duplicator first
        # avoids it entirely; CUDA initialises fine afterwards.
        self.frame_provider = frame_provider or create_frame_provider(
            prefer=self.config.capture_backend
        )

        self.extractor = extractor or Extract(
            max_det=self.config.yolo_max_det,
            verbose=self.config.yolo_verbose,
            conf=self.config.yolo_conf,
            infer_width=self.config.yolo_infer_width,
            infer_height=self.config.yolo_infer_height,
            imgsz=self.config.yolo_imgsz,
        )
        self.input_controller = input_controller or PyDirectInputController()
        self.reward_provider = reward_provider or NullRewardProvider()
        self.ground_contact_provider = ground_contact_provider
        self.opponent_dodge_detector = opponent_dodge_detector
        self.recovery_provider = recovery_provider
        if stocks_health_provider is not None:
            self.stocks_health_provider = stocks_health_provider
        elif self.config.ui_regions is not None:
            self.stocks_health_provider = PixelStocksHealthProvider(ui_regions=self.config.ui_regions)
        else:
            self.stocks_health_provider = None

        self.memory = Memory()
        self._last_step_time = time.perf_counter()
        self._step_count = 0
        self._last_detections: list = []
        self._step_time_sum = 0.0
        self._step_time_count = 0
        self._tap_latch_remaining = {key: 0 for key in sorted(TAP_KEYS)}
        self._last_obs: Optional[np.ndarray] = None  # cached for None-frame fallback
        self._last_movement: int = 0     # last horizontal direction (0 = neutral)
        self._movement_hold_count: int = 0  # consecutive steps with same movement
        self._max_movement_hold: int = 20   # force release+re-press after this many

        # One categorical head over the full control grammar. A factorised space
        # would model direction and attack as independent, which they are not: the
        # direction IS part of the move. See action_space.py.
        self.action_space = spaces.Discrete(ACTION_DIM)

        # Temporal window over the dynamic block only.
        offsets = tuple(sorted({max(1, int(v)) for v in self.config.history_offsets}))
        self._history_offsets = offsets
        self._history_len = (max(offsets) + 1) if offsets else 1
        self._dynamic_dim = StateSpec.dynamic_dim()
        self._state_history: deque[np.ndarray] = deque(maxlen=self._history_len)
        self._reward_sig_cache: dict[str, list[str]] = {}

        # Canonicalisation state. `_mirrored` is sticky across steps so the
        # decision has hysteresis and the observation cannot chatter.
        self._canonicalize = bool(self.config.canonicalize_observation)
        self._mirrored = False

        # Fine-grained step profiler (enabled when profile_step_timing=True).
        self._perf_inner_frames = 0
        self._perf_frame_grab_sum = 0.0
        self._perf_apply_action_sum = 0.0
        self._perf_detect_sum = 0.0
        self._perf_memory_sum = 0.0
        self._perf_logic_sum = 0.0
        self._perf_reward_sum = 0.0
        self._perf_inner_total_sum = 0.0
        self._perf_inner_report_every = 500

        # Obs dim from StateSpec (single source of truth):
        # [ core(t) | dynamic(t-k) for each history offset ]
        obs_dim = StateSpec.observation_dim(self._history_offsets)
        self._obs_buffer = np.zeros((obs_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    @staticmethod
    def _observation_feature_names() -> list[str]:
        """Feature names matching Memory.to_vector() / StateSpec layout."""
        return StateSpec.names()

    def get_observation_spec(self) -> list[str]:
        return StateSpec.observation_names(self._history_offsets)

        # Illegal combo mask approximation: no simultaneous dodge + attack
        if dodge == 1 and attack != 0:
            attack = 0

        return movement, jump, dodge, attack

    def _apply_action(self, action: int, *, emit_tap_actions: bool = True) -> None:
        """Translate a canonical action into physical key state.

        `to_keys` is the single place the mirror touches the action path: the policy
        chose in canonical (toward/away) space, and this converts to left/right.
        """
        held, tapped = to_keys(int(action), mirrored=self._mirrored)
        self.input_controller.set_pressed(held)

        if emit_tap_actions:
            latch_steps = max(1, int(self.config.tap_latch_steps))
            for key in tapped:
                self._tap_latch_remaining[key] = max(self._tap_latch_remaining.get(key, 0), latch_steps)

        # Hold each tap key down for its whole latch window, then release it.
        # A press-release pair inside one step lasts well under the game's ~16.7ms
        # input poll and is often dropped entirely.
        for key in list(self._tap_latch_remaining.keys()):
            remaining = self._tap_latch_remaining[key]
            if remaining > 0:
                self.input_controller.key_down(key)
                self._tap_latch_remaining[key] = remaining - 1
            else:
                self.input_controller.key_up(key)

    def _release_all_inputs(self) -> None:
        """Release movement holds and any tap key still inside its hold window.

        Used when the agent is not in control (no frame, dead, respawning), so a
        latched tap cannot stay physically held down across the gap.
        """
        self.input_controller.set_pressed(set())
        for key in list(self._tap_latch_remaining.keys()):
            self._tap_latch_remaining[key] = 0
            self.input_controller.key_up(key)

    def _get_effective_action(self, action) -> int:
        """Coerce the policy output into a valid action index.

        Unlike the old factorised space there is nothing to un-mirror here: actions
        are canonical, and the physical conversion happens in `_apply_action`.
        """
        return sanitize(action)

    def action_masks(self) -> np.ndarray:
        """Legal-action mask, in the convention masking-aware algorithms expect.

        The game already ignores impossible inputs, so this is not needed for
        correctness. Its value is exploration: with one categorical head the illegal
        logits can be zeroed before the softmax instead of being sampled and wasted.
        """
        mem = self.memory
        weapon_in_range = bool(
            mem.weapon.exists
            and mem.closest_weapon_distance <= mem.physics.pickup_distance_norm
        )
        return legal_action_mask(
            dodge_available=bool(mem.player.dodge_available),
            jumps_left=float(mem.player.jumps_left),
            weapon_in_range=weapon_in_range,
            has_weapon=bool(mem.player.weapon_state > 0.0),
            grounded=bool(mem.player.grounded),
        )

    def _sample_action_repeat_steps(self) -> int:
        fixed_steps = int(getattr(self.config, "action_repeat_steps", 0))
        if fixed_steps > 0:
            return fixed_steps

        min_steps = max(1, int(getattr(self.config, "action_repeat_min_steps", 1)))
        max_steps = max(min_steps, int(getattr(self.config, "action_repeat_max_steps", min_steps)))
        if max_steps == min_steps:
            return min_steps
        return int(np.random.randint(min_steps, max_steps + 1))

    def _get_detections(self, frame, *, force_infer: bool = False) -> list:
        if frame is None:
            return []
        self._last_detections = self.extractor.predict(frame)
        return self._last_detections

    def _update_mirror_decision(self, state: np.ndarray) -> None:
        """Refresh the sticky mirror decision from the current world-frame state."""
        if not self._canonicalize:
            self._mirrored = False
            return

        self._mirrored = should_mirror(
            rel_dx=StateSpec.get(state, "rel_dx"),
            opponent_exists=StateSpec.get(state, "opponent_exists") > 0.5,
            signed_dx_to_stage_center=StateSpec.get(state, "signed_dx_to_stage_center"),
            previous=self._mirrored,
            deadband=float(self.config.canonicalize_deadband),
        )

    def _get_obs(self) -> np.ndarray:
        # Memory emits the world frame; canonicalise before anything stores it so
        # the history window is entirely in one frame of reference.
        world_state = self.memory.to_vector()
        self._update_mirror_decision(world_state)

        # History is retained in the WORLD frame and mirrored at assembly time.
        # Storing it pre-mirrored would mix slices canonicalised under an older
        # decision with the current one whenever the mirror flips mid-episode.
        world_dynamic = world_state[: self._dynamic_dim].copy()
        self._state_history.append(world_dynamic)
        while len(self._state_history) < self._history_len:
            self._state_history.appendleft(world_dynamic)

        state = mirror_state_vector(world_state) if self._mirrored else world_state.copy()

        buf = self._obs_buffer
        core_dim = StateSpec.dim()
        buf[:core_dim] = state

        history = list(self._state_history)
        newest = len(history) - 1
        cursor = core_dim
        for offset in self._history_offsets:
            past = history[max(0, newest - offset)]
            if self._mirrored:
                past = mirror_dynamic_block(past)
            buf[cursor : cursor + self._dynamic_dim] = past
            cursor += self._dynamic_dim

        return buf

    def _distance_player_to_weapon(self) -> float:
        """Fast lookup from relational features."""
        if not self.memory.weapon.exists:
            return float("inf")
        dx = float(self.memory.weapon.x - self.memory.player.x)
        dy = float(self.memory.weapon.y - self.memory.player.y)
        return float((dx**2 + dy**2)**0.5)

    def _update_game_logic(self, detections, action_jump: bool, action_dodge: bool) -> None:
        frame = self._last_frame
        self.memory.update_on_ground()

        dt = max(1e-6, time.perf_counter() - self._last_step_time)
        self._last_step_time = time.perf_counter()

        opponent_dodge_detected = False
        if self.opponent_dodge_detector is not None:
            opponent_dodge_detected = bool(self.opponent_dodge_detector(detections, frame))

        self.memory.update_dodge_cooldowns(dt, action_dodge, opponent_dodge_detected)
        self.memory.update_jumps(action_jump)
        self.memory.update_existence_from_stocks(dt)
        self.memory.update_hitstun(dt)

        if self.stocks_health_provider is not None:
            self_stocks_left, op_stocks_left, self_health, op_health = self.stocks_health_provider(frame, detections)
            self.memory.update_stocks_and_health(
                self_stocks_left=self_stocks_left,
                self_health=self_health,
                op_stocks_left=op_stocks_left,
                op_health=op_health,
            )
            self._enforce_health_detection_consistency(self_health=self_health, op_health=op_health)

    def _enforce_health_detection_consistency(self, self_health: Optional[float], op_health: Optional[float]) -> None:
        # If health is readable from UI, the character is on-screen even if YOLO missed this frame.
        if self_health is not None and self.memory.self_stocks_left > 0.0 and self.memory.player_respawn_timer <= 0.0:
            self.memory.player.exists = True
            self.memory.player.missing_frames = 0
            self.memory.player.confidence = max(self.memory.player.confidence, 0.2)

        if op_health is not None and self.memory.op_stocks_left > 0.0 and self.memory.opponent_respawn_timer <= 0.0:
            self.memory.opponent.exists = True
            self.memory.opponent.missing_frames = 0
            self.memory.opponent.confidence = max(self.memory.opponent.confidence, 0.2)

    def observation_to_dict(self, obs: Optional[np.ndarray] = None) -> dict[str, float]:
        """Lazy dictionary conversion for profiling/debugging only."""
        if not self.config.emit_detailed_info:
            return {}
        if obs is None:
            obs = self._get_obs()
        feature_names = self.get_observation_spec()
        return {name: float(value) for name, value in zip(feature_names, obs)}

    def _call_reward_method(self, method_name: str, detections) -> Any:
        if not hasattr(self.reward_provider, method_name):
            return None

        reward_method = getattr(self.reward_provider, method_name)
        
        # Cache inspect.signature as it is extremely slow to call per frame
        if method_name not in self._reward_sig_cache:
            reward_sig = inspect.signature(reward_method)
            self._reward_sig_cache[method_name] = list(reward_sig.parameters.keys())
            
        reward_params = self._reward_sig_cache[method_name]

        kwargs = {}
        for param in reward_params:
            if param == "state":
                kwargs[param] = self.memory.player
            elif param == "memory":
                kwargs[param] = self.memory
            elif param == "frame":
                kwargs[param] = self._last_frame
            elif param == "detections":
                kwargs[param] = detections

        if len(kwargs) == len(reward_params):
            return reward_method(**kwargs)

        # Fallbacks for legacy positional providers
        if len(reward_params) >= 2 and reward_params[0] == "state" and reward_params[1] == "memory":
            return reward_method(self.memory.player, self.memory)
        if len(reward_params) >= 3 and reward_params[0] == "frame" and reward_params[1] == "detections" and reward_params[2] == "memory":
            return reward_method(self._last_frame, detections, self.memory)
        if len(reward_params) >= 2 and reward_params[0] == "frame" and reward_params[1] == "memory":
            return reward_method(self._last_frame, self.memory)
        if len(reward_params) >= 1 and reward_params[0] == "memory":
            return reward_method(self.memory)

        return reward_method(self._last_frame, detections, self.memory)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.input_controller.reset()
        # Preserve match-state continuity across episode resets (training mode: game doesn't restart).
        prev_mem = getattr(self, "memory", None)
        prev_weapon_state = prev_mem.player.weapon_state if prev_mem is not None else 0.0
        prev_self_stocks = prev_mem.self_stocks_left if prev_mem is not None else 3.0
        prev_op_stocks = prev_mem.op_stocks_left if prev_mem is not None else 3.0
        prev_self_health = prev_mem.self_health if prev_mem is not None else 351.0
        prev_op_health = prev_mem.op_health if prev_mem is not None else 351.0
        prev_player_respawn_timer = prev_mem.player_respawn_timer if prev_mem is not None else 0.0
        prev_opponent_respawn_timer = prev_mem.opponent_respawn_timer if prev_mem is not None else 0.0
        prev_self_stock_lost = (
            max(0.0, float(prev_mem.prev_self_stocks_left - prev_mem.self_stocks_left))
            if prev_mem is not None
            else 0.0
        )

        # If previous episode ended around a death/respawn transition, the agent must respawn unarmed.
        if prev_player_respawn_timer > 1e-6 or prev_self_stock_lost > 0.0:
            prev_weapon_state = 0.0

        self.memory = Memory()
        self.memory.player.weapon_state = prev_weapon_state
        self.memory.self_stocks_left = float(prev_self_stocks)
        self.memory.prev_self_stocks_left = float(prev_self_stocks)
        self.memory.op_stocks_left = float(prev_op_stocks)
        self.memory.prev_op_stocks_left = float(prev_op_stocks)
        self.memory.self_health = float(prev_self_health)
        self.memory.prev_self_health = float(prev_self_health)
        self.memory.op_health = float(prev_op_health)
        self.memory.prev_op_health = float(prev_op_health)
        self.memory.player.damage_percent = max(0.0, min(1.0, (self.memory.max_health - self.memory.self_health) / self.memory.max_health))
        self.memory.opponent.damage_percent = max(0.0, min(1.0, (self.memory.max_health - self.memory.op_health) / self.memory.max_health))
        self.memory.player.health = self.memory.self_health
        self.memory.opponent.health = self.memory.op_health
        self.memory.player.stocks = self.memory.self_stocks_left
        self.memory.opponent.stocks = self.memory.op_stocks_left
        self.memory.player_respawn_timer = float(prev_player_respawn_timer)
        self.memory.opponent_respawn_timer = float(prev_opponent_respawn_timer)
        self.memory.self_delta_damage = 0.0
        self.memory.op_delta_damage = 0.0
        self.memory.just_hit_opponent = 0.0
        self.memory.just_got_hit = 0.0

        reset_provider_fn = getattr(self.stocks_health_provider, "reset", None)
        if callable(reset_provider_fn):
            try:
                reset_provider_fn(preserve_match_state=True)
            except TypeError:
                reset_provider_fn()
        reset_reward_fn = getattr(self.reward_provider, "reset", None)
        if callable(reset_reward_fn):
            reset_reward_fn()
        self._last_step_time = time.perf_counter()
        self._step_count = 0
        self._last_detections = []
        self._step_time_sum = 0.0
        self._step_time_count = 0
        self._tap_latch_remaining = {key: 0 for key in sorted(TAP_KEYS)}
        self._last_movement = 0
        self._movement_hold_count = 0
        self._state_history.clear()
        self._mirrored = False
        self._last_frame = self.frame_provider.get_frame()
        detections = self._get_detections(self._last_frame, force_infer=True)

        # Use real elapsed dt; fall back to ~47hz estimate for the very first frame.
        now = time.perf_counter()
        reset_dt = max(1e-6, now - self._last_step_time) if self._last_step_time > 0 else 1.0 / 47.0
        self.memory.update_from_detections(
            detections,
            dt=reset_dt,
        )
        self._last_step_time = now

        # Re-sync health/stocks from UI on reset without introducing artificial stock-loss events.
        if self.stocks_health_provider is not None:
            self_stocks_left, op_stocks_left, self_health, op_health = self.stocks_health_provider(self._last_frame, detections)
            if self_stocks_left is not None:
                v = max(0.0, min(float(self_stocks_left), self.memory.max_stocks))
                self.memory.self_stocks_left = v
                self.memory.prev_self_stocks_left = v
                self.memory.player.stocks = v
                if v + 1e-6 < float(prev_self_stocks):
                    self.memory.player.weapon_state = 0.0
            if op_stocks_left is not None:
                v = max(0.0, min(float(op_stocks_left), self.memory.max_stocks))
                self.memory.op_stocks_left = v
                self.memory.prev_op_stocks_left = v
                self.memory.opponent.stocks = v
            if self_health is not None:
                h = max(0.0, min(float(self_health), self.memory.max_health))
                self.memory.self_health = h
                self.memory.prev_self_health = h
                self.memory.player.health = h
                self.memory.player.damage_percent = max(0.0, min(1.0, (self.memory.max_health - h) / self.memory.max_health))
            if op_health is not None:
                h = max(0.0, min(float(op_health), self.memory.max_health))
                self.memory.op_health = h
                self.memory.prev_op_health = h
                self.memory.opponent.health = h
                self.memory.opponent.damage_percent = max(0.0, min(1.0, (self.memory.max_health - h) / self.memory.max_health))
            self.memory.self_delta_damage = 0.0
            self.memory.op_delta_damage = 0.0
            self.memory.just_hit_opponent = 0.0
            self.memory.just_got_hit = 0.0

        if self.memory.player_respawn_timer > 1e-6:
            self.memory.player.weapon_state = 0.0

        self.reward_provider.update_memory(self._last_frame, self.memory)

        obs = self._get_obs()
        self._last_obs = obs.copy()
        info = {
            "detections": detections,
            "canon_mirrored": float(1.0 if self._mirrored else 0.0),
            "action_mask": self.action_masks(),
            "player_exists": float(1.0 if self.memory.player.exists else 0.0),
            "player_respawn_timer": float(self.memory.player_respawn_timer),
            "self_stock_lost_step": 0.0,
            "op_stock_lost_step": 0.0,
        }
        return obs, info

    def step(self, action: Sequence[int]):
        step_start = time.perf_counter()
        effective_action = self._get_effective_action(action)
        repeat_steps = self._sample_action_repeat_steps()

        comp = action_components(effective_action)
        action_jump = bool(comp.jump)
        action_dodge = bool(comp.dodge)
        action_pick_throw = bool(comp.interact)

        reward = 0.0
        reward_breakdown_total: dict[str, float] = {}
        terminated = False
        truncated = False
        detections = []

        for repeat_idx in range(repeat_steps):
            self._step_count += 1
            first_repeat_frame = bool(repeat_idx == 0)
            frame_jump = bool(action_jump and first_repeat_frame)
            frame_dodge = bool(action_dodge and first_repeat_frame)
            frame_pick_throw = bool(action_pick_throw and first_repeat_frame)

            inner_t0 = time.perf_counter()

            t0 = time.perf_counter()
            self._last_frame = self.frame_provider.get_frame()
            frame_grab_dt = time.perf_counter() - t0
            if self._last_frame is None:
                self._release_all_inputs()
                obs = self._last_obs if self._last_obs is not None else self._get_obs()
                null_breakdown = {k: 0.0 for k in (
                    "dmg_dealt", "ko_reward", "ko_penalty",
                    "game_win", "game_loss", "weapon_held",
                    "approach", "proximity_bonus", "edge",
                    "total_reward",
                )}
                return obs, 0.0, False, False, {
                    "detections": [],
                    "effective_action": int(Action.NOOP),
                    "op_stock_lost_step": 0.0,
                    "self_stock_lost_step": 0.0,
                    "reward_breakdown": null_breakdown,
                }

            movement_idx = int(comp.hdir)
            if movement_idx == self._last_movement and movement_idx != 0:
                self._movement_hold_count += 1
            else:
                self._movement_hold_count = 0
            self._last_movement = movement_idx

            if self._movement_hold_count >= self._max_movement_hold:
                self.input_controller.set_pressed(set())
                self._movement_hold_count = 0

            t0 = time.perf_counter()
            self._apply_action(effective_action, emit_tap_actions=first_repeat_frame)
            apply_action_dt = time.perf_counter() - t0

            t0 = time.perf_counter()
            detections = self._get_detections(self._last_frame)
            detect_dt = time.perf_counter() - t0

            t0 = time.perf_counter()
            step_now = time.perf_counter()
            dt_for_dets = max(1e-6, step_now - self._last_step_time)
            self.memory.update_from_detections(
                detections,
                dt=dt_for_dets,
            )
            dist_to_weapon = self._distance_player_to_weapon()
            self.memory.update_player_weapon_from_action(action_pick_throw=frame_pick_throw, dist_to_weapon=dist_to_weapon)
            self.memory.update_action(effective_action)
            memory_dt = time.perf_counter() - t0

            t0 = time.perf_counter()
            self._update_game_logic(detections, action_jump=frame_jump, action_dodge=frame_dodge)

            if not self.memory.player.exists or self.memory.player_respawn_timer > 0.0:
                self._release_all_inputs()
            logic_dt = time.perf_counter() - t0

            t0 = time.perf_counter()
            self.reward_provider.update_memory(self._last_frame, self.memory)

            reward_breakdown = self._call_reward_method("get_reward_breakdown", detections)
            if isinstance(reward_breakdown, dict):
                frame_reward = float(reward_breakdown.get("total_reward", 0.0))
            else:
                reward_raw = self._call_reward_method("get_reward", detections)
                frame_reward = float(reward_raw) if reward_raw is not None else 0.0
                reward_breakdown = {"total_reward": frame_reward}
            reward_dt = time.perf_counter() - t0

            if self.config.profile_step_timing:
                inner_total_dt = time.perf_counter() - inner_t0
                self._perf_inner_frames += 1
                self._perf_frame_grab_sum += frame_grab_dt
                self._perf_apply_action_sum += apply_action_dt
                self._perf_detect_sum += detect_dt
                self._perf_memory_sum += memory_dt
                self._perf_logic_sum += logic_dt
                self._perf_reward_sum += reward_dt
                self._perf_inner_total_sum += inner_total_dt

                if self._perf_inner_frames % self._perf_inner_report_every == 0:
                    denom = float(self._perf_inner_frames)
                    avg_total = self._perf_inner_total_sum / denom
                    avg_frame = self._perf_frame_grab_sum / denom
                    avg_apply = self._perf_apply_action_sum / denom
                    avg_detect = self._perf_detect_sum / denom
                    avg_memory = self._perf_memory_sum / denom
                    avg_logic = self._perf_logic_sum / denom
                    avg_reward = self._perf_reward_sum / denom
                    avg_other = max(0.0, avg_total - (avg_frame + avg_apply + avg_detect + avg_memory + avg_logic + avg_reward))
                    print(
                        f"[BrawlDeepEnv] avg inner frame over {self._perf_inner_frames}: "
                        f"total={avg_total * 1000:.2f}ms ({1.0 / max(1e-9, avg_total):.2f} hz), "
                        f"frame={avg_frame * 1000:.2f}ms, apply={avg_apply * 1000:.2f}ms, "
                        f"detect={avg_detect * 1000:.2f}ms, memory={avg_memory * 1000:.2f}ms, "
                        f"logic={avg_logic * 1000:.2f}ms, reward={avg_reward * 1000:.2f}ms, "
                        f"other={avg_other * 1000:.2f}ms"
                    )

            reward += frame_reward
            if isinstance(reward_breakdown, dict):
                for key, value in reward_breakdown.items():
                    reward_breakdown_total[key] = reward_breakdown_total.get(key, 0.0) + float(value)

            if self.config.terminate_on_stock_out:
                terminated = self.memory.self_stocks_left <= 0.0 or self.memory.op_stocks_left <= 0.0

            truncated = (
                self.config.max_episode_steps > 0
                and self._step_count >= self.config.max_episode_steps
            )
            if terminated or truncated:
                break

        obs = self._get_obs()
        self._last_obs = obs.copy()  # cache for None-frame fallback
        op_stock_lost_step = float(max(0.0, self.memory.prev_op_stocks_left - self.memory.op_stocks_left))
        self_stock_lost_step = float(max(0.0, self.memory.prev_self_stocks_left - self.memory.self_stocks_left))
        info: dict[str, Any] = {
            "detections": detections,
            "effective_action": int(effective_action),
            "action_name": describe_action(effective_action),
            "canon_mirrored": float(1.0 if self._mirrored else 0.0),
            "action_mask": self.action_masks(),
            "op_stock_lost_step": op_stock_lost_step,
            "self_stock_lost_step": self_stock_lost_step,
            "player_exists": float(1.0 if self.memory.player.exists else 0.0),
            "player_respawn_timer": float(self.memory.player_respawn_timer),
            "self_delta_damage": float(self.memory.self_delta_damage),
            "op_delta_damage": float(self.memory.op_delta_damage),
            # Raw match state as the memory holds it when the observation is
            # assembled. The deltas above are computed inside the inner frame loop
            # while the observation is built once at the end, so a health reading
            # that recovers before assembly produces reward with no visible state
            # change. Logging both makes that divergence measurable instead of
            # inferred -- see docs/tasks/damage-observation.md.
            "op_health": float(self.memory.op_health),
            "self_health": float(self.memory.self_health),
            "op_stocks_left": float(self.memory.op_stocks_left),
            "self_stocks_left": float(self.memory.self_stocks_left),
            "player_weapon_state": float(self.memory.player.weapon_state),
            "weapon_visible_this_frame": float(1.0 if self.memory.weapon_visible_this_frame else 0.0),
            "weapon_pickup_action": float(1.0 if self.memory.weapon_pickup_action_this_frame else 0.0),
            "weapon_drop_action": float(1.0 if self.memory.weapon_drop_action_this_frame else 0.0),
            "reward_breakdown": reward_breakdown_total if reward_breakdown_total else {"total_reward": reward},
            "frame_skip": int(repeat_steps),
        }
        if self.config.emit_detailed_info:
            info["observation_state"] = self.observation_to_dict(obs)
            info["reward"] = reward_breakdown_total if reward_breakdown_total else {"total_reward": reward}

        step_time_taken = time.perf_counter() - step_start
        if self.config.profile_step_timing:
            self._step_time_sum += step_time_taken
            self._step_time_count += 1
            window_size = max(1, int(self.config.profile_window_size))
            if self._step_time_count >= window_size:
                avg_step_time = self._step_time_sum / self._step_time_count
                info["perf"] = {
                    "avg_step_time_sec": float(avg_step_time),
                    "avg_step_hz": float(1.0 / max(1e-9, avg_step_time)),
                    "window_steps": int(self._step_time_count),
                    "yolo_infer_every_n_steps": int(max(1, int(self.config.yolo_infer_every_n_steps))),
                    "yolo_max_det": int(self.config.yolo_max_det),
                    "action_repeat_steps": int(repeat_steps),
                    "action_repeat_min_steps": int(max(1, int(getattr(self.config, "action_repeat_min_steps", 1)))),
                    "action_repeat_max_steps": int(max(1, int(getattr(self.config, "action_repeat_max_steps", 1)))),
                    "tap_latch_steps": int(max(1, int(self.config.tap_latch_steps))),
                }
                self._step_time_sum = 0.0
                self._step_time_count = 0

        return obs, reward, terminated, truncated, info

    def close(self):
        close_fn = getattr(self.frame_provider, "close", None)
        if callable(close_fn):
            close_fn()
        self.input_controller.reset()


# Re-exported so existing imports keep working after the split.
__all__ = [
    "BrawlDeepEnv",
    "EnvConfig",
    "KeySet",
    "NullInputController",
    "PyDirectInputController",
    "NullRewardProvider",
    "PixelStocksHealthProvider",
    "DxcamFrameProvider",
    "MssFrameProvider",
    "create_frame_provider",
]
