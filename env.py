from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import inspect
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import UI_REGIONS
from feature_extractor.memory.structured_memory import Memory
from feature_extractor.memory.state_spec import StateSpec
from feature_extractor.yolo.extract import Extract
from feature_extractor.yolo.tracker import SortLikeTracker
from reward.extract_rgb import get_rgb
from reward.rgb_to_dmg import get_dmg
from reward.stock import get_stock


KeySet = Iterable[str]


class NullInputController:
    def set_pressed(self, keys: KeySet) -> None:
        return

    def tap(self, keys: KeySet) -> None:
        return

    def reset(self) -> None:
        return


class NullRewardProvider:
    """Reward for training-mode vs easy bot.

    Components (8 total):
        1. dmg_dealt      – positive per opponent HP lost
        2. ko_reward       – bonus per opponent stock taken
        3. ko_penalty      – penalty per self stock lost
        4. game_win        – large terminal bonus
        5. game_loss       – large terminal penalty
        6. weapon_held     – small per-step bonus for holding a weapon
        7. approach        – continuous proximity shaping
        8. proximity_bonus – per-step reward for being in strike range
        9. edge            – mild penalty for being off-platform

    NOTE: dmg_taken removed — it made the agent scared/avoidant.
    KO penalty already punishes dying; per-hit pain caused camping.
    """

    # ── tuneable constants ─────────────────────────────────────────
    DMG_DEALT_COEFF: float = 0.15        # per raw-HP point dealt
    KO_REWARD: float = 15.0              # per opponent stock lost
    KO_PENALTY: float = -5.0             # per self stock lost (softened: less risk-averse)
    GAME_WIN_REWARD: float = 30.0        # terminal: opponent loses last stock
    GAME_LOSS_PENALTY: float = -15.0     # terminal: self loses last stock
    WEAPON_HELD_BONUS: float = 0.005     # per step while holding a weapon
    APPROACH_COEFF: float = 0.05         # continuous proximity: -dist * coeff (stronger pull)
    PROXIMITY_BONUS: float = 0.02        # per step while in strike range
    STRIKE_RANGE: float = 0.18           # normalised distance threshold

    # ── edge penalty (mild, prevents running to blast zone) ────────
    #   Platform: x∈[0.323, 0.674]  y∈[0.570, 0.808]
    EDGE_X_MIN: float = 0.30
    EDGE_X_MAX: float = 0.70
    EDGE_Y_MAX: float = 0.83
    EDGE_COEFF: float = -0.05            # gentle nudge back on stage

    def __init__(self) -> None:
        pass

    def reset(self) -> None:
        """Reset per-episode state (called at env.reset)."""
        pass

    def get_reward_breakdown(self, state, memory: Memory) -> dict[str, float]:
        # ── positions & distances ──────────────────────────────────
        # Use pre-calculated values from memory instead of math.hypot
        curr_dist = float(memory.rel_distance) if (memory.player.exists and memory.opponent.exists) else 0.5
        px, py = memory.player.x, memory.player.y
        ox, oy = memory.opponent.x, memory.opponent.y

        both_exist = memory.player.exists and memory.opponent.exists

        # ── stock changes ─────────────────────────────────────────
        op_stock_lost = max(0.0, memory.prev_op_stocks_left - memory.op_stocks_left)
        self_stock_lost = max(0.0, memory.prev_self_stocks_left - memory.self_stocks_left)

        # ═══ 1. DAMAGE DEALT ══════════════════════════════════════
        dmg_dealt = self.DMG_DEALT_COEFF * float(memory.op_delta_damage)

        # ═══ 2. KO EVENTS ═════════════════════════════════════════
        ko_reward = self.KO_REWARD * op_stock_lost
        ko_penalty = self.KO_PENALTY * self_stock_lost

        # ═══ 3. GAME WIN / LOSS (terminal) ════════════════════════
        game_win = self.GAME_WIN_REWARD if (memory.op_stocks_left <= 0.0 and op_stock_lost > 0.0) else 0.0
        game_loss = self.GAME_LOSS_PENALTY if (memory.self_stocks_left <= 0.0 and self_stock_lost > 0.0) else 0.0

        # ═══ 4. WEAPON HELD ═══════════════════════════════════════
        weapon_held = self.WEAPON_HELD_BONUS if memory.player.weapon_state > 0.0 else 0.0

        # ═══ 5. APPROACH (continuous proximity) ═══════════════════
        approach = 0.0
        if both_exist:
            approach = -self.APPROACH_COEFF * curr_dist

        # ═══ 6. PROXIMITY BONUS (in strike range) ════════════════
        proximity_bonus = 0.0
        if both_exist and curr_dist < self.STRIKE_RANGE:
            proximity_bonus = self.PROXIMITY_BONUS

        # ═══ 7. EDGE PENALTY (mild off-platform nudge) ═══════════
        edge = 0.0
        if memory.player.exists:
            overshoot = 0.0
            if px < self.EDGE_X_MIN:
                overshoot = max(overshoot, (self.EDGE_X_MIN - px) / max(self.EDGE_X_MIN, 1e-6))
            elif px > self.EDGE_X_MAX:
                overshoot = max(overshoot, (px - self.EDGE_X_MAX) / max(1.0 - self.EDGE_X_MAX, 1e-6))
            if py > self.EDGE_Y_MAX:
                overshoot = max(overshoot, (py - self.EDGE_Y_MAX) / max(1.0 - self.EDGE_Y_MAX, 1e-6))
            edge = self.EDGE_COEFF * min(overshoot, 1.0)

        # ═══ TOTAL ════════════════════════════════════════════════
        total = (
            dmg_dealt
            + ko_reward + ko_penalty
            + game_win + game_loss
            + weapon_held
            + approach + proximity_bonus
            + edge
        )

        return {
            "dmg_dealt": float(dmg_dealt),
            "ko_reward": float(ko_reward),
            "ko_penalty": float(ko_penalty),
            "game_win": float(game_win),
            "game_loss": float(game_loss),
            "weapon_held": float(weapon_held),
            "approach": float(approach),
            "proximity_bonus": float(proximity_bonus),
            "edge": float(edge),
            "total_reward": float(total),
        }
    
    def get_reward(self, state, memory: Memory) -> float:
        reward_dict = self.get_reward_breakdown(state, memory)
        return float(reward_dict["total_reward"])

    def update_memory(self, frame, memory: Memory,) -> None:
        return


class DxcamFrameProvider:
    def __init__(self, region: Optional[Tuple[int, int, int, int]] = None,
                 output_idx: int = 0, target_fps: int = 60):
        try:
            import dxcam
        except Exception as exc:
            raise RuntimeError("dxcam is required for DxcamFrameProvider") from exc

        self._camera = dxcam.create(output_idx=output_idx, output_color="BGR")
        # Prefer non-blocking capture mode when available: it returns the latest
        # frame immediately (possibly duplicate) instead of waiting for a fresh one.
        # This is critical for RL control loops where policy steps can be faster
        # than the capture thread.
        if region is None:
            try:
                self._camera.start(target_fps=target_fps, video_mode=True)
            except TypeError:
                self._camera.start(target_fps=target_fps)
        else:
            try:
                self._camera.start(region=region, target_fps=target_fps, video_mode=True)
            except TypeError:
                self._camera.start(region=region, target_fps=target_fps)
        self._last_good_frame = None

    def get_frame(self):
        """Return latest frame, falling back to the last good one if DXCam has no new frame."""
        frame = self._camera.get_latest_frame()
        if frame is not None:
            self._last_good_frame = frame
            return frame
        return self._last_good_frame

    def close(self) -> None:
        self._camera.stop()


class PyDirectInputController:
    _HOLDABLE_KEYS: frozenset[str] = frozenset({"a", "d", "s"})

    def __init__(self):
        try:
            import pydirectinput
        except Exception as exc:
            raise RuntimeError("pydirectinput is required for PyDirectInputController") from exc
        self._pydirectinput = pydirectinput
        if hasattr(self._pydirectinput, "PAUSE"):
            self._pydirectinput.PAUSE = 0
        if hasattr(self._pydirectinput, "FAILSAFE"):
            self._pydirectinput.FAILSAFE = False
        if hasattr(self._pydirectinput, "MINIMUM_DURATION"):
            setattr(self._pydirectinput, "MINIMUM_DURATION", 0)
        if hasattr(self._pydirectinput, "MINIMUM_SLEEP"):
            setattr(self._pydirectinput, "MINIMUM_SLEEP", 0)
        if hasattr(self._pydirectinput, "DARWIN_CATCH_UP_TIME"):
            setattr(self._pydirectinput, "DARWIN_CATCH_UP_TIME", 0)
        self._pressed: set[str] = set()

    def set_pressed(self, keys: KeySet) -> None:
        target = set(keys)

        # Release holdable keys only when they are currently pressed.
        # Avoiding redundant keyUp() calls significantly reduces input overhead.
        for key in self._HOLDABLE_KEYS:
            if key not in target and key in self._pressed:
                self._pydirectinput.keyUp(key)
                self._pressed.discard(key)

        # Release any other tracked key not in target
        for key in list(self._pressed):
            if key not in target and key not in self._HOLDABLE_KEYS:
                self._pydirectinput.keyUp(key)
                self._pressed.discard(key)

        # Press keys that should be held
        for key in target:
            if key not in self._pressed:
                self._pydirectinput.keyDown(key)
                self._pressed.add(key)

    def tap(self, keys: KeySet) -> None:
        for key in keys:
            self._pydirectinput.keyDown(key)
        for key in keys:
            self._pydirectinput.keyUp(key)

    def reset(self) -> None:
        for key in list(self._pressed):
            self._pydirectinput.keyUp(key)
        for key in self._HOLDABLE_KEYS:
            self._pydirectinput.keyUp(key)
        self._pressed.clear()


class PixelStocksHealthProvider:
    def __init__(
        self,
        ui_regions: dict,
        max_health: float = 351.0,
        self_stocks: float = 3.0,
        op_stocks: float = 3.0,
        stock_confirm_frames: int = 2,
        stock_event_cooldown_sec: float = 0.8,
    ):
        self.ui_regions = ui_regions
        self.max_health = max_health
        self._initial_self_stocks = self_stocks
        self._initial_op_stocks = op_stocks
        self.self_stocks_left = self_stocks
        self.op_stocks_left = op_stocks
        self._last_stock_signal = 0
        self._stable_stock_signal = 0
        self._stable_stock_frames = 0
        self._stock_confirm_frames = stock_confirm_frames
        self._stock_event_cooldown_sec = stock_event_cooldown_sec
        self._last_stock_event_time = 0.0
        self._neutral_frames = 0
        self._armed_for_event = True

    def reset(self) -> None:
        self.self_stocks_left = float(self._initial_self_stocks)
        self.op_stocks_left = float(self._initial_op_stocks)
        self._last_stock_signal = 0
        self._stable_stock_signal = 0
        self._stable_stock_frames = 0
        self._last_stock_event_time = 0.0
        self._neutral_frames = 0
        self._armed_for_event = True

    def _read_pixel(self, frame, coord: Tuple[int, int]) -> Optional[np.ndarray]:
        if frame is None:
            return None
        x, y = coord
        if y < 0 or x < 0 or y >= frame.shape[0] or x >= frame.shape[1]:
            return None
        return frame[y, x]

    def __call__(self, frame, detections):
        stock_coord = self.ui_regions.get("stock")
        op_coord = self.ui_regions.get("op")
        agent_coord = self.ui_regions.get("agent")

        if stock_coord is not None:
            stock_pixel = self._read_pixel(frame, stock_coord)
            if stock_pixel is not None:
                stock_rgb = np.asarray(get_rgb(stock_pixel), dtype=np.float32)
                stock_signal = int(get_stock(stock_rgb))
                if stock_signal == self._stable_stock_signal:
                    self._stable_stock_frames += 1
                else:
                    self._stable_stock_signal = stock_signal
                    self._stable_stock_frames = 1

                if stock_signal == 0:
                    self._neutral_frames += 1
                else:
                    self._neutral_frames = 0

                if self._neutral_frames >= max(1, int(self._stock_confirm_frames)):
                    self._armed_for_event = True

                now = time.perf_counter()
                stable_confirmed = self._stable_stock_frames >= max(1, int(self._stock_confirm_frames))
                cooldown_ready = (now - self._last_stock_event_time) >= float(self._stock_event_cooldown_sec)

                if stable_confirmed and stock_signal != 0 and stock_signal != self._last_stock_signal and cooldown_ready and self._armed_for_event:
                    if stock_signal < 0:
                        self.self_stocks_left = max(0.0, self.self_stocks_left - 1.0)
                    else:
                        self.op_stocks_left = max(0.0, self.op_stocks_left - 1.0)
                    self._last_stock_event_time = now
                    self._last_stock_signal = stock_signal
                    self._armed_for_event = False
                if stock_signal == 0:
                    self._last_stock_signal = 0
            else:
                self._last_stock_signal = 0
                self._stable_stock_signal = 0
                self._stable_stock_frames = 0
                self._neutral_frames = 0

        self_health = None
        op_health = None

        if agent_coord is not None:
            agent_pixel = self._read_pixel(frame, agent_coord)
            if agent_pixel is not None:
                agent_rgb = np.asarray(get_rgb(agent_pixel), dtype=np.float32)
                dmg = float(get_dmg(agent_rgb))
                self_health = max(0.0, self.max_health - dmg)

        if op_coord is not None:
            op_pixel = self._read_pixel(frame, op_coord)
            if op_pixel is not None:
                op_rgb = np.asarray(get_rgb(op_pixel), dtype=np.float32)
                dmg = float(get_dmg(op_rgb))
                op_health = max(0.0, self.max_health - dmg)

        return self.self_stocks_left, self.op_stocks_left, self_health, op_health


@dataclass
class EnvConfig:
    max_vel: float = 0.15
    max_weapon_missing: int = 5
    vy_ground_threshold: float = 0.01
    terminate_on_stock_out: bool = True
    ui_regions: Optional[dict] = field(default_factory=lambda: dict(UI_REGIONS))
    yolo_infer_every_n_steps: int = 3
    yolo_max_det: int = 5
    yolo_conf: float = 0.25
    yolo_verbose: bool = False
    yolo_infer_width: int = 640
    yolo_infer_height: int = 360
    use_tracker_layer: bool = True
    tracker_max_missing: int = 8
    tracker_iou_threshold: float = 0.1
    tracker_smooth_alpha: float = 0.6
    temporal_stack_size: int = 1  # 1 = single frame (LSTM handles time)
    temporal_offsets: tuple[int, ...] = (0,)  # only t-0; LSTM does the rest
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
        self.extractor = extractor or Extract(
            max_det=self.config.yolo_max_det,
            verbose=self.config.yolo_verbose,
            conf=self.config.yolo_conf,
            infer_width=self.config.yolo_infer_width,
            infer_height=self.config.yolo_infer_height,
        )
        self.frame_provider = frame_provider or DxcamFrameProvider()
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

        self._tracker: Optional[SortLikeTracker] = None
        if self.config.use_tracker_layer:
            self._tracker = SortLikeTracker(
                max_missing=self.config.tracker_max_missing,
                iou_threshold=self.config.tracker_iou_threshold,
                smooth_alpha=self.config.tracker_smooth_alpha,
            )

        self.memory = Memory()
        self._last_step_time = time.perf_counter()
        self._step_count = 0
        self._last_raw_detections: list = []
        self._last_detections: list = []
        self._step_time_sum = 0.0
        self._step_time_count = 0
        self._action_repeat_remaining = 0
        self._repeated_action = (0, 0, 0, 0)
        self._tap_latch_remaining = {"space": 0, "e": 0, "h": 0, "k": 0, "j": 0}
        self._last_obs: Optional[np.ndarray] = None  # cached for None-frame fallback
        self._last_movement: int = 3     # last movement index (3 = idle)
        self._movement_hold_count: int = 0  # consecutive steps with same movement
        self._max_movement_hold: int = 20   # force release+re-press after this many

        self.action_space = spaces.MultiDiscrete([4, 2, 2, 4])

        offsets = tuple(int(v) for v in self.config.temporal_offsets)
        if len(offsets) == 0:
            offsets = (0,)
        offsets = tuple(max(0, v) for v in offsets)
        self._temporal_offsets = offsets
        offsets = tuple(max(0, v) for v in offsets)
        self._temporal_offsets = offsets
        self._history_len = max(self._temporal_offsets) + 1
        self._state_history: deque[np.ndarray] = deque(maxlen=self._history_len)
        self._reward_sig_cache: dict[str, list[str]] = {}

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

        # Obs dim from StateSpec (single source of truth)
        base_dim = StateSpec.dim()
        obs_dim = base_dim * len(self._temporal_offsets)
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
        base = self._observation_feature_names()
        names: list[str] = []
        for offset in self._temporal_offsets:
            t_name = "t" if offset == 0 else f"t-{offset}"
            for feature in base:
                names.append(f"{t_name}_{feature}")
        return names

    def _sanitize_action(self, action: Sequence[int]) -> tuple[int, int, int, int]:
        movement = int(np.clip(int(action[0]), 0, 3))
        jump = int(np.clip(int(action[1]), 0, 1))
        dodge = int(np.clip(int(action[2]), 0, 1))
        attack = int(np.clip(int(action[3]), 0, 3))

        # Illegal combo mask approximation: no simultaneous dodge + attack
        if dodge == 1 and attack != 0:
            attack = 0

        return movement, jump, dodge, attack

    def _apply_action(self, action: Sequence[int]) -> None:
        movement, jump, dodge, attack = action

        movement_keys = set()
        if movement == 0:
            movement_keys.add("a")
        elif movement == 1:
            movement_keys.add("d")
        elif movement == 2:
            movement_keys.add("s")

        self.input_controller.set_pressed(movement_keys)

        tap_keys = set()
        if jump == 1:
            tap_keys.add("space")
        if dodge == 1:
            tap_keys.add("e")

        if attack == 1:
            tap_keys.add("h")
        elif attack == 2:
            tap_keys.add("k")
        elif attack == 3:
            tap_keys.add("j")

        latch_steps = max(1, int(self.config.tap_latch_steps))
        for key in tap_keys:
            self._tap_latch_remaining[key] = max(self._tap_latch_remaining.get(key, 0), latch_steps)

        latched_tap_keys = set()
        for key in list(self._tap_latch_remaining.keys()):
            remaining = self._tap_latch_remaining[key]
            if remaining > 0:
                latched_tap_keys.add(key)
                self._tap_latch_remaining[key] = remaining - 1

        if latched_tap_keys:
            self.input_controller.tap(latched_tap_keys)

    def _get_effective_action(self, action: Sequence[int]) -> tuple[int, int, int, int]:
        return self._sanitize_action(action)

    def _sample_action_repeat_steps(self) -> int:
        min_steps = max(1, int(getattr(self.config, "action_repeat_min_steps", 1)))
        max_steps = max(min_steps, int(getattr(self.config, "action_repeat_max_steps", min_steps)))
        if max_steps == min_steps:
            return min_steps
        return int(np.random.randint(min_steps, max_steps + 1))

    def _get_detections(self, frame, *, force_infer: bool = False) -> list:
        if frame is None:
            return []

        infer_interval = max(1, int(self.config.yolo_infer_every_n_steps))
        should_infer = force_infer or self._step_count == 0 or (self._step_count % infer_interval == 0)

        if should_infer:
            self._last_raw_detections = self.extractor.predict(frame)

        if self._tracker is not None:
            if should_infer:
                tracker_input = self._last_raw_detections
            else:
                tracker_input = []
            self._last_detections = self._tracker.update(tracker_input)
        elif should_infer:
            self._last_detections = list(self._last_raw_detections)

        return self._last_detections

    def _get_obs(self) -> np.ndarray:
        base_state = self.memory.to_vector()  # already float32

        # Fast path: single-frame obs (LSTM handles temporality)
        if self._history_len <= 1:
            return base_state

        # Multi-frame path (kept for experimentation)
        self._state_history.append(base_state)
        while len(self._state_history) < self._history_len:
            self._state_history.appendleft(base_state)

        history = list(self._state_history)
        current_idx = len(history) - 1
        selected = [history[max(0, current_idx - off)] for off in self._temporal_offsets]
        return np.concatenate(selected, dtype=np.float32)

    def _distance_player_to_weapon(self) -> float:
        """Fast lookup from relational features."""
        if not self.memory.weapon.exists:
            return float("inf")
        return float((self.memory.weapon_dx**2 + self.memory.weapon_dy**2)**0.5)

    def _update_game_logic(self, detections, action_jump: bool, action_dodge: bool) -> None:
        frame = self._last_frame
        self.memory.update_on_ground(vy_threshold=self.config.vy_ground_threshold)

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
        # Preserve weapon state across resets (training mode: game doesn't restart)
        prev_weapon_state = getattr(self, "memory", None) and self.memory.player.weapon_state or 0.0
        self.memory = Memory()
        self.memory.player.weapon_state = prev_weapon_state
        reset_provider_fn = getattr(self.stocks_health_provider, "reset", None)
        if callable(reset_provider_fn):
            reset_provider_fn()
        reset_reward_fn = getattr(self.reward_provider, "reset", None)
        if callable(reset_reward_fn):
            reset_reward_fn()
        self._last_step_time = time.perf_counter()
        self._step_count = 0
        self._last_raw_detections = []
        self._last_detections = []
        self._step_time_sum = 0.0
        self._step_time_count = 0
        self._action_repeat_remaining = 0
        self._repeated_action = (0, 0, 0, 0)
        self._tap_latch_remaining = {"space": 0, "e": 0, "h": 0, "k": 0, "j": 0}
        self._last_movement = 3
        self._movement_hold_count = 0
        self._state_history.clear()
        if self._tracker is not None:
            self._tracker.reset()
        self._last_frame = self.frame_provider.get_frame()
        detections = self._get_detections(self._last_frame, force_infer=True)

        # Use real elapsed dt; fall back to ~47hz estimate for the very first frame.
        now = time.perf_counter()
        reset_dt = max(1e-6, now - self._last_step_time) if self._last_step_time > 0 else 1.0 / 47.0
        self.memory.update_from_detections(detections, dt=reset_dt)
        self._last_step_time = now
        self.reward_provider.update_memory(self._last_frame, self.memory)

        obs = self._get_obs()
        self._last_obs = obs.copy()
        info = {"detections": detections}
        return obs, info

    def step(self, action: Sequence[int]):
        step_start = time.perf_counter()
        effective_action = self._get_effective_action(action)
        repeat_steps = self._sample_action_repeat_steps()

        action_jump = bool(effective_action[1] == 1)
        action_dodge = bool(effective_action[2] == 1)
        action_pick_throw = bool(effective_action[3] == 3)

        reward = 0.0
        reward_breakdown_total: dict[str, float] = {}
        terminated = False
        truncated = False
        detections = []

        for _ in range(repeat_steps):
            self._step_count += 1

            inner_t0 = time.perf_counter()

            t0 = time.perf_counter()
            self._last_frame = self.frame_provider.get_frame()
            frame_grab_dt = time.perf_counter() - t0
            if self._last_frame is None:
                self.input_controller.set_pressed(set())
                obs = self._last_obs if self._last_obs is not None else self._get_obs()
                null_breakdown = {k: 0.0 for k in (
                    "dmg_dealt", "ko_reward", "ko_penalty",
                    "game_win", "game_loss", "weapon_held",
                    "approach", "proximity_bonus", "edge",
                    "total_reward",
                )}
                return obs, 0.0, False, False, {
                    "detections": [],
                    "effective_action": [0, 0, 0, 0],
                    "op_stock_lost_step": 0.0,
                    "self_stock_lost_step": 0.0,
                    "reward_breakdown": null_breakdown,
                }

            movement_idx = int(effective_action[0])
            if movement_idx == self._last_movement and movement_idx != 3:
                self._movement_hold_count += 1
            else:
                self._movement_hold_count = 0
            self._last_movement = movement_idx

            if self._movement_hold_count >= self._max_movement_hold:
                self.input_controller.set_pressed(set())
                self._movement_hold_count = 0

            t0 = time.perf_counter()
            self._apply_action(effective_action)
            apply_action_dt = time.perf_counter() - t0

            t0 = time.perf_counter()
            detections = self._get_detections(self._last_frame)
            detect_dt = time.perf_counter() - t0

            t0 = time.perf_counter()
            step_now = time.perf_counter()
            dt_for_dets = max(1e-6, step_now - self._last_step_time)
            self.memory.update_from_detections(detections, dt=dt_for_dets)
            dist_to_weapon = self._distance_player_to_weapon()
            self.memory.update_player_weapon_from_action(action_pick_throw=action_pick_throw, dist_to_weapon=dist_to_weapon)
            self.memory.update_action(effective_action)
            memory_dt = time.perf_counter() - t0

            t0 = time.perf_counter()
            self._update_game_logic(detections, action_jump=action_jump, action_dodge=action_dodge)

            if not self.memory.player.exists or self.memory.player_respawn_timer > 0.0:
                self.input_controller.set_pressed(set())
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
            "effective_action": [
                int(effective_action[0]),
                int(effective_action[1]),
                int(effective_action[2]),
                int(effective_action[3]),
            ],
            "op_stock_lost_step": op_stock_lost_step,
            "self_stock_lost_step": self_stock_lost_step,
            "player_exists": float(1.0 if self.memory.player.exists else 0.0),
            "player_respawn_timer": float(self.memory.player_respawn_timer),
            "self_delta_damage": float(self.memory.self_delta_damage),
            "op_delta_damage": float(self.memory.op_delta_damage),
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