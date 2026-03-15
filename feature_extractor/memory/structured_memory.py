from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from feature_extractor.memory.utils import bbox_center, clamp, euclidian
from feature_extractor.memory.state_spec import StateSpec


@dataclass
class FighterState:
    exists: bool = True
    x: float = 0.5
    y: float = 0.5
    vx: float = 0.0
    vy: float = 0.0
    last_x: float = 0.5
    last_y: float = 0.5
    grounded: bool = False
    damage_percent: float = 0.0
    weapon_state: float = 0.0
    dodge_available: bool = True
    last_action_id: float = 0.0
    action_cooldown_remaining: float = 0.0
    confidence: float = 0.0
    missing_frames: int = 0
    jumps_left: int = 3
    on_edge: bool = False
    off_stage: bool = False


@dataclass
class WeaponState:
    exists: bool = False
    x: float = 0.5
    y: float = 0.5

# Platform geometry (normalised)
PLATFORM_X_MIN = 0.315
PLATFORM_X_MAX = 0.683
PLATFORM_Y_MIN = 0.5527
PLATFORM_Y_MAX = 0.8149
MID_X = (PLATFORM_X_MAX + PLATFORM_X_MIN) / 2.0

# Player Geometry
HEIGHT = 0.0722
WIDTH = 10 / 1920
PLAYER_MARKER_Y_OFFSET = HEIGHT / 2.0

# Hitstun constants (Brawlhalla-approximate)
HITSTUN_MAX_FRAMES = 30   # ~0.5s at 60fps — max hitstun duration
AIRBORNE_MAX_FRAMES = 60  # normalise to 1s
DODGE_COOLDOWN_AIR = 3.2
DODGE_COOLDOWN_GROUNDED = 1.0
RESPAWN_DURATION_SECONDS = 4.7
GROUND_Y_TOLERANCE = 0.028
MAX_WEAPON_MISSING_FRAMES = 4

def _vertical_depth(py: float, Y_STAGE: float) -> float:
        """Positive if below stage level."""
        return max(0.0, Y_STAGE - (py + HEIGHT / 2.0))

def _nearest_ledge(px: float, Y_STAGE: float, X_MIN: float, X_MAX: float) -> tuple:
    """Return coordinates of nearest ledge."""
    left = (X_MIN, Y_STAGE)
    right = (X_MAX, Y_STAGE)

    if abs(px - X_MIN) < abs(px - X_MAX):
        return left
    return right

class Memory:
    def __init__(self):
        self.max_health = 351.0
        self.max_stocks = 3.0
        self.min_xy = 0.0
        self.max_xy = 1.0

        self.player = FighterState()
        self.opponent = FighterState(exists=False)
        self.weapon = WeaponState()
        self._weapon_missing_frames = 0

        self.self_stocks_left = self.max_stocks
        self.op_stocks_left = self.max_stocks
        self.prev_self_stocks_left = self.self_stocks_left
        self.prev_op_stocks_left = self.op_stocks_left

        self.self_health = self.max_health
        self.op_health = self.max_health
        self.prev_self_health = self.self_health
        self.prev_op_health = self.op_health

        self.self_delta_damage = 0.0
        self.op_delta_damage = 0.0

        self.self_total_damage_taken_before_stock_loss = 0.0
        self.op_total_damage_done_before_stock_loss = 0.0

        self.just_hit_opponent = 0.0
        self.just_got_hit = 0.0

        self.player_respawn_timer = 0.0
        self.opponent_respawn_timer = 0.0

        self.weapon_spawn_available = 0.0

        self.action_movement = 0.0
        self.action_jump = 0.0
        self.action_dodge = 0.0
        self.action_attack = 0.0
        # Edge geometry for on_edge detection
        self._edge_x_radius = 0.04
        self._edge_y_tolerance = 0.03
        
        # ── Relational features (stored for reuse) ──────────────────
        self.rel_dx: float = 0.0
        self.rel_dy: float = 0.0
        self.rel_distance: float = 1.0
        self.weapon_dx: float = 0.0
        self.weapon_dy: float = 0.0

        # ── NEW: temporal / strategic features ──────────────────────
        self.player_airborne_frames: int = 0
        self.opponent_airborne_frames: int = 0
        self.player_hitstun_timer: float = 0.0
        self.opponent_hitstun_timer: float = 0.0
        # Last known movement direction (for facing detection)
        self._player_last_dx: float = 0.0
        self.player_dodge_cooldown_remaining: float = 0.0
        self.player_dodge_cooldown_max: float = DODGE_COOLDOWN_AIR
        self._player_ground_grace_frames: int = 0
        self._opponent_ground_grace_frames: int = 0
        self._opponent_last_dx: float = 0.0

        # Added strategic/combat features
        self.player_time_since_hit: float = 0.0
        self.opponent_time_since_hit: float = 0.0
        self.last_knockback_dx: float = 0.0
        self.last_knockback_dy: float = 0.0

        # recovery information
        self.dist_to_nearest_ledge: float = 0.0
        self.signed_dx_to_ledge: float = 0.0
        self.dy_to_ledge: float = 0.0

        # ── Pre-allocated observation buffer ────────────────────────
        self._obs_buffer: np.ndarray = np.zeros(StateSpec.dim(), dtype=np.float32)

    def _clamp_position(self, x: float, y: float) -> tuple[float, float]:
        return (
            clamp(float(x), self.min_xy, self.max_xy),
            clamp(float(y), self.min_xy, self.max_xy),
        )

    def _select_detection(self, detections: List[dict], names: List[str], state: Optional[FighterState] = None) -> Optional[dict]:
        candidates = [d for d in detections if d.get("class_name") in names]
        if not candidates:
            return None

        if state is not None and state.exists:
            def score(det: dict) -> tuple[float, float]:
                x, y = bbox_center(det)
                dist = euclidian((x, y), (state.x, state.y))
                return (dist, -float(det.get("confidence", 0.0)))

            return min(candidates, key=score)

        return max(candidates, key=lambda d: float(d.get("confidence", 0.0)))

    def _update_fighter(
        self,
        state: FighterState,
        detection: Optional[dict],
        dt: float = 1.0 / 20.0,
        max_vel: float = 3.0,
        max_missing: int = 10,
        y_offset: float = 0.0,
    ) -> None:
        if detection is not None:
            x, y = bbox_center(detection)
            y = y + float(y_offset)
            x, y = self._clamp_position(x, y)

            dt = max(1e-6, float(dt))

            state.last_x, state.last_y = state.x, state.y
            state.vx = clamp((x - state.x) / dt, -max_vel, max_vel)
            state.vy = clamp((y - state.y) / dt, -max_vel, max_vel)
            state.x, state.y = x, y
            state.exists = True
            state.missing_frames = 0
            state.confidence = min(1.0, state.confidence + 0.15)
        else:
            state.last_x, state.last_y = state.x, state.y
            dt = max(1e-6, float(dt))
            state.x += state.vx * dt
            state.y += state.vy * dt
            state.x, state.y = self._clamp_position(state.x, state.y)
            state.vx *= 0.9
            state.vy *= 0.9

            state.missing_frames += 1
            state.confidence *= 0.95
            if state.missing_frames > max_missing:
                state.exists = False

    def update_from_detections(self, detections: List[dict], dt: float = 1.0 / 20.0) -> None:
        player_det = self._select_detection(detections, ["agent"], self.player)
        opponent_det = self._select_detection(detections, ["op", "op1", "op2"], self.opponent)
        weapon_candidates = [d for d in detections if d.get("class_name") == "weapons"]

        # Agent marker is above the body center; shift down to center coordinates.
        self._update_fighter(self.player, player_det, dt=dt, y_offset=PLAYER_MARKER_Y_OFFSET)
        self._update_fighter(self.opponent, opponent_det, dt=dt)

        if opponent_det is not None:
            op_name = str(opponent_det.get("class_name", "op"))
            if op_name == "op1":
                self.opponent.weapon_state = 1.0
            elif op_name == "op2":
                self.opponent.weapon_state = 2.0
            else:
                self.opponent.weapon_state = 0.0

        if weapon_candidates:
            closest = min(
                weapon_candidates,
                key=lambda d: euclidian(
                    bbox_center(d),
                    (self.player.x, self.player.y),
                ),
            )
            wx, wy = bbox_center(closest)
            wx, wy = self._clamp_position(wx, wy)
            self.weapon.x = wx
            self.weapon.y = wy
            self.weapon.exists = True
            self._weapon_missing_frames = 0
        else:
            # Keep last-known weapon location for a few frames to absorb one-frame YOLO misses.
            self._weapon_missing_frames += 1
            if self._weapon_missing_frames > MAX_WEAPON_MISSING_FRAMES:
                self.weapon.exists = False

        self.weapon_spawn_available = 1.0 if self.weapon.exists else 0.0

        # Track player facing direction from velocity
        if abs(self.player.vx) > 0.005:
            self._player_last_dx = 1.0 if self.player.vx > 0 else -1.0
        if abs(self.opponent.vx) > 0.005:
            self._opponent_last_dx = 1.0 if self.opponent.vx > 0 else -1.0

    def update_player_weapon_from_action(self, action_pick_throw: bool, dist_to_weapon: float, max_dist_threshold: float = 0.08) -> None:
        if not action_pick_throw:
            return
        if self.player.weapon_state > 0.0:
            self.player.weapon_state = 0.0
            return
        if dist_to_weapon <= max_dist_threshold and self.weapon.exists:
            self.player.weapon_state = 1.0
            # Weapon leaves the ground as soon as we pick it up.
            self.weapon.exists = False
            self._weapon_missing_frames = MAX_WEAPON_MISSING_FRAMES + 1

    def update_action(self, action) -> None:
        movement, jump, dodge, attack = action
        self.action_movement = float(movement)
        self.action_jump = float(jump)
        self.action_dodge = float(dodge)
        self.action_attack = float(attack)

        self.player.last_action_id = float(movement * 100 + jump * 10 + dodge * 5 + attack)

        if attack > 0:
            self.player.action_cooldown_remaining = 0.20
        if dodge > 0:
            self.player.action_cooldown_remaining = max(self.player.action_cooldown_remaining, 0.15)

    def update_on_ground(self, vy_threshold: float = 0.01) -> None:
        # Determine grounded state (close to platform y and within platform x bounds).
        # Use position tolerance + velocity gating + short grace to avoid one-frame flicker.
        # `player.y` / `opponent.y` are stored as body-center coordinates.
        player_foot_y = self.player.y + (HEIGHT / 2.0)
        opp_foot_y = self.opponent.y + (HEIGHT / 2.0)

        player_in_x = PLATFORM_X_MIN <= self.player.x <= PLATFORM_X_MAX
        opp_in_x = PLATFORM_X_MIN <= self.opponent.x <= PLATFORM_X_MAX

        player_close_ground = abs(player_foot_y - PLATFORM_Y_MIN) <= GROUND_Y_TOLERANCE
        opp_close_ground = abs(opp_foot_y - PLATFORM_Y_MIN) <= GROUND_Y_TOLERANCE

        player_raw_grounded = player_in_x and player_close_ground
        opp_raw_grounded = opp_in_x and opp_close_ground

        if player_raw_grounded:
            self._player_ground_grace_frames = 2
        else:
            self._player_ground_grace_frames = max(0, self._player_ground_grace_frames - 1)

        if opp_raw_grounded:
            self._opponent_ground_grace_frames = 2
        else:
            self._opponent_ground_grace_frames = max(0, self._opponent_ground_grace_frames - 1)

        self.player.grounded = bool(player_raw_grounded or self._player_ground_grace_frames > 0)
        self.opponent.grounded = bool(opp_raw_grounded or self._opponent_ground_grace_frames > 0)

        # Edge detection: near left or right ledge and near platform top
        def detect_on_edge(x: float, y: float) -> bool:
            near_left = abs(x - PLATFORM_X_MIN) < self._edge_x_radius and PLATFORM_Y_MIN <= y <= (PLATFORM_Y_MAX + self._edge_y_tolerance)
            near_right = abs(x - PLATFORM_X_MAX) < self._edge_x_radius and PLATFORM_Y_MIN <= y <= (PLATFORM_Y_MAX + self._edge_y_tolerance)
            return bool(near_left or near_right)

        self.player.on_edge = detect_on_edge(self.player.x, self.player.y + (HEIGHT / 2.0))
        self.opponent.on_edge = detect_on_edge(self.opponent.x, self.opponent.y + (HEIGHT / 2.0))

        # Reset jumps when landing on stage or touching the edge.
        if self.player.grounded or self.player.on_edge:
            self.player.jumps_left = 3
        else:
            self.player.jumps_left = min(self.player.jumps_left, 2)

        if self.opponent.grounded or self.opponent.on_edge:
            self.opponent.jumps_left = 3
        else:
            self.opponent.jumps_left = min(self.opponent.jumps_left, 2)

        # ── NEW: airborne frame tracking ─────────────────────────
        if self.player.grounded or self.player.on_edge:
            self.player_airborne_frames = 0
        else:
            self.player_airborne_frames += 1

        if self.opponent.grounded or self.opponent.on_edge:
            self.opponent_airborne_frames = 0
        else:
            self.opponent_airborne_frames += 1
            
        # ── Off-stage detection ──────────────────────────────────
        self.player.off_stage = self.update_player_off_stage()
        self.opponent.off_stage = self.update_off_stage(self.opponent)

        # Ground touch reduces dodge cooldown max from 3.2s to 1.0s.
        if self.player.grounded or self.player.on_edge:
            self.player_dodge_cooldown_max = DODGE_COOLDOWN_GROUNDED
            if self.player_dodge_cooldown_remaining > DODGE_COOLDOWN_GROUNDED:
                self.player_dodge_cooldown_remaining = DODGE_COOLDOWN_GROUNDED
        else:
            self.player_dodge_cooldown_max = DODGE_COOLDOWN_AIR

    def update_hitstun(self, dt: float) -> None:
        """Decay hitstun timers. Called from env._update_game_logic()."""
        self.player_time_since_hit += dt
        self.opponent_time_since_hit += dt

        if self.player_hitstun_timer > 0.0:
            self.player_hitstun_timer = max(0.0, self.player_hitstun_timer - dt)
        if self.opponent_hitstun_timer > 0.0:
            self.opponent_hitstun_timer = max(0.0, self.opponent_hitstun_timer - dt)

        # Trigger hitstun from damage events
        if self.just_got_hit > 0.5:
            # Hitstun proportional to damage dealt (Brawlhalla mechanic)
            hitstun_duration = 0.15 + 0.25 * self.player.damage_percent
            self.player_hitstun_timer = min(hitstun_duration, HITSTUN_MAX_FRAMES / 60.0)
            self.player_time_since_hit = 0.0
            self.last_knockback_dx = clamp(self.player.vx, -1.0, 1.0)
            self.last_knockback_dy = clamp(self.player.vy, -1.0, 1.0)
        if self.just_hit_opponent > 0.5:
            hitstun_duration = 0.15 + 0.25 * self.opponent.damage_percent
            self.opponent_hitstun_timer = min(hitstun_duration, HITSTUN_MAX_FRAMES / 60.0)
            self.opponent_time_since_hit = 0.0
            self.last_knockback_dx = clamp(self.opponent.vx, -1.0, 1.0)
            self.last_knockback_dy = clamp(self.opponent.vy, -1.0, 1.0)

    def update_dodge_cooldowns(self, dt: float, action_dodge: bool, opponent_dodge_detected: bool = False) -> None:
        current_player_max = DODGE_COOLDOWN_GROUNDED if (self.player.grounded or self.player.on_edge) else DODGE_COOLDOWN_AIR

        if action_dodge and self.player.dodge_available:
            self.player.dodge_available = False
            self.player_dodge_cooldown_max = current_player_max
            self.player_dodge_cooldown_remaining = current_player_max

        if opponent_dodge_detected and self.opponent.dodge_available:
            self.opponent.dodge_available = False
            self.opponent.action_cooldown_remaining = DODGE_COOLDOWN_AIR

        if not self.player.dodge_available:
            self.player_dodge_cooldown_max = current_player_max
            self.player_dodge_cooldown_remaining = min(self.player_dodge_cooldown_remaining, self.player_dodge_cooldown_max)
            self.player_dodge_cooldown_remaining = max(0.0, self.player_dodge_cooldown_remaining - dt)
            if self.player_dodge_cooldown_remaining <= 0.0:
                self.player.dodge_available = True
                self.player_dodge_cooldown_remaining = 0.0
        else:
            self.player_dodge_cooldown_max = current_player_max
            self.player_dodge_cooldown_remaining = 0.0

        if self.opponent.action_cooldown_remaining > 0.0:
            self.opponent.action_cooldown_remaining = max(0.0, self.opponent.action_cooldown_remaining - dt)
        else:
            self.opponent.dodge_available = True

    def update_jumps(self, action_jump: bool) -> None:
        if action_jump:
            self.player.action_cooldown_remaining = max(self.player.action_cooldown_remaining, 0.1)
            # Consume one jump if available
            if getattr(self.player, "jumps_left", 0) > 0:
                self.player.jumps_left = max(0, int(self.player.jumps_left - 1))

    def update_stocks_and_health(
        self,
        self_stocks_left: Optional[float] = None,
        self_health: Optional[float] = None,
        op_stocks_left: Optional[float] = None,
        op_health: Optional[float] = None,
    ) -> None:
        prev_self_health = self.self_health
        prev_op_health = self.op_health

        self.prev_self_stocks_left = self.self_stocks_left
        self.prev_op_stocks_left = self.op_stocks_left

        if self_stocks_left is not None:
            self.self_stocks_left = clamp(float(self_stocks_left), 0.0, self.max_stocks)
        if op_stocks_left is not None:
            self.op_stocks_left = clamp(float(op_stocks_left), 0.0, self.max_stocks)
        if self_health is not None:
            self.self_health = clamp(float(self_health), 0.0, self.max_health)
        if op_health is not None:
            self.op_health = clamp(float(op_health), 0.0, self.max_health)

        self.self_delta_damage = max(0.0, prev_self_health - self.self_health)
        self.op_delta_damage = max(0.0, prev_op_health - self.op_health)

        self.player.damage_percent = clamp((self.max_health - self.self_health) / self.max_health, 0.0, 1.0)
        self.opponent.damage_percent = clamp((self.max_health - self.op_health) / self.max_health, 0.0, 1.0)

        if self.self_stocks_left < self.prev_self_stocks_left:
            self.self_total_damage_taken_before_stock_loss = 0.0
            self.player_respawn_timer = RESPAWN_DURATION_SECONDS
            self.player.exists = False
        else:
            self.self_total_damage_taken_before_stock_loss += self.self_delta_damage

        if self.op_stocks_left < self.prev_op_stocks_left:
            self.op_total_damage_done_before_stock_loss = 0.0
            self.opponent_respawn_timer = RESPAWN_DURATION_SECONDS
            self.opponent.exists = False
            self.opponent.weapon_state = 0.0
        else:
            self.op_total_damage_done_before_stock_loss += self.op_delta_damage

        self.just_hit_opponent = 1.0 if self.op_delta_damage > 0.0 else 0.0
        self.just_got_hit = 1.0 if self.self_delta_damage > 0.0 else 0.0

        self.prev_self_health = self.self_health
        self.prev_op_health = self.op_health

    def update_existence_from_stocks(self, dt: float) -> None:
        if self.player_respawn_timer > 0.0:
            self.player_respawn_timer = max(0.0, self.player_respawn_timer - dt)
            self.player.exists = False
        elif self.self_stocks_left > 0.0:
            self.player.exists = True

        if self.opponent_respawn_timer > 0.0:
            self.opponent_respawn_timer = max(0.0, self.opponent_respawn_timer - dt)
            self.opponent.exists = False
        elif self.op_stocks_left > 0.0:
            self.opponent.exists = True

    def update_off_stage(self, state: FighterState):
        return (state.x < PLATFORM_X_MIN or state.x > PLATFORM_X_MAX) and state.y < PLATFORM_Y_MIN

    def update_player_off_stage(self):
        return self.update_off_stage(self.player)
    
    def _env_features(self) -> tuple[float, ...]:
        """Relational + game-state features for the observation vector."""
        both = self.player.exists and self.opponent.exists

        # ── NEW: weapon spatial features ──
        self.weapon_dx = float(self.weapon.x - self.player.x) if self.weapon.exists else 0.0
        self.weapon_dy = float(self.weapon.y - self.player.y) if self.weapon.exists else 0.0

        # ── signed relative position (critical for attack direction) ──
        self.rel_dx = float(self.opponent.x - self.player.x) if both else 0.0
        self.rel_dy = float(self.opponent.y - self.player.y) if both else 0.0
        self.rel_distance = euclidian(
            (self.player.x, self.player.y),
            (self.opponent.x, self.opponent.y),
        ) if both else 1.0

        # ── range booleans ──
        STRIKE_RANGE = 0.18
        in_strike_range = 1.0 if (both and self.rel_distance < STRIKE_RANGE) else 0.0

        # ── NEW: facing opponent ──
        # +1 if player is facing toward opponent, −1 if facing away, 0 if unknown
        facing = 0.0
        if both and abs(self._player_last_dx) > 0.1:
            facing = 1.0 if (self.rel_dx * self._player_last_dx > 0) else -1.0

        # +1 right, -1 left, 0 unknown/idle
        player_facing_dir = 0.0
        if abs(self._player_last_dx) > 0.1:
            player_facing_dir = 1.0 if self._player_last_dx > 0.0 else -1.0

        opponent_facing_dir = 0.0
        if abs(self._opponent_last_dx) > 0.1:
            opponent_facing_dir = 1.0 if self._opponent_last_dx > 0.0 else -1.0

        dodge_cooldown_norm = clamp(
            self.player_dodge_cooldown_remaining / max(1e-6, self.player_dodge_cooldown_max),
            0.0,
            1.0,
        )

        # Recovery information
        nearest_ledge = _nearest_ledge(self.player.x, PLATFORM_Y_MIN, PLATFORM_X_MIN, PLATFORM_X_MAX)
        self.dist_to_nearest_ledge = euclidian((self.player.x, self.player.y), nearest_ledge)
        self.signed_dx_to_ledge = nearest_ledge[0] - self.player.x  # Positive = need to move right, Negative = need to move left
        self.dy_to_ledge = self.player.y + HEIGHT / 2 - nearest_ledge[1]

        dist_to_stage_center = euclidian((self.player.x, self.player.y), (MID_X, PLATFORM_Y_MIN))
        signed_dx_to_stage_center = MID_X - self.player.x
        ledge_is_occupied = 1.0 if (self.player.on_edge or self.opponent.on_edge) else 0.0

        opponent_dodge_cooldown_norm = clamp(
            self.opponent.action_cooldown_remaining / max(1e-6, DODGE_COOLDOWN_AIR),
            0.0,
            1.0,
        )

        rel_vx = clamp(self.opponent.vx - self.player.vx, -3.0, 3.0)
        rel_vy = clamp(self.opponent.vy - self.player.vy, -3.0, 3.0)

        player_hitstun_norm = clamp(self.player_hitstun_timer / (HITSTUN_MAX_FRAMES / 60.0), 0.0, 1.0)
        opponent_hitstun_norm = clamp(self.opponent_hitstun_timer / (HITSTUN_MAX_FRAMES / 60.0), 0.0, 1.0)
        frame_advantage_estimate = clamp(opponent_hitstun_norm - player_hitstun_norm, -1.0, 1.0)

        return (
            # relational (8)
            clamp(self.rel_dx, -1.0, 1.0),
            clamp(self.rel_dy, -1.0, 1.0),
            clamp(self.rel_distance, 0.0, 2.0),
            in_strike_range,
            facing,
            player_facing_dir,
            clamp(self.weapon_dx, -1.0, 1.0),
            clamp(self.weapon_dy, -1.0, 1.0),
            # game state (4)
            clamp(self.self_stocks_left / self.max_stocks, 0.0, 1.0),
            clamp(self.op_stocks_left / self.max_stocks, 0.0, 1.0),
            dodge_cooldown_norm,
            1.0 if self.weapon.exists else 0.0,
            # recovery information (3)
            self.dist_to_nearest_ledge,
            self.signed_dx_to_ledge,
            self.dy_to_ledge,
            # added strategic/combat features (12)
            opponent_facing_dir,
            clamp(self.player_time_since_hit / 2.0, 0.0, 1.0),
            clamp(self.opponent_time_since_hit / 2.0, 0.0, 1.0),
            clamp(self.last_knockback_dx, -1.0, 1.0),
            clamp(self.last_knockback_dy, -1.0, 1.0),
            clamp(dist_to_stage_center, 0.0, 2.0),
            clamp(signed_dx_to_stage_center, -1.0, 1.0),
            ledge_is_occupied,
            opponent_dodge_cooldown_norm,
            clamp(rel_vx, -1.0, 1.0),
            clamp(rel_vy, -1.0, 1.0),
            frame_advantage_estimate,
        )

    def to_vector(self) -> np.ndarray:
        """51-dim observation with added strategic/combat awareness features.

        Uses pre-allocated buffer — no per-step allocation.
        Layout is defined by StateSpec.FEATURES.
        """
        buf = self._obs_buffer
        p = self.player
        o = self.opponent
        env = self._env_features()

        # ── player (10) ──
        buf[0] = p.x
        buf[1] = p.y
        buf[2] = p.vx
        buf[3] = p.vy
        buf[4] = 1.0 if p.grounded else 0.0
        buf[5] = clamp(p.damage_percent, 0.0, 1.0)
        buf[6] = 1.0 if p.weapon_state > 0.0 else 0.0
        buf[7] = clamp(float(p.jumps_left) / 3.0, 0.0, 1.0)
        buf[8] = 1.0 if p.on_edge else 0.0
        buf[9] = 1.0 if p.off_stage else 0.0

        # ── opponent (10) — now includes absolute position ─z─
        buf[10] = o.x
        buf[11] = o.y
        buf[12] = o.vx
        buf[13] = o.vy
        buf[14] = 1.0 if o.grounded else 0.0
        buf[15] = clamp(o.damage_percent, 0.0, 1.0)
        buf[16] = 1.0 if o.exists else 0.0
        buf[17] = clamp(float(o.jumps_left) / 3.0, 0.0, 1.0)
        buf[18] = 1.0 if o.on_edge else 0.0
        buf[19] = 1.0 if o.off_stage else 0.0

        # ── relational (8) + game (4) + recovery (3) = 15 ──
        buf[20] = env[0]   # rel_dx
        buf[21] = env[1]   # rel_dy
        buf[22] = env[2]   # rel_distance
        buf[23] = env[3]   # in_strike_range
        buf[24] = env[4]   # facing_opponent
        buf[25] = env[5]   # player_facing_dir
        buf[26] = env[6]   # weapon_dx
        buf[27] = env[7]   # weapon_dy
        buf[28] = env[8]   # self_stocks_norm
        buf[29] = env[9]   # op_stocks_norm
        buf[30] = env[10]  # dodge_cooldown_norm
        buf[31] = env[11]  # weapon_on_ground
        buf[32] = env[12]  # dist_to_nearest_ledge
        buf[33] = env[13]  # signed_dx_to_ledge
        buf[34] = env[14]  # dy_to_ledge

        # ── temporal / strategic (4) ──
        buf[35] = clamp(float(self.player_airborne_frames) / AIRBORNE_MAX_FRAMES, 0.0, 1.0)
        buf[36] = clamp(float(self.opponent_airborne_frames) / AIRBORNE_MAX_FRAMES, 0.0, 1.0)
        buf[37] = clamp(self.player_hitstun_timer / (HITSTUN_MAX_FRAMES / 60.0), 0.0, 1.0)
        buf[38] = clamp(self.opponent_hitstun_timer / (HITSTUN_MAX_FRAMES / 60.0), 0.0, 1.0)

        # ── added strategic/combat features (12) ──
        buf[39] = env[15]  # opponent_facing_dir
        buf[40] = env[16]  # player_time_since_hit
        buf[41] = env[17]  # opponent_time_since_hit
        buf[42] = env[18]  # last_knockback_dx
        buf[43] = env[19]  # last_knockback_dy
        buf[44] = env[20]  # dist_to_stage_center
        buf[45] = env[21]  # signed_dx_to_stage_center
        buf[46] = env[22]  # ledge_is_occupied
        buf[47] = env[23]  # opponent_dodge_cooldown_norm
        buf[48] = env[24]  # rel_vx
        buf[49] = env[25]  # rel_vy
        buf[50] = env[26]  # frame_advantage_estimate

        return buf
