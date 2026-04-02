from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from feature_extractor.memory.state_spec import StateSpec
from feature_extractor.memory.utils import _nearest_ledge, bbox_center, clamp, euclidian


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
    confidence: float = 0.0
    missing_frames: int = 0
    jumps_left: int = 3
    on_edge: bool = False
    off_stage: bool = False
    health: float = 351.0
    stocks: float = 3.0
    hitstun: bool = False
    height: float = 0.0722
    airborne_frames: float = 0.0
    dodge_cooldown: float = 0.0
    hitstun_duration: float = 0.0
    got_hit: bool = False


@dataclass
class WeaponState:
    exists: bool = False
    x: float = 0.5
    y: float = 0.5
    missing_frames: float = 0


@dataclass
class Platform:
    x_min: float = 0.315
    x_max: float = 0.683
    y_min: float = 0.5527
    y_max: float = 0.8149


@dataclass
class Physics:
    hitstun_max_frames: float = 30
    airborne_max_frames: float = 60
    dodge_air_cooldown: float = 3.2
    dodge_ground_cooldown: float = 1.0
    respawn_duration: float = 4.7
    ground_y_tolerance: float = 0.028
    max_weapon_missing_frames: float = 4
    max_dist_threshold: float = 0.1


class Memory:
    def __init__(self):
        self.max_health = 351.0
        self.max_stocks = 3.0
        self.min_xy = 0.0
        self.max_xy = 1.0

        self.player = FighterState()
        self.opponent = FighterState(exists=False)
        self.weapon = WeaponState()
        self.platform = Platform()
        self.physics = Physics()

        self._obs_buffer = np.zeros((StateSpec.dim(),), dtype=np.float32)

        self._edge_x_radius = 0.04
        self._edge_y_tolerance = 0.03
        self._player_last_dx = 1.0
        self._opponent_last_dx = -1.0

        self.rel_dx: float = 0.0
        self.rel_dy: float = 0.0
        self.rel_distance: float = 1.0
        self.weapon_dx: float = 0.0
        self.weapon_dy: float = 0.0

        self.player_time_since_hit: float = 2.0
        self.opponent_time_since_hit: float = 2.0
        self.last_knockback_dx: float = 0.0
        self.last_knockback_dy: float = 0.0

        self.player_dodge_cooldown_remaining: float = 0.0
        self.player_dodge_cooldown_max: float = self.physics.dodge_air_cooldown
        self.opponent_dodge_cooldown_remaining: float = 0.0

        self.self_stocks_left: float = self.max_stocks
        self.op_stocks_left: float = self.max_stocks
        self.prev_self_stocks_left: float = self.max_stocks
        self.prev_op_stocks_left: float = self.max_stocks

        self.self_health: float = self.max_health
        self.op_health: float = self.max_health
        self.prev_self_health: float = self.max_health
        self.prev_op_health: float = self.max_health

        self.self_delta_damage: float = 0.0
        self.op_delta_damage: float = 0.0
        self.just_hit_opponent: float = 0.0
        self.just_got_hit: float = 0.0

        self.self_total_damage_taken_before_stock_loss: float = 0.0
        self.op_total_damage_done_before_stock_loss: float = 0.0

        self.player_respawn_timer: float = 0.0
        self.opponent_respawn_timer: float = 0.0

    def _clamp_position(self, x: float, y: float) -> Tuple[float, float]:
        return (
            clamp(x, self.min_xy, self.max_xy),
            clamp(y, self.min_xy, self.max_xy),
        )

    def _select_detection(self, detections: List[dict], names: List[str], state: Optional[FighterState] = None) -> Optional[dict]:
        candidates = [d for d in detections if d.get("class_name") in names]
        if not candidates:
            return None

        if state is not None and state.exists:
            def score(det: dict) -> Tuple[float, float]:
                x, y = bbox_center(det)
                dist = euclidian((x, y), (state.x, state.y))
                return (dist, -float(det.get("confidence", 0.0)))

            return min(candidates, key=score)

        return max(candidates, key=lambda d: float(d.get("confidence", 0.0)))

    def _update_fighter(
        self,
        state: FighterState,
        detection: Optional[dict],
        dt: float = 1.0 / 41.0,
        max_vel: float = 3.0,
        max_missing: int = 10,
        y_offset: float = 0.0,
    ) -> None:
        dt = max(1e-6, float(dt))

        if detection is not None:
            x, y = bbox_center(detection)
            y += float(y_offset)
            x, y = self._clamp_position(x, y)

            state.last_x, state.last_y = state.x, state.y
            state.vx = clamp((x - state.x) / dt, -max_vel, max_vel)
            state.vy = clamp((y - state.y) / dt, -max_vel, max_vel)
            state.x, state.y = x, y
            state.exists = True
            state.missing_frames = 0
            state.confidence = min(1.0, state.confidence + 0.15)
            return

        state.last_x, state.last_y = state.x, state.y
        state.x += state.vx * dt
        state.y += state.vy * dt
        state.x, state.y = self._clamp_position(state.x, state.y)
        state.vx *= 0.995
        state.vy *= 0.995

        state.missing_frames += 1
        state.confidence *= 0.95
        if state.missing_frames > max_missing:
            state.exists = False

    def update_from_detections(self, detections: List[dict], dt: float = 1.0 / 41.0) -> None:
        player_det = self._select_detection(detections, ["agent"], self.player)
        opponent_det = self._select_detection(detections, ["op", "op1", "op2"], self.opponent)
        weapon_candidates = [d for d in detections if d.get("class_name") == "weapons"]

        self._update_fighter(self.player, player_det, dt=dt, y_offset=self.player.height)
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
                key=lambda d: euclidian(bbox_center(d), (self.player.x, self.player.y)),
            )
            wx, wy = bbox_center(closest)
            wx, wy = self._clamp_position(wx, wy)
            self.weapon.x, self.weapon.y = wx, wy
            self.weapon.exists = True
            self.weapon.missing_frames = 0
        else:
            self.weapon.missing_frames += 1
            if self.weapon.missing_frames >= self.physics.max_weapon_missing_frames:
                self.weapon.exists = False

        if abs(self.player.vx) > 0.005:
            self._player_last_dx = 1.0 if self.player.vx > 0 else -1.0
        if abs(self.opponent.vx) > 0.005:
            self._opponent_last_dx = 1.0 if self.opponent.vx > 0 else -1.0

        self.update_on_ground()

    def update_player_weapon_from_action(self, action_pick_throw: bool, dist_to_weapon: float) -> None:
        if not action_pick_throw:
            return

        if self.player.weapon_state > 0.0:
            self.player.weapon_state = 0.0
            return

        if dist_to_weapon <= self.physics.max_dist_threshold and self.weapon.exists:
            self.player.weapon_state = 1.0
            self.weapon.exists = False
            self.weapon.missing_frames = self.physics.max_weapon_missing_frames + 1

    def update_on_ground(self, vy_threshold: float | None = None) -> None:
        _ = vy_threshold
        player_foot_y = self.player.y + (self.player.height / 2.0)
        opponent_foot_y = self.opponent.y + (self.opponent.height / 2.0)

        player_in_x = self.platform.x_min <= self.player.x <= self.platform.x_max
        opp_in_x = self.platform.x_min <= self.opponent.x <= self.platform.x_max

        player_close_ground = abs(player_foot_y - self.platform.y_min) <= self.physics.ground_y_tolerance
        opponent_close_ground = abs(opponent_foot_y - self.platform.y_min) <= self.physics.ground_y_tolerance

        player_grounded = player_in_x and player_close_ground
        opponent_grounded = opp_in_x and opponent_close_ground

        self.player.grounded = bool(player_grounded)
        self.opponent.grounded = bool(opponent_grounded)

        def detect_on_edge(x: float, foot_y: float) -> bool:
            near_left = (
                abs(x - self.platform.x_min) < self._edge_x_radius
                and self.platform.y_min - self._edge_y_tolerance <= foot_y <= self.platform.y_max + self._edge_y_tolerance
            )
            near_right = (
                abs(x - self.platform.x_max) < self._edge_x_radius
                and self.platform.y_min - self._edge_y_tolerance <= foot_y <= self.platform.y_max + self._edge_y_tolerance
            )
            return bool(near_left or near_right)

        self.player.on_edge = detect_on_edge(self.player.x, player_foot_y)
        self.opponent.on_edge = detect_on_edge(self.opponent.x, opponent_foot_y)

        if self.player.grounded:
            self.player.jumps_left = 3
        elif self.player.on_edge:
            self.player.jumps_left = 2

        if self.opponent.grounded:
            self.opponent.jumps_left = 3
        elif self.opponent.on_edge:
            self.opponent.jumps_left = 2

        if self.player.grounded or self.player.on_edge:
            self.player.airborne_frames = 0
        else:
            self.player.airborne_frames += 1

        if self.opponent.grounded or self.opponent.on_edge:
            self.opponent.airborne_frames = 0
        else:
            self.opponent.airborne_frames += 1

        self.player.off_stage = self.update_player_off_stage()
        self.opponent.off_stage = self.update_off_stage(self.opponent)

        if self.player.grounded or self.player.on_edge:
            self.player_dodge_cooldown_max = self.physics.dodge_ground_cooldown
            if self.player_dodge_cooldown_remaining > self.player_dodge_cooldown_max:
                self.player_dodge_cooldown_remaining = self.player_dodge_cooldown_max
        else:
            self.player_dodge_cooldown_max = self.physics.dodge_air_cooldown

    def update_hitstun(self, dt: float) -> None:
        dt = max(1e-6, float(dt))

        self.player.hitstun_duration = max(0.0, self.player.hitstun_duration - dt)
        self.opponent.hitstun_duration = max(0.0, self.opponent.hitstun_duration - dt)

        if self.player.got_hit:
            self.player.hitstun_duration = 0.15 + 0.25 * self.player.damage_percent
            self.player.got_hit = False
        if self.opponent.got_hit:
            self.opponent.hitstun_duration = 0.15 + 0.25 * self.opponent.damage_percent
            self.opponent.got_hit = False

        self.player.hitstun = self.player.hitstun_duration > 0.0
        self.opponent.hitstun = self.opponent.hitstun_duration > 0.0

    def update_dodge_cooldown(self, dt: float, action_dodge: bool) -> None:
        dt = max(1e-6, float(dt))
        current_player_max = (
            self.physics.dodge_ground_cooldown if (self.player.grounded or self.player.on_edge) else self.physics.dodge_air_cooldown
        )

        if action_dodge and self.player.dodge_available:
            self.player.dodge_available = False
            self.player_dodge_cooldown_max = current_player_max
            self.player_dodge_cooldown_remaining = current_player_max

        if not self.player.dodge_available:
            self.player_dodge_cooldown_max = current_player_max
            self.player_dodge_cooldown_remaining = min(
                self.player_dodge_cooldown_remaining,
                self.player_dodge_cooldown_max,
            )
            self.player_dodge_cooldown_remaining = max(0.0, self.player_dodge_cooldown_remaining - dt)
            if self.player_dodge_cooldown_remaining <= 0.0:
                self.player.dodge_available = True
                self.player_dodge_cooldown_remaining = 0.0
        else:
            self.player_dodge_cooldown_max = current_player_max
            self.player_dodge_cooldown_remaining = 0.0

    def update_dodge_cooldowns(self, dt: float, action_dodge: bool, opponent_dodge_detected: bool = False) -> None:
        dt = max(1e-6, float(dt))
        self.update_dodge_cooldown(dt=dt, action_dodge=action_dodge)

        if opponent_dodge_detected and self.opponent.dodge_available:
            self.opponent.dodge_available = False
            self.opponent_dodge_cooldown_remaining = self.physics.dodge_air_cooldown

        if not self.opponent.dodge_available:
            self.opponent_dodge_cooldown_remaining = max(0.0, self.opponent_dodge_cooldown_remaining - dt)
            if self.opponent_dodge_cooldown_remaining <= 0.0:
                self.opponent_dodge_cooldown_remaining = 0.0
                self.opponent.dodge_available = True

    def update_action(self, action: np.ndarray | list[int] | tuple[int, ...]) -> None:
        action_arr = np.asarray(action, dtype=np.int64).reshape(-1)
        if action_arr.size == 0:
            return

        movement = int(action_arr[0])
        jump = int(action_arr[1]) if action_arr.size > 1 else 0
        dodge = int(action_arr[2]) if action_arr.size > 2 else 0
        attack = int(action_arr[3]) if action_arr.size > 3 else 0

        packed_action_id = movement + (4 * jump) + (8 * dodge) + (16 * attack)
        self.player.last_action_id = float(packed_action_id)

    def update_jumps(self, action_jump: bool) -> None:
        if action_jump and not (self.player.grounded or self.player.on_edge):
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

        self.just_hit_opponent = 1.0 if self.op_delta_damage > 0.0 else 0.0
        self.just_got_hit = 1.0 if self.self_delta_damage > 0.0 else 0.0

        if self.just_got_hit > 0.0:
            self.player_time_since_hit = 0.0
            self.last_knockback_dx = clamp(self.player.vx, -1.0, 1.0)
            self.last_knockback_dy = clamp(self.player.vy, -1.0, 1.0)
            self.player.got_hit = True
        else:
            self.player_time_since_hit = min(2.0, self.player_time_since_hit + 1.0 / 41.0)

        if self.just_hit_opponent > 0.0:
            self.opponent_time_since_hit = 0.0
            self.opponent.got_hit = True
        else:
            self.opponent_time_since_hit = min(2.0, self.opponent_time_since_hit + 1.0 / 41.0)

        if self.self_stocks_left < self.prev_self_stocks_left:
            self.self_total_damage_taken_before_stock_loss = 0.0
            self.player_respawn_timer = self.physics.respawn_duration
            self.player.exists = False
            self.player.weapon_state = 0.0
            self.player.jumps_left = 3
        else:
            self.self_total_damage_taken_before_stock_loss += self.self_delta_damage

        if self.op_stocks_left < self.prev_op_stocks_left:
            self.op_total_damage_done_before_stock_loss = 0.0
            self.opponent_respawn_timer = self.physics.respawn_duration
            self.opponent.exists = False
            self.opponent.weapon_state = 0.0
            self.opponent.jumps_left = 3
        else:
            self.op_total_damage_done_before_stock_loss += self.op_delta_damage

        self.player.health = self.self_health
        self.opponent.health = self.op_health
        self.player.stocks = self.self_stocks_left
        self.opponent.stocks = self.op_stocks_left

        self.prev_self_health = self.self_health
        self.prev_op_health = self.op_health

    def update_existence_from_stocks(self, dt: float) -> None:
        dt = max(1e-6, float(dt))

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

    def update_off_stage(self, state: FighterState) -> bool:
        return (state.x < self.platform.x_min or state.x > self.platform.x_max) and (state.y > self.platform.y_min)

    def update_player_off_stage(self) -> bool:
        return self.update_off_stage(self.player)

    def _env_features(self) -> Tuple[float, ...]:
        both = self.player.exists and self.opponent.exists

        self.weapon_dx = float(self.weapon.x - self.player.x) if self.weapon.exists else 0.0
        self.weapon_dy = float(self.weapon.y - self.player.y) if self.weapon.exists else 0.0

        self.rel_dx = float(self.opponent.x - self.player.x) if both else 0.0
        self.rel_dy = float(self.opponent.y - self.player.y) if both else 0.0
        self.rel_distance = (
            euclidian((self.player.x, self.player.y), (self.opponent.x, self.opponent.y)) if both else 1.0
        )

        strike_range = 0.18
        in_strike_range = 1.0 if (both and self.rel_distance < strike_range) else 0.0

        facing = 0.0
        if both and abs(self._player_last_dx) > 0.1:
            facing = 1.0 if (self.rel_dx * self._player_last_dx > 0) else -1.0

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

        nearest_ledge = _nearest_ledge(self.player.x, self.platform.y_min, self.platform.x_min, self.platform.x_max)
        dist_to_nearest_ledge = euclidian((self.player.x, self.player.y), nearest_ledge)
        signed_dx_to_ledge = nearest_ledge[0] - self.player.x
        dy_to_ledge = self.player.y - nearest_ledge[1]

        mid_x = (self.platform.x_min + self.platform.x_max) * 0.5
        dist_to_stage_center = euclidian((self.player.x, self.player.y), (mid_x, self.platform.y_min))
        signed_dx_to_stage_center = mid_x - self.player.x
        ledge_is_occupied = 1.0 if (self.player.on_edge or self.opponent.on_edge) else 0.0

        opponent_dodge_cooldown_norm = clamp(
            self.opponent_dodge_cooldown_remaining / max(1e-6, self.physics.dodge_air_cooldown),
            0.0,
            1.0,
        )

        rel_vx = clamp(self.opponent.vx - self.player.vx, -3.0, 3.0)
        rel_vy = clamp(self.opponent.vy - self.player.vy, -3.0, 3.0)

        player_hitstun_norm = clamp(self.player.hitstun_duration / 0.8, 0.0, 1.0)
        opponent_hitstun_norm = clamp(self.opponent.hitstun_duration / 0.8, 0.0, 1.0)
        frame_advantage_estimate = clamp(opponent_hitstun_norm - player_hitstun_norm, -1.0, 1.0)

        return (
            clamp(self.rel_dx, -1.0, 1.0),
            clamp(self.rel_dy, -1.0, 1.0),
            clamp(self.rel_distance, 0.0, 2.0),
            in_strike_range,
            facing,
            player_facing_dir,
            clamp(self.weapon_dx, -1.0, 1.0),
            clamp(self.weapon_dy, -1.0, 1.0),
            clamp(self.self_stocks_left / self.max_stocks, 0.0, 1.0),
            clamp(self.op_stocks_left / self.max_stocks, 0.0, 1.0),
            dodge_cooldown_norm,
            1.0 if self.weapon.exists else 0.0,
            clamp(dist_to_nearest_ledge, 0.0, 2.0),
            clamp(signed_dx_to_ledge, -1.0, 1.0),
            clamp(dy_to_ledge, -1.0, 1.0),
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
        buf = self._obs_buffer
        p = self.player
        o = self.opponent
        env = self._env_features()

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

        buf[20] = env[0]
        buf[21] = env[1]
        buf[22] = env[2]
        buf[23] = env[3]
        buf[24] = env[4]
        buf[25] = env[5]
        buf[26] = env[6]
        buf[27] = env[7]
        buf[28] = env[8]
        buf[29] = env[9]
        buf[30] = env[10]
        buf[31] = env[11]
        buf[32] = env[12]
        buf[33] = env[13]
        buf[34] = env[14]

        buf[35] = clamp(float(p.airborne_frames) / self.physics.airborne_max_frames, 0.0, 1.0)
        buf[36] = clamp(float(o.airborne_frames) / self.physics.airborne_max_frames, 0.0, 1.0)
        buf[37] = clamp(p.hitstun_duration / 0.8, 0.0, 1.0)
        buf[38] = clamp(o.hitstun_duration / 0.8, 0.0, 1.0)

        buf[39] = env[15]
        buf[40] = env[16]
        buf[41] = env[17]
        buf[42] = env[18]
        buf[43] = env[19]
        buf[44] = env[20]
        buf[45] = env[21]
        buf[46] = env[22]
        buf[47] = env[23]
        buf[48] = env[24]
        buf[49] = env[25]
        buf[50] = env[26]

        return buf
