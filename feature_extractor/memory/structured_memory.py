from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from feature_extractor.memory.detection_schema import (
    OPPONENT_WEAPON_STATE_FROM_CLASS,
    resolve as resolve_detections,
)
from action_space import ACTION_DIM, Action, component_vector, components
from feature_extractor.memory.state_spec import StateSpec
from feature_extractor.memory.utils import _nearest_ledge, bbox_center, bbox_size, clamp, euclidian


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
    # Detector bounding-box extent. The silhouette carries animation information
    # the centre point throws away: a swing widens the box, a dodge compresses it.
    bbox_w: float = 0.0
    bbox_h: float = 0.0
    prev_bbox_w: float = 0.0
    prev_bbox_h: float = 0.0


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
    pickup_distance_norm: float = 0.05
    # Movement physics (normalised-coordinate space)
    gravity: float = 3.0             # norm_coords/s²  (positive = downward in screen coords)
    ground_friction: float = 8.0     # 1/s  velocity damping on ground
    air_friction: float = 0.5        # 1/s  velocity damping in air


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
        # Enter outside the calibrated platform by more than detector jitter, then
        # require a small inward return before clearing. Without hysteresis, one
        # noisy box at the ledge fabricates alternating recovery/onstage states.
        self._offstage_enter_x_margin = 0.01
        self._offstage_exit_x_inset = 0.015
        self._player_last_dx = 1.0
        self._opponent_last_dx = -1.0

        self.rel_dx: float = 0.0
        self.rel_dy: float = 0.0
        self.rel_distance: float = 1.0
        self.weapon_dx: float = 0.0
        self.weapon_dy: float = 0.0
        self.weapon_visible_this_frame: bool = False
        self.visible_weapon_centers: list[Tuple[float, float]] = []
        self.closest_weapon_distance: float = float("inf")

        # Perception provenance for the current frame. `identity_observed` is False
        # when the self-indicator was not detected and agent identity was carried
        # forward from the previous frame, so consumers can distinguish a measured
        # identity from a guessed one.
        self.detection_schema: str = "none"
        self.identity_source: str = "none"
        self.identity_observed: bool = False
        self.indicator_match_score: float = float("inf")
        self.time_since_indicator: float = 2.0
        self.time_since_dodge_input: float = 2.0

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
        self.weapon_pickup_action_this_frame: bool = False
        self.weapon_drop_action_this_frame: bool = False

        # Decomposed previous action, in canonical space. All-zero means NOOP, which
        # is the correct "nothing happened yet" state -- unlike the old encoding where
        # a zeroed movement channel meant "holding left".
        self._prev_action_id: int = int(Action.NOOP)
        self._prev_action = component_vector(Action.NOOP)

    def _clamp_position(self, x: float, y: float) -> Tuple[float, float]:
        return (
            clamp(x, self.min_xy, self.max_xy),
            clamp(y, self.min_xy, self.max_xy),
        )

    def _last_xy(self, state: FighterState) -> Optional[Tuple[float, float]]:
        """Last known position, or None when the fighter is not currently tracked."""
        return (state.x, state.y) if state.exists else None

    def _closest_weapon_center(self, centers: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if not centers:
            self.closest_weapon_distance = float("inf")
            return None

        closest = min(
            centers,
            key=lambda xy: euclidian(xy, (self.player.x, self.player.y)),
        )
        self.closest_weapon_distance = float(euclidian(closest, (self.player.x, self.player.y)))
        return closest

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

        if detection is None:
            # No physics fallback: keep last location until detections return.
            state.last_x, state.last_y = state.x, state.y
            state.vx = 0.0
            state.vy = 0.0
            state.prev_bbox_w, state.prev_bbox_h = state.bbox_w, state.bbox_h
            state.missing_frames += 1
            state.confidence *= 0.92
            if state.missing_frames > max_missing:
                state.exists = False
            return

        x, y = bbox_center(detection)
        bbox_w, bbox_h = bbox_size(detection)

        # Shift the box centre down to the feet, using the MEASURED half-height.
        #
        # Both fighters must use the same convention. Previously the agent was
        # shifted by a full `height` at detection time while the opponent was
        # shifted by `height / 2` at ground-check time, so the agent's foot sat
        # roughly half a body too low: it read as airborne while standing on the
        # platform, and grounded movement goals became unreachable because
        # `player_y` was biased past the target by more than the success radius.
        #
        # Measuring per frame also tracks the sprite: crouches, aerial poses and
        # attack animations change the box, and a fixed constant cannot follow that.
        half_height = (bbox_h if bbox_h > 0.0 else state.height) / 2.0
        x_new, y_new = self._clamp_position(x, y + half_height + float(y_offset))
        state.last_x, state.last_y = state.x, state.y
        state.vx = clamp((x_new - state.x) / dt, -max_vel, max_vel)
        state.vy = clamp((y_new - state.y) / dt, -max_vel, max_vel)
        state.x, state.y = x_new, y_new

        state.prev_bbox_w, state.prev_bbox_h = state.bbox_w, state.bbox_h
        state.bbox_w, state.bbox_h = bbox_w, bbox_h

        state.exists = True
        state.missing_frames = 0

        det_conf = float(detection.get("confidence", 0.0))
        state.confidence = float(np.clip((0.5 * state.confidence) + (0.5 * det_conf), 0.0, 1.0))

    def update_from_detections(
        self,
        detections: List[dict],
        dt: float = 1.0 / 41.0,
    ) -> None:
        # YOLO is the sole detection source; tracker layer removed.
        yolo_detections = list(detections or [])

        # Identity is resolved by the self-indicator under the current schema, and
        # by class label under the legacy one. The schema is inferred per frame.
        resolution = resolve_detections(
            yolo_detections,
            last_agent_xy=self._last_xy(self.player),
            last_opponent_xy=self._last_xy(self.opponent),
        )
        self.detection_schema = resolution.schema
        self.identity_source = resolution.identity_source
        self.identity_observed = resolution.identity_is_observed
        self.indicator_match_score = resolution.indicator_score
        if resolution.identity_is_observed:
            self.time_since_indicator = 0.0
        else:
            self.time_since_indicator = min(2.0, self.time_since_indicator + dt)

        self._update_fighter(self.player, resolution.agent, dt=dt)
        self._update_fighter(self.opponent, resolution.opponent, dt=dt)

        opponent_det = resolution.opponent
        if opponent_det is not None and resolution.schema == "legacy":
            # Legacy only: opponent weapon state was encoded in the op1/op2 split.
            # Under the 3-class schema it must come from the crop classifier, so
            # the previous value is left untouched rather than asserted as zero.
            op_name = str(opponent_det.get("class_name", "op"))
            self.opponent.weapon_state = OPPONENT_WEAPON_STATE_FROM_CLASS.get(op_name, 0.0)

        visible_weapon_centers: list[Tuple[float, float]] = [
            self._clamp_position(*bbox_center(d)) for d in resolution.weapons
        ]
        self.visible_weapon_centers = visible_weapon_centers
        self.weapon_visible_this_frame = bool(visible_weapon_centers)

        closest_weapon = self._closest_weapon_center(visible_weapon_centers)
        if closest_weapon is not None:
            wx, wy = closest_weapon
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
        self.weapon_pickup_action_this_frame = False
        self.weapon_drop_action_this_frame = False

        if not action_pick_throw:
            return

        if self.player.weapon_state > 0.0:
            self.player.weapon_state = 0.0
            self.weapon_drop_action_this_frame = True
            return

        distance_to_weapon = float(max(0.0, dist_to_weapon))
        if self.weapon.exists and distance_to_weapon <= float(self.physics.pickup_distance_norm):
            self.player.weapon_state = 1.0
            self.weapon_pickup_action_this_frame = True
            self.weapon.exists = False
            self.weapon.missing_frames = self.physics.max_weapon_missing_frames + 1

    def update_on_ground(self, vy_threshold: float | None = None) -> None:
        _ = vy_threshold
        # Player y is already shifted by player.height at detection update time
        # (see update_from_detections -> _update_fighter with y_offset=self.player.height).
        # Both fighters store the FOOT position in .y (shifted from the box centre by
        # the measured half-height in _update_fighter), so neither is re-shifted here.
        # Applying a second offset to one and not the other is exactly what made the
        # agent read as airborne while standing on the platform.
        player_foot_y = self.player.y
        opponent_foot_y = self.opponent.y

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
        # Measured ingredient beside the simulated cooldown estimate: the policy
        # can use it to learn when the estimate has drifted.
        if action_dodge:
            self.time_since_dodge_input = 0.0
        else:
            self.time_since_dodge_input = min(2.0, self.time_since_dodge_input + dt)
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

    def update_action(self, action: int) -> None:
        """Record the executed action, decomposed, in canonical space."""
        action_id = int(np.clip(int(np.asarray(action).reshape(-1)[0]), 0, ACTION_DIM - 1))
        self._prev_action_id = action_id
        self._prev_action = component_vector(action_id)
        self.player.last_action_id = float(action_id)

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
        """Return whether a tracked airborne fighter has left the playable top.

        Lateral displacement and passing below the platform are independent ways
        to be offstage. Requiring both hid high recoveries in the live overlay: a
        fighter far beyond a ledge read as onstage until falling below its top.
        """
        if not state.exists or state.grounded:
            return False

        if state.off_stage:
            outside_x = bool(
                state.x < self.platform.x_min + self._offstage_exit_x_inset
                or state.x > self.platform.x_max - self._offstage_exit_x_inset
            )
            # Clear vertical offstage only after returning above half the ordinary
            # ground tolerance; this avoids flicker around the calibrated surface.
            below_platform = bool(
                state.y
                > self.platform.y_min + (0.5 * self.physics.ground_y_tolerance)
            )
        else:
            outside_x = bool(
                state.x < self.platform.x_min - self._offstage_enter_x_margin
                or state.x > self.platform.x_max + self._offstage_enter_x_margin
            )
            below_platform = bool(
                state.y
                > self.platform.y_min + self.physics.ground_y_tolerance
            )

        return bool(outside_x or below_platform)

    def update_player_off_stage(self) -> bool:
        return self.update_off_stage(self.player)

    def _refresh_relational(self) -> None:
        """Recompute relational quantities and cache them on the instance.

        Reward providers read ``rel_distance`` / ``weapon_dx`` directly, so these
        stay as attributes rather than becoming locals of ``to_vector``.
        """
        both = self.player.exists and self.opponent.exists

        self.weapon_dx = float(self.weapon.x - self.player.x) if self.weapon.exists else 0.0
        self.weapon_dy = float(self.weapon.y - self.player.y) if self.weapon.exists else 0.0

        self.rel_dx = float(self.opponent.x - self.player.x) if both else 0.0
        self.rel_dy = float(self.opponent.y - self.player.y) if both else 0.0
        self.rel_distance = (
            euclidian((self.player.x, self.player.y), (self.opponent.x, self.opponent.y)) if both else 1.0
        )

    def _in_strike_range(self) -> float:
        """Smooth distance band. Derived from measured geometry, not from frame data.

        User-calibrated windows: ~0.01 is a guaranteed connect, ~0.15 is the outer
        awareness band that decays to zero.
        """
        if not (self.player.exists and self.opponent.exists):
            return 0.0

        guaranteed, extended = 0.01, 0.15
        if self.rel_distance <= guaranteed:
            return 1.0
        if self.rel_distance <= extended:
            return float((extended - self.rel_distance) / max(1e-6, extended - guaranteed))
        return 0.0

    def _weapon_type(self, state: FighterState) -> float:
        """0 = unarmed, 1 / 2 = the legend's two weapons.

        Kept as a small ordinal rather than a one-hot on request. The values are
        nominal, so the ordering carries no meaning; if aliasing between the two
        weapons shows up in training, widen this to a one-hot before adding features.
        """
        return float(clamp(state.weapon_state, 0.0, 2.0))

    def to_vector(self) -> np.ndarray:
        """Assemble the single-frame observation in StateSpec order.

        Every entry is measured, derived from measurement plus fixed stage
        calibration, an estimate shipped alongside its measured ingredient, or the
        agent's own previous action. Nothing here is an unverifiable guess at hidden
        game state.
        """
        buf = self._obs_buffer
        p = self.player
        o = self.opponent

        self._refresh_relational()

        # ══ DYNAMIC BLOCK ═════════════════════════════════════════════════
        buf[0] = p.x
        buf[1] = p.y
        buf[2] = clamp(p.vx, -1.0, 1.0)
        buf[3] = clamp(p.vy, -1.0, 1.0)
        buf[4] = clamp(p.bbox_w, 0.0, 1.0)
        buf[5] = clamp(p.bbox_h, 0.0, 1.0)
        buf[6] = clamp(p.bbox_w - p.prev_bbox_w, -1.0, 1.0)
        buf[7] = clamp(p.bbox_h - p.prev_bbox_h, -1.0, 1.0)

        buf[8] = o.x
        buf[9] = o.y
        buf[10] = clamp(o.vx, -1.0, 1.0)
        buf[11] = clamp(o.vy, -1.0, 1.0)
        buf[12] = clamp(o.bbox_w, 0.0, 1.0)
        buf[13] = clamp(o.bbox_h, 0.0, 1.0)
        buf[14] = clamp(o.bbox_w - o.prev_bbox_w, -1.0, 1.0)
        buf[15] = clamp(o.bbox_h - o.prev_bbox_h, -1.0, 1.0)

        buf[16] = clamp(self.rel_dx, -1.0, 1.0)
        buf[17] = clamp(self.rel_dy, -1.0, 1.0)
        buf[18] = clamp(self.rel_distance, 0.0, 2.0)
        buf[19] = clamp(o.vx - p.vx, -1.0, 1.0)
        buf[20] = clamp(o.vy - p.vy, -1.0, 1.0)

        # Executed previous action, decomposed. Inside the dynamic block so every
        # history slice carries the action that produced the next state.
        offset = StateSpec.ACTION_OFFSET
        buf[offset : offset + self._prev_action.shape[0]] = self._prev_action

        # ══ STATIC CONTEXT ════════════════════════════════════════════════
        i = StateSpec.DYNAMIC_DIM

        nearest_ledge = _nearest_ledge(p.x, self.platform.y_min, self.platform.x_min, self.platform.x_max)
        mid_x = (self.platform.x_min + self.platform.x_max) * 0.5
        half_span = min(mid_x, 1.0 - mid_x)

        buf[i + 0] = clamp(nearest_ledge[0] - p.x, -1.0, 1.0)
        buf[i + 1] = clamp(p.y - nearest_ledge[1], -1.0, 1.0)
        buf[i + 2] = clamp(euclidian((p.x, p.y), nearest_ledge), 0.0, 2.0)
        buf[i + 3] = 1.0 if p.grounded else 0.0
        buf[i + 4] = 1.0 if p.on_edge else 0.0
        buf[i + 5] = 1.0 if p.off_stage else 0.0
        buf[i + 6] = clamp(mid_x - p.x, -1.0, 1.0)
        # Blast-zone margins measured about the STAGE centre, so they stay symmetric
        # under the canonicalisation mirror.
        buf[i + 7] = clamp(half_span - abs(p.x - mid_x), 0.0, 1.0)
        buf[i + 8] = clamp(1.0 - p.y, 0.0, 1.0)
        buf[i + 9] = 1.0 if o.grounded else 0.0
        buf[i + 10] = 1.0 if o.off_stage else 0.0

        buf[i + 11] = clamp(p.damage_percent, 0.0, 1.0)
        buf[i + 12] = clamp(o.damage_percent, 0.0, 1.0)
        buf[i + 13] = clamp(self.self_stocks_left / self.max_stocks, 0.0, 1.0)
        buf[i + 14] = clamp(self.op_stocks_left / self.max_stocks, 0.0, 1.0)
        buf[i + 15] = 1.0 if p.weapon_state > 0.0 else 0.0
        buf[i + 16] = self._weapon_type(p)
        buf[i + 17] = self._weapon_type(o)
        buf[i + 18] = clamp(self.weapon_dx, -1.0, 1.0)
        buf[i + 19] = clamp(self.weapon_dy, -1.0, 1.0)
        buf[i + 20] = 1.0 if self.weapon.exists else 0.0

        buf[i + 21] = clamp(float(p.jumps_left) / 3.0, 0.0, 1.0)
        buf[i + 22] = clamp(float(p.airborne_frames) / self.physics.airborne_max_frames, 0.0, 1.0)
        buf[i + 23] = clamp(
            self.player_dodge_cooldown_remaining / max(1e-6, self.player_dodge_cooldown_max), 0.0, 1.0
        )
        buf[i + 24] = clamp(self.time_since_dodge_input / 2.0, 0.0, 1.0)
        buf[i + 25] = clamp(float(o.airborne_frames) / self.physics.airborne_max_frames, 0.0, 1.0)

        buf[i + 26] = 1.0 if self.identity_observed else 0.0
        buf[i + 27] = clamp(self.time_since_indicator / 2.0, 0.0, 1.0)
        buf[i + 28] = clamp(float(p.missing_frames) / 10.0, 0.0, 1.0)
        buf[i + 29] = clamp(float(o.missing_frames) / 10.0, 0.0, 1.0)
        buf[i + 30] = clamp(p.confidence, 0.0, 1.0)
        buf[i + 31] = clamp(o.confidence, 0.0, 1.0)

        buf[i + 32] = self._in_strike_range()
        buf[i + 33] = 1.0 if o.exists else 0.0

        return buf
