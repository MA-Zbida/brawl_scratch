"""Gym-compatible environment powered by a trained world model.

Instead of interacting with the real game, ``WorldModelEnv`` uses the
learned dynamics model (``WorldModel``) to predict transitions.  This
allows training a policy at ~1000x real-time speed.

Usage (standalone smoke test)::

    python -m world_model.imagination_env --model world_model/data/world_model.pt \
           --data world_model/data/transitions.npz --steps 200
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from feature_extractor.memory.state_spec import StateSpec
from world_model.model import WorldModel, STATE_DIM, ACTION_DIMS

# ── platform constants (from Memory.Platform defaults) ───────────────
_PLAT_X_MIN = 0.315
_PLAT_X_MAX = 0.683
_PLAT_Y_MIN = 0.5527
_CENTER_X = (_PLAT_X_MIN + _PLAT_X_MAX) * 0.5   # 0.499
_CENTER_Y = _PLAT_Y_MIN                           # 0.5527


class WorldModelEnv(gym.Env):
    """Drop-in replacement for BrawlDeepEnv that runs inside a world model.

    * Same observation / action spaces as the real env.
    * ``reset()`` samples a real starting state from collected data.
    * ``step()`` predicts the next state via ``WorldModel.predict_np()``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model_path: str | Path,
        data_path: str | Path,
        max_episode_steps: int = 1200,
        device: str = "cpu",
        reset_noise_scale: float = 0.0,
    ):
        super().__init__()

        # ── load world model ────────────────────────────────────────
        self._device = device
        self._model = WorldModel()
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        self._model.load_state_dict(ckpt)
        self._model.to(device)
        self._model.eval()

        # ── load real-state pool for reset() ────────────────────────
        data = np.load(data_path)
        self._state_pool = data["states"].astype(np.float32)  # (N, state_dim)
        assert self._state_pool.shape[1] == STATE_DIM, (
            f"State pool dim {self._state_pool.shape[1]} != StateSpec.dim() {STATE_DIM}"
        )

        # Keep only grounded states — with jump disabled the agent
        # cannot control Y, so airborne starts produce unachievable goals.
        grounded_idx = StateSpec.index("player_grounded")
        grounded_mask = self._state_pool[:, grounded_idx] > 0.5
        n_grounded = int(grounded_mask.sum())
        if n_grounded >= 100:
            self._state_pool = self._state_pool[grounded_mask]
            print(f"[WorldModelEnv] Filtered state pool to {n_grounded}/{len(grounded_mask)} grounded states")
        else:
            print(f"[WorldModelEnv] WARNING: only {n_grounded} grounded states, keeping full pool")

        self._pool_size = self._state_pool.shape[0]

        # ── spaces (match BrawlDeepEnv exactly) ─────────────────────
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(STATE_DIM,), dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiDiscrete(ACTION_DIMS)

        # ── episode tracking ────────────────────────────────────────
        self._max_episode_steps = max_episode_steps
        self._reset_noise_scale = reset_noise_scale
        self._state: np.ndarray = np.zeros(STATE_DIM, dtype=np.float32)
        self._step_count: int = 0
        self._prev_stocks: float = 1.0

        # ── indices for prev_action features ────────────────────────
        self._prev_action_idx = [
            StateSpec.index("prev_movement"),
            StateSpec.index("prev_jump"),
            StateSpec.index("prev_dodge"),
            StateSpec.index("prev_attack"),
        ]
        # normalisation divisors matching structured_memory.to_vector()
        self._prev_action_div = np.array([3.0, 1.0, 1.0, 3.0], dtype=np.float32)

        # ── indices for clamping ────────────────────────────────────
        self._stocks_idx = StateSpec.index("self_stocks_norm")

    # ── reset ────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        idx = self.np_random.integers(0, self._pool_size)
        self._state = self._state_pool[idx].copy()

        # Zero out prev_action features (like a fresh episode)
        for i in self._prev_action_idx:
            self._state[i] = 0.0

        # Optional position jitter for diversity
        if self._reset_noise_scale > 0.0:
            noise = self.np_random.normal(0.0, self._reset_noise_scale, size=STATE_DIM).astype(np.float32)
            # Only add noise to continuous features (first ~4: x, y, vx, vy)
            self._state[:4] += noise[:4]

        self._clamp_state()
        self._step_count = 0
        self._prev_stocks = self._state[self._stocks_idx]

        return self._state.copy(), {}

    # ── step ─────────────────────────────────────────────────────────
    def step(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(1, -1)  # (1, 4)
        state_in = self._state.reshape(1, -1)                       # (1, D)

        next_state = self._model.predict_np(state_in, action).squeeze(0)  # (D,)

        # Overwrite prev_action with ground truth (not model prediction)
        flat_action = action.squeeze(0)
        for k, idx in enumerate(self._prev_action_idx):
            next_state[idx] = float(flat_action[k]) / self._prev_action_div[k]

        self._state = next_state
        self._clamp_state()
        self._enforce_consistency()
        self._step_count += 1

        # Detect stock loss for death penalty
        curr_stocks = self._state[self._stocks_idx]
        stock_lost = max(0.0, self._prev_stocks - curr_stocks)
        self._prev_stocks = curr_stocks

        truncated = self._step_count >= self._max_episode_steps
        terminated = False

        info = {
            "self_stock_lost_step": float(stock_lost),
            "imagination": True,
        }

        return self._state.copy(), 0.0, terminated, truncated, info

    # ── state clamping ───────────────────────────────────────────────
    def _clamp_state(self) -> None:
        """Clamp predicted state to physically valid ranges."""
        s = self._state

        # Positions [0, 1]
        for idx_name in ("player_x", "player_y", "opponent_x", "opponent_y"):
            i = StateSpec.index(idx_name)
            s[i] = np.clip(s[i], 0.0, 1.0)

        # Booleans → round to {0, 1}
        for idx_name in (
            "player_grounded", "player_has_weapon", "player_on_edge",
            "player_is_offstage", "opponent_grounded", "opponent_exists",
            "opponent_on_edge", "opponent_is_offstage", "in_strike_range",
            "weapon_on_ground", "ledge_is_occupied",
        ):
            i = StateSpec.index(idx_name)
            s[i] = float(round(np.clip(s[i], 0.0, 1.0)))

        # Velocities [-3, 3]
        for idx_name in ("player_vx", "player_vy", "opponent_vx", "opponent_vy",
                         "rel_vx", "rel_vy"):
            i = StateSpec.index(idx_name)
            s[i] = np.clip(s[i], -3.0, 3.0)

        # Normalised [0, 1] features
        for idx_name in (
            "player_damage_pct", "player_jumps_norm",
            "opponent_damage_pct", "opponent_jumps_norm",
            "self_stocks_norm", "op_stocks_norm",
            "dodge_cooldown_norm", "opponent_dodge_cooldown_norm",
            "player_airborne_time", "opponent_airborne_time",
            "player_hitstun", "opponent_hitstun",
            "rel_distance", "dist_to_nearest_ledge", "dist_to_stage_center",
        ):
            i = StateSpec.index(idx_name)
            s[i] = np.clip(s[i], 0.0, 1.0)

        # Signed normalised [-1, 1]
        for idx_name in (
            "facing_opponent", "player_facing_dir", "opponent_facing_dir",
            "last_knockback_dx", "last_knockback_dy",
            "frame_advantage_estimate",
        ):
            i = StateSpec.index(idx_name)
            s[i] = np.clip(s[i], -1.0, 1.0)

        # prev_action features [0, 1]
        for i in self._prev_action_idx:
            s[i] = np.clip(s[i], 0.0, 1.0)

    def _enforce_consistency(self) -> None:
        """Recompute derived features from primary ones (physics projection).

        The world model predicts every feature independently; over long
        rollouts, algebraically-related features drift apart.  This step
        enforces exact geometric constraints cheaply.
        """
        s = self._state
        px = s[StateSpec.index("player_x")]
        py = s[StateSpec.index("player_y")]
        ox = s[StateSpec.index("opponent_x")]
        oy = s[StateSpec.index("opponent_y")]
        pvx = s[StateSpec.index("player_vx")]
        pvy = s[StateSpec.index("player_vy")]
        ovx = s[StateSpec.index("opponent_vx")]
        ovy = s[StateSpec.index("opponent_vy")]

        # relational geometry
        rdx = ox - px
        rdy = oy - py
        rdist = np.sqrt(rdx ** 2 + rdy ** 2)
        s[StateSpec.index("rel_dx")] = np.clip(rdx, -3.0, 3.0)
        s[StateSpec.index("rel_dy")] = np.clip(rdy, -3.0, 3.0)
        s[StateSpec.index("rel_distance")] = np.clip(rdist, 0.0, 1.0)

        # relative velocity
        s[StateSpec.index("rel_vx")] = np.clip(ovx - pvx, -3.0, 3.0)
        s[StateSpec.index("rel_vy")] = np.clip(ovy - pvy, -3.0, 3.0)

        # distance to stage center
        dx_center = px - _CENTER_X
        dy_center = py - _CENTER_Y
        dist_center = np.sqrt(dx_center ** 2 + dy_center ** 2)
        s[StateSpec.index("dist_to_stage_center")] = np.clip(dist_center, 0.0, 1.0)
        s[StateSpec.index("signed_dx_to_stage_center")] = np.clip(_CENTER_X - px, -1.0, 1.0)

        # distance to nearest ledge
        dl = abs(px - _PLAT_X_MIN)
        dr = abs(px - _PLAT_X_MAX)
        if dl < dr:
            lx, ly = _PLAT_X_MIN, _PLAT_Y_MIN
        else:
            lx, ly = _PLAT_X_MAX, _PLAT_Y_MIN
        dist_ledge = np.sqrt((px - lx) ** 2 + (py - ly) ** 2)
        s[StateSpec.index("dist_to_nearest_ledge")] = np.clip(dist_ledge, 0.0, 1.0)
        s[StateSpec.index("signed_dx_to_ledge")] = np.clip(lx - px, -1.0, 1.0)
        s[StateSpec.index("dy_to_ledge")] = np.clip(py - ly, -1.0, 1.0)

        # offstage flag (player outside platform bounds)
        offstage = 1.0 if (px < _PLAT_X_MIN or px > _PLAT_X_MAX) else 0.0
        s[StateSpec.index("player_is_offstage")] = offstage


# ── CLI smoke test ───────────────────────────────────────────────────
def _smoke_test():
    parser = argparse.ArgumentParser(description="WorldModelEnv smoke test")
    parser.add_argument("--model", type=str, required=True, help="Path to world_model.pt")
    parser.add_argument("--data", type=str, required=True, help="Path to transitions.npz")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    env = WorldModelEnv(model_path=args.model, data_path=args.data)

    print(f"obs_space: {env.observation_space.shape}, action_space: {env.action_space}")
    print(f"State pool: {env._pool_size} states")

    obs, _ = env.reset()
    assert obs.shape == (STATE_DIM,), f"Bad obs shape: {obs.shape}"
    assert not np.any(np.isnan(obs)), "NaN in initial obs"

    t0 = time.perf_counter()
    total_reward = 0.0
    for i in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        assert obs.shape == (STATE_DIM,), f"Step {i}: bad obs shape {obs.shape}"
        assert not np.any(np.isnan(obs)), f"Step {i}: NaN in obs"
        if terminated or truncated:
            obs, _ = env.reset()
    elapsed = time.perf_counter() - t0

    fps = args.steps / elapsed
    print(f"\n{args.steps} steps in {elapsed:.3f}s  ({fps:.0f} fps)")
    print(f"Total reward: {total_reward:.4f}")
    print("Smoke test PASSED")


if __name__ == "__main__":
    _smoke_test()
