"""Centralised observation-vector layout registry.

Every consumer (LLC, HSP, diagnostics) imports from here instead of
hard-coding array indices. If the layout ever changes, this file is the
single source of truth.

Usage::

    from feature_extractor.memory.state_spec import StateSpec

    idx = StateSpec.index("rel_distance")
    val = StateSpec.get(obs, "opponent_damage_pct")
    px, py = StateSpec.get_multi(obs, "player_x", "player_y")
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class StateSpec:
    """Central registry mapping feature names → observation vector indices."""

    # ── Canonical feature order ─────────────────────────────────────
    # Changing this list changes the obs layout for ALL agents.
    # After any edit: retrain LLC and HSP from scratch.
    FEATURES: list[str] = [
        # ── player (10) ─────────────────────────────────────────────
        "player_x",             # 0   normalised [0, 1]
        "player_y",             # 1
        "player_vx",            # 2   clamped velocity
        "player_vy",            # 3
        "player_grounded",      # 4   bool → float
        "player_damage_pct",    # 5   [0, 1]
        "player_has_weapon",    # 6   bool → float
        "player_jumps_norm",    # 7   jumps_left / 3
        "player_on_edge",       # 8   bool → float
        "player_is_offstage",   # 9   bool → float

        # ── opponent (10) ───────────────────────────────────────────
        "opponent_x",           # 10  NEW — absolute position
        "opponent_y",           # 11  NEW
        "opponent_vx",          # 12
        "opponent_vy",          # 13
        "opponent_grounded",    # 14
        "opponent_damage_pct",  # 15
        "opponent_exists",      # 16
        "opponent_jumps_norm",  # 17
        "opponent_on_edge",     # 18
        "opponent_is_offstage", # 19  bool → float

        # ── relational (8) ──────────────────────────────────────────
        "rel_dx",               # 20  signed (opponent.x − player.x)
        "rel_dy",               # 21
        "rel_distance",         # 22  euclidean
        "in_strike_range",      # 23  bool → float
        "facing_opponent",      # 24  +1 facing, −1 away, 0 unknown
        "player_facing_dir",    # 25  +1 right, −1 left, 0 unknown
        "weapon_dx",            # 26  signed (weapon.x − player.x)
        "weapon_dy",            # 27  signed (weapon.y − player.y)

        # ── game state (4) ──────────────────────────────────────────
        "self_stocks_norm",     # 28
        "op_stocks_norm",       # 29
        "dodge_cooldown_norm",  # 30  cooldown / dynamic_max (3.2 air, 1.0 grounded)
        "weapon_on_ground",     # 31

        # ── recovery information (3) ────────────────────────────────
        "dist_to_nearest_ledge", # 32
        "signed_dx_to_ledge",    # 33
        "dy_to_ledge",           # 34

        # ── temporal / strategic (4) ────────────────────────────────
        "player_airborne_time",   # 35  frames airborne / 60
        "opponent_airborne_time", # 36
        "player_hitstun",        # 37  hitstun timer / max
        "opponent_hitstun",      # 38

        # ── added strategic/combat features (12) ───────────────────
        "opponent_facing_dir",            # 39  +1 right, -1 left, 0 unknown
        "player_time_since_hit",          # 40  normalised timer
        "opponent_time_since_hit",        # 41
        "last_knockback_dx",              # 42  signed velocity proxy
        "last_knockback_dy",              # 43
        "dist_to_stage_center",           # 44
        "signed_dx_to_stage_center",      # 45
        "ledge_is_occupied",              # 46
        "opponent_dodge_cooldown_norm",   # 47
        "rel_vx",                         # 48  opponent_vx - player_vx
        "rel_vy",                         # 49  opponent_vy - player_vy
        "frame_advantage_estimate",       # 50  opponent_hitstun - player_hitstun
    ]

    _INDEX: dict[str, int] = {name: i for i, name in enumerate(FEATURES)}

    # ── public API ──────────────────────────────────────────────────

    @classmethod
    def dim(cls) -> int:
        """Total observation dimensions."""
        return len(cls.FEATURES)

    @classmethod
    def index(cls, name: str) -> int:
        """Return the integer index of *name* in the observation vector.

        Raises KeyError if the name is not in the registry.
        """
        return cls._INDEX[name]

    @classmethod
    def get(cls, obs: np.ndarray, name: str) -> float:
        """Fetch a single named feature from a flat obs array."""
        return float(obs[cls._INDEX[name]])

    @classmethod
    def get_multi(cls, obs: np.ndarray, *names: str) -> np.ndarray:
        """Fetch multiple named features as a float32 array."""
        indices = [cls._INDEX[n] for n in names]
        return obs[indices].astype(np.float32)

    @classmethod
    def names(cls) -> list[str]:
        """Return a copy of the feature name list."""
        return list(cls.FEATURES)

    @classmethod
    def validate_vector(cls, vec: np.ndarray) -> None:
        """Assert that *vec* has the expected shape."""
        expected = cls.dim()
        actual = vec.shape[-1] if vec.ndim >= 1 else 0
        if actual != expected:
            raise ValueError(
                f"Observation vector has {actual} dims, expected {expected}. "
                f"Did you update StateSpec.FEATURES without rebuilding "
                f"Memory.to_vector()?"
            )
