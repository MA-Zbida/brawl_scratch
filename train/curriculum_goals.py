from __future__ import annotations

from typing import Dict

import numpy as np

from feature_extractor.memory.state_spec import StateSpec


# ── Goal feature set (11 dims, one family per stage) ───────────────────────────
CURRICULUM_GOAL_FEATURES: list[str] = [
    # Stage 1 – Recovery
    "signed_dx_to_ledge",       # 0  signed dx to nearest ledge, normalised [-1,1]→[0,1]
    "dy_to_ledge",              # 1  dy to nearest ledge, normalised [-1,1]→[0,1]
    # Stage 2 – Movement
    "player_x",                 # 2  absolute x position [0,1]
    "player_y",                 # 3  absolute y position [0,1]
    # Stage 3 – Weapon
    "player_has_weapon",        # 4  holding weapon {0,1}
    "weapon_dx",                # 5  signed dx to weapon, normalised [-1,1]→[0,1]
    "weapon_dy",                # 6  signed dy to weapon, normalised [-1,1]→[0,1]
    # Stage 4 – Spacing
    "rel_distance",             # 7  distance to opponent, normalised [0,2]→[0,1]
    "rel_dy",                   # 8  signed vertical offset to opponent, normalised [-1,1]→[0,1]
    # Stage 5 – Combat
    "in_strike_range",          # 9  in punish range {0,1}
    "opponent_damage_pct",      # 10 opponent damage, measured from the UI [0,1]
]

GOAL_INDEX: Dict[str, int] = {name: i for i, name in enumerate(CURRICULUM_GOAL_FEATURES)}
GOAL_DIM: int = len(CURRICULUM_GOAL_FEATURES)

GOAL_TYPES: tuple[str, ...] = (
    "recovery",
    "movement",
    "weapon_acquisition",
    "spacing",
    "combat",
)
GOAL_TYPE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(GOAL_TYPES)}
GOAL_TYPE_DIM: int = len(GOAL_TYPES)


def normalize_goal_type(goal_type: str) -> str:
    key = str(goal_type).strip().lower()
    if key not in GOAL_TYPE_INDEX:
        raise ValueError(f"Unknown goal type '{goal_type}'. Expected one of: {', '.join(GOAL_TYPES)}")
    return key


def goal_type_to_index(goal_type: str) -> int:
    return int(GOAL_TYPE_INDEX[normalize_goal_type(goal_type)])


def goal_type_onehot(goal_type: str) -> np.ndarray:
    onehot = np.zeros((GOAL_TYPE_DIM,), dtype=np.float32)
    onehot[goal_type_to_index(goal_type)] = 1.0
    return onehot


def _norm01(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    z = (float(v) - float(lo)) / (float(hi) - float(lo))
    return float(np.clip(z, 0.0, 1.0))


def extract_curriculum_goal_features(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    return np.array(
        [
            # Stage 1 – Recovery geometry
            _norm01(StateSpec.get(obs, "signed_dx_to_ledge"), -1.0, 1.0),      # 0
            _norm01(StateSpec.get(obs, "dy_to_ledge"), -1.0, 1.0),             # 1
            # Stage 2 – Movement positioning
            float(np.clip(StateSpec.get(obs, "player_x"), 0.0, 1.0)),          # 2
            float(np.clip(StateSpec.get(obs, "player_y"), 0.0, 1.0)),          # 3
            # Stage 3 – Weapon acquisition
            float(np.clip(StateSpec.get(obs, "player_has_weapon"), 0.0, 1.0)), # 4
            _norm01(StateSpec.get(obs, "weapon_dx"), -1.0, 1.0),               # 5
            _norm01(StateSpec.get(obs, "weapon_dy"), -1.0, 1.0),               # 6
            # Stage 4 – Spacing geometry
            _norm01(StateSpec.get(obs, "rel_distance"), 0.0, 2.0),             # 7
            _norm01(StateSpec.get(obs, "rel_dy"), -1.0, 1.0),                  # 8
            # Stage 5 – Combat execution
            float(np.clip(StateSpec.get(obs, "in_strike_range"), 0.0, 1.0)),   # 9
            float(np.clip(StateSpec.get(obs, "opponent_damage_pct"), 0.0, 1.0)), # 10
        ],
        dtype=np.float32,
    )


def default_goal_target() -> np.ndarray:
    """All dimensions at 0.5 — the normalised midpoint / neutral position for every family."""
    return np.full((GOAL_DIM,), 0.5, dtype=np.float32)


def clip_goal_target(target: np.ndarray) -> np.ndarray:
    arr = np.asarray(target, dtype=np.float32).reshape(-1)
    if arr.shape[0] != GOAL_DIM:
        raise ValueError(f"Expected goal dim={GOAL_DIM}, got {arr.shape[0]}")
    return np.clip(arr, 0.0, 1.0).astype(np.float32)
