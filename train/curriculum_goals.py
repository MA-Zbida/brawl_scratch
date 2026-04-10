from __future__ import annotations

from typing import Dict

import numpy as np

from feature_extractor.memory.state_spec import StateSpec


CURRICULUM_GOAL_FEATURES: list[str] = [
    "player_x",
    "player_y",
    "player_has_weapon",
    "weapon_dx",
    "weapon_dy",
    "in_strike_range",
    "rel_distance",
    "facing_opponent",
    "frame_advantage_estimate",
    "opponent_damage_pct",
    "player_is_offstage",
]

GOAL_INDEX: Dict[str, int] = {name: i for i, name in enumerate(CURRICULUM_GOAL_FEATURES)}
GOAL_DIM: int = len(CURRICULUM_GOAL_FEATURES)


def _norm01(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    z = (float(v) - float(lo)) / (float(hi) - float(lo))
    return float(np.clip(z, 0.0, 1.0))


def extract_curriculum_goal_features(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    return np.array(
        [
            float(np.clip(StateSpec.get(obs, "player_x"), 0.0, 1.0)),
            float(np.clip(StateSpec.get(obs, "player_y"), 0.0, 1.0)),
            float(np.clip(StateSpec.get(obs, "player_has_weapon"), 0.0, 1.0)),
            _norm01(StateSpec.get(obs, "weapon_dx"), -1.0, 1.0),
            _norm01(StateSpec.get(obs, "weapon_dy"), -1.0, 1.0),
            float(np.clip(StateSpec.get(obs, "in_strike_range"), 0.0, 1.0)),
            _norm01(StateSpec.get(obs, "rel_distance"), 0.0, 2.0),
            _norm01(StateSpec.get(obs, "facing_opponent"), -1.0, 1.0),
            _norm01(StateSpec.get(obs, "frame_advantage_estimate"), -1.0, 1.0),
            float(np.clip(StateSpec.get(obs, "opponent_damage_pct"), 0.0, 1.0)),
            float(np.clip(StateSpec.get(obs, "player_is_offstage"), 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


def default_goal_target() -> np.ndarray:
    target = np.zeros((GOAL_DIM,), dtype=np.float32)
    target[GOAL_INDEX["weapon_dx"]] = 0.5
    target[GOAL_INDEX["weapon_dy"]] = 0.5
    target[GOAL_INDEX["facing_opponent"]] = 0.5
    target[GOAL_INDEX["frame_advantage_estimate"]] = 0.5
    return target


def clip_goal_target(target: np.ndarray) -> np.ndarray:
    arr = np.asarray(target, dtype=np.float32).reshape(-1)
    if arr.shape[0] != GOAL_DIM:
        raise ValueError(f"Expected goal dim={GOAL_DIM}, got {arr.shape[0]}")
    return np.clip(arr, 0.0, 1.0).astype(np.float32)
