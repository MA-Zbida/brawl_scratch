"""Structured masked goals for goal-conditioned LLC/HSP.

Goal vector format is concatenated as:
    goal = [target_0..target_{k-1}, mask_0..mask_{k-1}]

Targets are normalized in [0, 1], masks are binary {0, 1}.
"""

from __future__ import annotations

import numpy as np

from feature_extractor.memory.state_spec import StateSpec

GOAL_FEATURE_NAMES = [
    "dist_center",
    "dist_ledge",
    "in_strike_range",
    "grounded",
    "offstage",
    "rel_distance",
    "frame_advantage",
]

GOAL_TARGET_DIM = len(GOAL_FEATURE_NAMES)
GOAL_DIM = GOAL_TARGET_DIM * 2

GOAL_NAMES = [f"g_{n}" for n in GOAL_FEATURE_NAMES] + [f"m_{n}" for n in GOAL_FEATURE_NAMES]

GOAL_TARGET_LOW = np.zeros((GOAL_TARGET_DIM,), dtype=np.float32)
GOAL_TARGET_HIGH = np.ones((GOAL_TARGET_DIM,), dtype=np.float32)
GOAL_MASK_LOW = np.zeros((GOAL_TARGET_DIM,), dtype=np.float32)
GOAL_MASK_HIGH = np.ones((GOAL_TARGET_DIM,), dtype=np.float32)

GOAL_LOW = np.concatenate([GOAL_TARGET_LOW, GOAL_MASK_LOW]).astype(np.float32)
GOAL_HIGH = np.concatenate([GOAL_TARGET_HIGH, GOAL_MASK_HIGH]).astype(np.float32)


def _norm01(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    z = (float(v) - float(lo)) / (float(hi) - float(lo))
    return float(np.clip(z, 0.0, 1.0))


def split_goal(goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    goal = np.asarray(goal, dtype=np.float32).reshape(-1)
    if goal.shape[0] != GOAL_DIM:
        raise ValueError(f"Expected goal dim={GOAL_DIM}, got {goal.shape[0]}")
    target = goal[:GOAL_TARGET_DIM]
    mask = goal[GOAL_TARGET_DIM:]
    return target.astype(np.float32), mask.astype(np.float32)


def pack_goal(target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    mask = np.asarray(mask, dtype=np.float32).reshape(-1)
    if target.shape[0] != GOAL_TARGET_DIM or mask.shape[0] != GOAL_TARGET_DIM:
        raise ValueError(
            f"Expected target/mask dims={GOAL_TARGET_DIM}, got {target.shape[0]}/{mask.shape[0]}"
        )
    return np.concatenate([target, mask], dtype=np.float32)


def extract_goal_features(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    return np.array(
        [
            _norm01(StateSpec.get(obs, "dist_to_stage_center"), 0.0, 2.0),
            _norm01(StateSpec.get(obs, "dist_to_nearest_ledge"), 0.0, 2.0),
            float(np.clip(StateSpec.get(obs, "in_strike_range"), 0.0, 1.0)),
            float(np.clip(StateSpec.get(obs, "player_grounded"), 0.0, 1.0)),
            float(np.clip(StateSpec.get(obs, "player_is_offstage"), 0.0, 1.0)),
            _norm01(StateSpec.get(obs, "rel_distance"), 0.0, 2.0),
            _norm01(StateSpec.get(obs, "frame_advantage_estimate"), -1.0, 1.0),
        ],
        dtype=np.float32,
    )


def clip_goal(goal: np.ndarray) -> np.ndarray:
    goal = np.asarray(goal, dtype=np.float32)
    if goal.shape[0] != GOAL_DIM:
        raise ValueError(f"Expected goal dim={GOAL_DIM}, got {goal.shape[0]}")
    target, mask = split_goal(goal)
    target = np.clip(target, GOAL_TARGET_LOW, GOAL_TARGET_HIGH).astype(np.float32)
    mask = (mask >= 0.5).astype(np.float32)
    if float(np.sum(mask)) <= 0.0:
        mask[0] = 1.0
    return pack_goal(target, mask)


class GoalSampler:
    """Structured sampler for combat tasks as target-state regions."""

    def __init__(
        self,
        center_prob: float = 0.30,
        recovery_prob: float = 0.25,
        spacing_prob: float = 0.20,
        pressure_prob: float = 0.25,
        seed: int | None = None,
    ):
        self.center_prob = float(center_prob)
        self.recovery_prob = float(recovery_prob)
        self.spacing_prob = float(spacing_prob)
        self.pressure_prob = float(pressure_prob)
        self.rng = np.random.default_rng(seed)
        self.last_lane = "center_control"

    def _make_goal(self, target: np.ndarray, mask: np.ndarray, lane: str) -> np.ndarray:
        self.last_lane = lane
        return clip_goal(pack_goal(target.astype(np.float32), mask.astype(np.float32)))

    def _sample_center_control(self) -> np.ndarray:
        t = np.zeros((GOAL_TARGET_DIM,), dtype=np.float32)
        m = np.zeros((GOAL_TARGET_DIM,), dtype=np.float32)

        t[0] = self.rng.uniform(0.04, 0.20)
        t[3] = 1.0
        t[4] = 0.0
        m[[0, 3, 4]] = 1.0
        return self._make_goal(t, m, "center_control")

    def _sample_recovery(self) -> np.ndarray:
        t = np.zeros((GOAL_TARGET_DIM,), dtype=np.float32)
        m = np.zeros((GOAL_TARGET_DIM,), dtype=np.float32)

        t[1] = self.rng.uniform(0.02, 0.16)
        t[4] = 0.0
        t[3] = self.rng.uniform(0.6, 1.0)
        m[[1, 4, 3]] = 1.0
        return self._make_goal(t, m, "recovery")

    def _sample_spacing(self) -> np.ndarray:
        t = np.zeros((GOAL_TARGET_DIM,), dtype=np.float32)
        m = np.zeros((GOAL_TARGET_DIM,), dtype=np.float32)

        t[5] = self.rng.uniform(0.35, 0.60)
        t[2] = 0.0
        m[[5, 2]] = 1.0
        return self._make_goal(t, m, "spacing")

    def _sample_pressure(self) -> np.ndarray:
        t = np.zeros((GOAL_TARGET_DIM,), dtype=np.float32)
        m = np.zeros((GOAL_TARGET_DIM,), dtype=np.float32)

        t[2] = 1.0
        t[6] = self.rng.uniform(0.60, 1.0)
        t[5] = self.rng.uniform(0.10, 0.35)
        m[[2, 6, 5]] = 1.0
        return self._make_goal(t, m, "pressure")

    def sample(self, obs: np.ndarray) -> np.ndarray:
        _ = obs
        total = max(1e-8, self.center_prob + self.recovery_prob + self.spacing_prob + self.pressure_prob)
        p = self.rng.random()
        c0 = self.center_prob / total
        c1 = c0 + (self.recovery_prob / total)
        c2 = c1 + (self.spacing_prob / total)

        if p < c0:
            return self._sample_center_control()
        if p < c1:
            return self._sample_recovery()
        if p < c2:
            return self._sample_spacing()
        return self._sample_pressure()
