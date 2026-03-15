"""Continuous goal definitions and structured sampling for LLC/HSP."""

from __future__ import annotations

import numpy as np

from feature_extractor.memory.state_spec import StateSpec

GOAL_DIM = 6

GOAL_NAMES = [
    "target_rel_dx",
    "target_rel_dy",
    "target_player_vx",
    "target_player_vy",
    "target_dx_to_ledge",
    "target_dy_to_ledge",
]

# Goal bounds aligned with training design.
GOAL_LOW = np.array([-0.6, -0.4, -0.8, -1.0, -1.0, -1.0], dtype=np.float32)
GOAL_HIGH = np.array([0.6, 0.4, 0.8, 1.0, 1.0, 1.0], dtype=np.float32)


def clip_goal(goal: np.ndarray) -> np.ndarray:
    goal = np.asarray(goal, dtype=np.float32)
    return np.clip(goal, GOAL_LOW, GOAL_HIGH).astype(np.float32)


class GoalSampler:
    """Structured sampler for realistic continuous goals."""

    def __init__(
        self,
        recovery_prob: float = 0.18,
        engage_prob: float = 0.22,
        spacing_prob: float = 0.20,
        seed: int | None = None,
    ):
        self.recovery_prob = float(recovery_prob)
        self.engage_prob = float(engage_prob)
        self.spacing_prob = float(spacing_prob)
        self.rng = np.random.default_rng(seed)

    def sample(self, obs: np.ndarray) -> np.ndarray:
        # Base sample in realistic combat ranges.
        goal = np.array(
            [
                self.rng.uniform(-0.6, 0.6),
                self.rng.uniform(-0.4, 0.4),
                self.rng.uniform(-0.8, 0.8),
                self.rng.uniform(-1.0, 1.0),
                self.rng.uniform(-1.0, 1.0),
                self.rng.uniform(-1.0, 1.0),
            ],
            dtype=np.float32,
        )

        p = self.rng.random()

        # Recovery-focused goals: move toward ledge and stabilise vertically.
        if p < self.recovery_prob:
            goal[4] = self.rng.normal(0.0, 0.08)
            goal[5] = self.rng.normal(0.0, 0.08)
            return clip_goal(goal)

        # Engagement goals: seek close proximity and pressure windows.
        if p < self.recovery_prob + self.engage_prob:
            goal[0] = self.rng.normal(0.0, 0.08)
            goal[1] = self.rng.normal(0.0, 0.06)
            return clip_goal(goal)

        # Spacing goals: keep actionable distance while preserving center control.
        if p < self.recovery_prob + self.engage_prob + self.spacing_prob:
            sign = 1.0 if self.rng.random() > 0.5 else -1.0
            goal[0] = sign * self.rng.uniform(0.2, 0.45)
            goal[1] = self.rng.uniform(-0.12, 0.12)

        # Softly bias ledge targets around current side when available.
        if obs is not None and len(obs) >= StateSpec.dim():
            current_dx = StateSpec.get(obs, "signed_dx_to_ledge")
            goal[4] = np.clip(0.5 * goal[4] + 0.5 * np.tanh(current_dx), -1.0, 1.0)

        return clip_goal(goal)
