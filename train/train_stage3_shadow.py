#!/usr/bin/env python
"""Stage 3 LLC training: Shadow (Relational Positioning).

Goal: maintain a controlled distance to opponent (rel_distance -> 0.2).
Masking: rel_dx, rel_dy, rel_distance.
Constraint: attack buttons disabled.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.llc_stage_common import StageGoalEnv, StageSpec, make_base_env, parse_train_args, train_stage_model


def _target_sampler(obs: np.ndarray) -> np.ndarray:
    # [rel_dx, rel_dy, rel_distance]
    # keep ~0.2 spacing, neutral vertical alignment
    side = 1.0 if np.random.rand() < 0.5 else -1.0
    return np.array([0.2 * side, 0.0, 0.2], dtype=np.float32)


def make_env(max_episode_steps: int):
    base = make_base_env(max_episode_steps=max_episode_steps, terminate_on_stock_out=False)
    spec = StageSpec(
        stage_id=3,
        name="stage3_shadow",
        feature_names=["rel_dx", "rel_dy", "rel_distance"],
        mask=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        target_sampler=_target_sampler,
        min_goal_duration=24,
        max_goal_duration=40,
        progress_scale=1.0,
        progress_clip_min=-0.1,
        progress_clip_max=0.3,
        success_threshold=0.10,
        success_bonus=0.25,
        reward_clip=1.0,
        disable_attack=True,
    )
    return StageGoalEnv(base, spec)


def main() -> None:
    args = parse_train_args(default_name="llc_stage3_shadow", default_steps=900_000)
    train_stage_model(args=args, make_env=lambda: make_env(args.max_episode_steps))


if __name__ == "__main__":
    main()
