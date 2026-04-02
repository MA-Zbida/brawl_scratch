#!/usr/bin/env python
"""Stage 3 LLC training: Shadow (Relational Positioning).

Goal: maintain a controlled spacing to opponent.
Goal dim: 7 (unified); active feature: rel_distance (idx 5).
  rel_distance = rel_distance_world / 2.0, normalized [0,1]. Target ~0.35.
Constraint: attack buttons disabled.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.llc_stage_common import StageGoalEnv, StageSpec, make_base_env, parse_train_args, train_stage_model


def _target_sampler(obs: np.ndarray) -> np.ndarray:
    # 7-dim: target rel_distance only (index 5).
    # Jitter around 0.35 (world dist ~0.70) for variety.
    rel_dist_target = float(np.clip(np.random.uniform(0.28, 0.42), 0.20, 0.50))
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, rel_dist_target, 0.0], dtype=np.float32)


def make_env(max_episode_steps: int):
    base = make_base_env(max_episode_steps=max_episode_steps, terminate_on_stock_out=False)
    spec = StageSpec(
        stage_id=3,
        name="stage3_shadow",
        mask=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
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
