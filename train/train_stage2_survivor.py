#!/usr/bin/env python
"""Stage 2 LLC training: Survivor (Recovery & Platforms).

Goal: return to stage / recover from offstage.
Goal dim: 7 (unified); active features: dist_ledge (idx 1), grounded (idx 3).
  dist_ledge = dist_to_nearest_ledge / 2.0, normalized [0,1]. Target ~0.04.
  grounded   = player_grounded binary. Target 1.0.
Curriculum: reset perturbation to force aerial/offstage-like starts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.llc_stage_common import StageGoalEnv, StageSpec, make_base_env, parse_train_args, train_stage_model


def _target_sampler(_: np.ndarray) -> np.ndarray:
    # 7-dim: minimize dist_ledge + be grounded.
    # Slight jitter on dist_ledge for variety; grounded is always 1.0.
    dist_ledge_target = float(np.clip(np.random.uniform(0.02, 0.08), 0.02, 0.10))
    return np.array([0.0, dist_ledge_target, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def make_env(max_episode_steps: int):
    base = make_base_env(max_episode_steps=max_episode_steps, terminate_on_stock_out=False)
    spec = StageSpec(
        stage_id=2,
        name="stage2_survivor",
        mask=np.array([0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32),
        target_sampler=_target_sampler,
        min_goal_duration=16,
        max_goal_duration=32,
        progress_scale=1.3,
        progress_clip_min=-0.1,
        progress_clip_max=0.3,
        success_threshold=0.10,
        success_bonus=0.30,
        reward_clip=1.0,
        disable_attack=True,
        reset_perturb_steps=18,
    )
    return StageGoalEnv(base, spec)


def main() -> None:
    args = parse_train_args(default_name="llc_stage2_survivor", default_steps=700_000)
    train_stage_model(args=args, make_env=lambda: make_env(args.max_episode_steps))


if __name__ == "__main__":
    main()
