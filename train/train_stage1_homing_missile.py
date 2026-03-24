#!/usr/bin/env python
"""Stage 1 LLC training: Homing Missile (Locomotion).

Goal: reach stage center.
Masking: dist_to_stage_center + player_x + player_y.
Expected behavior: consistently move and remain near center.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.llc_stage_common import StageGoalEnv, StageSpec, make_base_env, parse_train_args, train_stage_model


def _target_sampler(_: np.ndarray) -> np.ndarray:
    # [dist_to_stage_center, player_x, player_y]
    # stage center x = 0.5, y around main platform line
    return np.array([0.0, 0.5, 0.55], dtype=np.float32)


def make_env(max_episode_steps: int):
    base = make_base_env(max_episode_steps=max_episode_steps, terminate_on_stock_out=False)
    spec = StageSpec(
        stage_id=1,
        name="stage1_homing_missile",
        feature_names=["dist_to_stage_center", "player_x", "player_y"],
        mask=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        target_sampler=_target_sampler,
        min_goal_duration=32,
        max_goal_duration=64,
        progress_scale=1.2,
        progress_clip_min=-0.1,
        progress_clip_max=0.3,
        success_threshold=0.08,
        success_bonus=0.20,
        reward_clip=1.0,
        disable_attack=True,
    )
    return StageGoalEnv(base, spec)


def main() -> None:
    args = parse_train_args(default_name="llc_stage1_homing_missile", default_steps=500_000)
    train_stage_model(args=args, make_env=lambda: make_env(args.max_episode_steps))


if __name__ == "__main__":
    main()
