#!/usr/bin/env python
"""Stage 4 LLC training: Punisher (Combat Basics).

Goal: maximize advantage in close combat.
Goal dim: 7 (unified); active features:
  in_strike_range (idx 2): target 1.0 (always in range).
  rel_distance    (idx 5): target ~0.12 (close, normalized).
  frame_advantage (idx 6): target ~0.75 (normalized from [-1,1] range).
Full action space enabled.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.llc_stage_common import StageGoalEnv, StageSpec, make_base_env, parse_train_args, train_stage_model


def _target_sampler(_: np.ndarray) -> np.ndarray:
    # 7-dim: in_strike_range=1, rel_distance close, frame_advantage high.
    rel_dist_target = float(np.clip(np.random.uniform(0.08, 0.18), 0.05, 0.22))
    frame_adv_target = float(np.clip(np.random.uniform(0.60, 1.0), 0.50, 1.0))
    return np.array([0.0, 0.0, 1.0, 0.0, 0.0, rel_dist_target, frame_adv_target], dtype=np.float32)


def make_env(max_episode_steps: int):
    base = make_base_env(max_episode_steps=max_episode_steps, terminate_on_stock_out=False)
    spec = StageSpec(
        stage_id=4,
        name="stage4_punisher",
        mask=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        target_sampler=_target_sampler,
        min_goal_duration=16,
        max_goal_duration=24,
        progress_scale=1.1,
        progress_clip_min=-0.1,
        progress_clip_max=0.3,
        success_threshold=0.10,
        success_bonus=0.40,
        reward_clip=1.2,
        disable_attack=False,
    )
    return StageGoalEnv(base, spec)


def main() -> None:
    args = parse_train_args(default_name="llc_stage4_punisher", default_steps=1_200_000)
    train_stage_model(args=args, make_env=lambda: make_env(args.max_episode_steps))


if __name__ == "__main__":
    main()
