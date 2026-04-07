#!/usr/bin/env python
"""Stage 1 LLC training: Homing Missile (Locomotion).

Goal: reach stage center.
Goal dim: 7 (unified); active feature: dist_center (index 0).
dist_center = dist_to_stage_center / 2.0 (normalized [0,1]).
Target ~0.07 ≈ 0.14 world units from center — close but not exact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hierarchical.goals import GOAL_FEATURE_NAMES
from train.llc_stage_common import StageGoalEnv, StageSpec, make_base_env, parse_train_args, train_stage_model
from wrappers.goal_env_wrapper import FlattenMultiDiscreteWrapper, StageGoalDictEnv


_curriculum_step: int = 0  # shared counter incremented by StageGoalEnv or manually


def _target_sampler(obs: np.ndarray) -> np.ndarray:
    # Sample a random x position on the platform [0.34, 0.66].
    global _curriculum_step
    target_x = np.random.uniform(0.34, 0.66)
    _curriculum_step += 1
    return np.array([target_x, 0, 0, 0, 0, 0, 0], dtype=np.float32)


def _has_cli_flag(flag: str) -> bool:
    for token in sys.argv[1:]:
        if token == flag or token.startswith(f"{flag}="):
            return True
    return False


def make_env(max_episode_steps: int, algo: str = "ppo"):
    base = make_base_env(max_episode_steps=max_episode_steps, terminate_on_stock_out=False)
    spec = _make_spec()
    env = StageGoalEnv(base, spec)
    if algo == "sac":
        env = FlattenMultiDiscreteWrapper(env)
        env = StageGoalDictEnv(
            env,
            proximity_scale=spec.proximity_scale,
            success_threshold=spec.success_threshold,
            success_bonus=spec.success_bonus,
            mask=spec.mask,
        )
    return env


def _make_spec() -> StageSpec:
    return StageSpec(
        stage_id=1,
        name="stage1_homing_missile",
        mask=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        target_sampler=_target_sampler,
        min_goal_duration=16,
        max_goal_duration=28,
        progress_scale=4.0,
        progress_clip_min=-0.10,
        progress_clip_max=0.80,
        success_threshold=0.03,
        success_bonus=2.0,
        proximity_scale=8.0,
        death_penalty=2.0,
        velocity_penalty_scale=0.0,
        stay_bonus=0.0,
        reward_clip=3.0,
        disable_attack=True,
        reset_perturb_steps=6,
        feature_names=list(GOAL_FEATURE_NAMES),
    )


def main() -> None:
    args = parse_train_args(default_name="llc_stage1_homing_missile", default_steps=500_000)
    algo = getattr(args, "algo", "ppo")

    if not _has_cli_flag("--learning-rate"):
        args.learning_rate = 2e-4

    if algo == "ppo":
        if not _has_cli_flag("--n-steps"):
            args.n_steps = 512
        if not _has_cli_flag("--clip-range"):
            args.clip_range = 0.15
        if not _has_cli_flag("--ent-coef"):
            args.ent_coef = 0.02

    spec = _make_spec()

    train_stage_model(
        args=args,
        make_env=lambda: make_env(args.max_episode_steps, algo=algo),
        stage_spec=spec,
    )


if __name__ == "__main__":
    main()
