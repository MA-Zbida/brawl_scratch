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
    # 7-dim unified goal; only dist_center (index 0) is active (mask=[1,0,...,0]).
    # dist_center = dist_to_stage_center / 2.0, normalized [0, 1].
    #
    # Curriculum: start with a generous target radius shrinking from 0.15 → 0.07
    # over 200K steps, ensuring early success signals.
    global _curriculum_step
    ramp = min(1.0, _curriculum_step / 200_000)
    # 0.15 → 0.07 (world dist 0.30 → 0.14 from stage center)
    dist_center_target = 0.15 - ramp * (0.15 - 0.07)
    # add small jitter so agent doesn't overfit to a single target value
    dist_center_target += np.random.uniform(-0.02, 0.02)
    dist_center_target = float(np.clip(dist_center_target, 0.04, 0.18))
    _curriculum_step += 1
    return np.array([dist_center_target, 0, 0, 0, 0, 0, 0], dtype=np.float32)


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
        progress_scale=2.0,
        progress_clip_min=-0.05,
        progress_clip_max=0.40,
        success_threshold=0.07,
        success_bonus=0.30,
        proximity_scale=0.5,
        reward_clip=1.0,
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
