#!/usr/bin/env python
"""Evaluate a trained Stage 1 (Homing Missile) policy on the live game.

Loads an imagination-trained (or fine-tuned) PPO model, wraps
the real BrawlDeepEnv in StageGoalEnv, and runs episodes while
logging goal error, success rate, and action distribution.

Usage:
    python -m train.eval_stage1 \
        --model train/models/stage1_imagination.zip \
        --episodes 20 \
        --delay 3.0
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO

from hierarchical.goals import GOAL_FEATURE_NAMES
from train.llc_stage_common import StageGoalEnv, StageSpec, make_base_env


# ── target sampler (fixed targets for evaluation) ────────────────────
def _eval_target_sampler(obs: np.ndarray) -> np.ndarray:
    """Sample random x positions across the platform for evaluation."""
    target_x = np.random.uniform(0.34, 0.66)
    return np.array([target_x, 0, 0, 0, 0, 0, 0], dtype=np.float32)


def _make_eval_spec() -> StageSpec:
    return StageSpec(
        stage_id=1,
        name="stage1_homing_missile",
        mask=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        target_sampler=_eval_target_sampler,
        min_goal_duration=20,
        max_goal_duration=40,
        progress_scale=0.0,
        success_threshold=0.03,
        success_bonus=2.0,
        proximity_scale=8.0,
        death_penalty=0.0,
        reward_clip=3.0,
        disable_attack=True,
        disable_dodge=True,
        disable_jump=True,
        reset_perturb_steps=6,
        feature_names=list(GOAL_FEATURE_NAMES),
    )


def make_eval_env(max_episode_steps: int) -> StageGoalEnv:
    base = make_base_env(max_episode_steps=max_episode_steps, terminate_on_stock_out=False)
    return StageGoalEnv(base, _make_eval_spec())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Stage 1 policy on live game")
    p.add_argument("--model", type=str, required=True,
                   help="Path to trained PPO .zip (e.g. train/models/stage1_imagination.zip)")
    p.add_argument("--episodes", type=int, default=10,
                   help="Number of evaluation episodes to run")
    p.add_argument("--max-episode-steps", type=int, default=600,
                   help="Max steps per episode")
    p.add_argument("--delay", type=float, default=3.0,
                   help="Seconds to wait before starting (switch to game window)")
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="Use deterministic (greedy) actions")
    p.add_argument("--stochastic", action="store_true",
                   help="Use stochastic (sampled) actions")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    deterministic = not args.stochastic

    print(f"Loading model from {args.model} ...")
    model = PPO.load(args.model)

    print(f"Creating live-game environment ...")
    env = make_eval_env(max_episode_steps=args.max_episode_steps)

    print(f"\n{'='*60}")
    print(f"  Stage 1 Evaluation — {'deterministic' if deterministic else 'stochastic'}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Max steps/ep: {args.max_episode_steps}")
    print(f"{'='*60}")
    print(f"\nSwitch to the game window. Starting in {args.delay:.0f}s ...")
    time.sleep(args.delay)

    # ── episode loop ─────────────────────────────────────────────────
    all_errors: list[float] = []
    all_successes: list[float] = []
    all_returns: list[float] = []
    all_lengths: list[int] = []
    action_counts: Counter = Counter()

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_errors: list[float] = []
        ep_successes: list[float] = []
        step = 0

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            action_counts[int(action[0])] += 1  # movement distribution
            ep_reward += reward
            ep_errors.append(info.get("goal_error", 0.0))
            ep_successes.append(info.get("goal_success", 0.0))
            step += 1

            if terminated or truncated:
                break

        mean_error = float(np.mean(ep_errors)) if ep_errors else 0.0
        success_rate = float(np.mean(ep_successes)) if ep_successes else 0.0

        all_errors.append(mean_error)
        all_successes.append(success_rate)
        all_returns.append(ep_reward)
        all_lengths.append(step)

        print(
            f"  Ep {ep:3d}/{args.episodes} | "
            f"steps={step:4d} | "
            f"return={ep_reward:+8.1f} | "
            f"mean_error={mean_error:.4f} | "
            f"success_rate={success_rate:.1%}"
        )

    # ── summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY over {args.episodes} episodes")
    print(f"{'='*60}")
    print(f"  Mean return:       {np.mean(all_returns):+.1f}  (std {np.std(all_returns):.1f})")
    print(f"  Mean goal error:   {np.mean(all_errors):.4f}  (std {np.std(all_errors):.4f})")
    print(f"  Mean success rate: {np.mean(all_successes):.1%}")
    print(f"  Mean ep length:    {np.mean(all_lengths):.0f}")
    print()

    total_actions = sum(action_counts.values())
    movement_names = {0: "left", 1: "right", 2: "down", 3: "idle"}
    print("  Action distribution (movement):")
    for idx in sorted(action_counts.keys()):
        pct = action_counts[idx] / max(1, total_actions) * 100
        print(f"    {movement_names.get(idx, str(idx)):>5s}: {action_counts[idx]:6d}  ({pct:.1f}%)")

    # Release keys
    try:
        env.unwrapped.input_controller.reset()
    except Exception:
        pass
    env.close()


if __name__ == "__main__":
    main()
