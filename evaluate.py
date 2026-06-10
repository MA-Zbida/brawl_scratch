#!/usr/bin/env python
from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path
from typing import Any, Sequence, cast

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env import BrawlDeepEnv, EnvConfig
from train.curriculum_config import PHASES, build_phase_spec
from train.llc_stage_common import StageGoalEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a curriculum phase policy over stock-out matches")
    p.add_argument("--model", type=str, required=True, help="Path to PPO .zip checkpoint")
    p.add_argument("--phase", type=str, required=True, choices=list(PHASES), help="Curriculum phase to evaluate")
    p.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    p.add_argument("--max-episode-steps", type=int, default=0, help="Hard cap (0 = no cap)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stochastic", action="store_true", help="Use stochastic actions instead of deterministic")

    p.add_argument("--yolo-every", type=int, default=1)

    p.add_argument("--death-penalty", type=float, default=1.0, help="Only used when --phase is set")
    return p.parse_args()


class GoalMouseTrackerWrapper(gym.Wrapper):
    """Move mouse cursor to target_x,target_y whenever goal is sampled."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._user32 = None
        try:
            self._user32 = ctypes.windll.user32
            try:
                self._user32.SetProcessDPIAware()
            except Exception:
                pass
        except Exception:
            self._user32 = None

    def _screen_size(self) -> tuple[int, int]:
        base = self.unwrapped
        frame = getattr(base, "_last_frame", None)
        if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 2:
            h, w = int(frame.shape[0]), int(frame.shape[1])
            if w > 0 and h > 0:
                return w, h
        if self._user32 is None:
            return 1920, 1080
        return int(self._user32.GetSystemMetrics(0)), int(self._user32.GetSystemMetrics(1))

    def _goal_xy_from_info(self, info) -> np.ndarray | None:
        goal_target = None if info is None else info.get("goal_target")
        if goal_target is None:
            return None
        goal_target = np.asarray(goal_target, dtype=np.float32).reshape(-1)
        if goal_target.shape[0] < 2:
            return None
        names = list(info.get("stage_feature_names", [])) if isinstance(info, dict) else []
        mask = np.asarray(info.get("goal_mask", np.zeros_like(goal_target)), dtype=np.float32).reshape(-1) if isinstance(info, dict) else np.zeros_like(goal_target)
        if "player_x" in names and "player_y" in names:
            x_idx = names.index("player_x")
            y_idx = names.index("player_y")
            if x_idx < goal_target.shape[0] and y_idx < goal_target.shape[0]:
                if x_idx >= mask.shape[0] or y_idx >= mask.shape[0] or mask[x_idx] > 0.0 or mask[y_idx] > 0.0:
                    return np.asarray([goal_target[x_idx], goal_target[y_idx]], dtype=np.float32)
                return None
        return goal_target[:2].astype(np.float32)

    def _set_cursor_to_goal(self, info) -> None:
        if self._user32 is None:
            return
        goal_xy = self._goal_xy_from_info(info)
        if goal_xy is None:
            return

        w, h = self._screen_size()
        x = int(np.clip(float(goal_xy[0]), 0.0, 1.0) * max(1, w - 1))
        y = int(np.clip(float(goal_xy[1]), 0.0, 1.0) * max(1, h - 1))
        self._user32.SetCursorPos(x, y)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._set_cursor_to_goal(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if bool(info.get("goal_new_sampled", False)):
            self._set_cursor_to_goal(info)
        return obs, reward, terminated, truncated, info


def make_env(args: argparse.Namespace):
    config = EnvConfig(
        terminate_on_stock_out=True,
        max_episode_steps=max(0, int(args.max_episode_steps)),
        yolo_infer_every_n_steps=max(1, int(args.yolo_every)),
        action_repeat_steps=1,
        action_repeat_min_steps=1,
        action_repeat_max_steps=1,
        tap_latch_steps=1,
    )
    base_env = BrawlDeepEnv(config=config)
    stage_spec = build_phase_spec(
        phase=args.phase,
        death_penalty=float(args.death_penalty),
        terminate_on_death=False,
    )
    # During evaluation, terminate only when a stock-out happens (or max-steps if configured).
    stage_spec.terminate_on_death = False
    stage_spec.terminate_on_hit_event = False
    env: gym.Env = StageGoalEnv(base_env, stage_spec)

    if str(args.phase) in ("recovery_mastery", "movement_fluency"):
        env = GoalMouseTrackerWrapper(env)
        print("[evaluate] Movement phase detected: forcing mouse goal tracking ON.")

    return env


def _resolve_outcome(self_stocks: float, op_stocks: float, truncated: bool) -> str:
    if op_stocks <= 0.0 and self_stocks > 0.0:
        return "WIN"
    if self_stocks <= 0.0 and op_stocks > 0.0:
        return "LOSS"
    if self_stocks <= 0.0 and op_stocks <= 0.0:
        return "DRAW"
    if truncated:
        return "TRUNC"
    return "UNRESOLVED"


def _to_env_action(action: Any) -> Sequence[int]:
    arr = np.asarray(action, dtype=np.int64).reshape(-1)
    if arr.shape[0] < 4:
        raise ValueError(f"Predicted action must have 4 components, got shape {arr.shape}")
    return [int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])]


def _get_base_env(env) -> BrawlDeepEnv:
    base = env.unwrapped
    if not isinstance(base, BrawlDeepEnv):
        raise TypeError(f"Expected unwrapped env to be BrawlDeepEnv, got {type(base)}")
    return base


def main() -> None:
    args = parse_args()
    args.phase = str(args.phase).strip().lower()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    env = make_env(args)
    deterministic = not bool(args.stochastic)
    model = PPO.load(str(model_path), device=args.device)

    rewards: list[float] = []
    lengths: list[int] = []
    outcomes: list[str] = []
    self_stocks_list: list[float] = []
    op_stocks_list: list[float] = []
    op_dmg_totals: list[float] = []
    self_dmg_totals: list[float] = []
    goal_error_means: list[float] = []
    goal_success_ratios: list[float] = []

    print("=" * 84)
    print(f"Evaluating: {model_path}")
    print(
        f"Episodes: {args.episodes} | Phase: {args.phase} | Deterministic: {deterministic}"
    )
    print("Evaluation mode: phase-conditioned policy with stock-out match outcomes.")
    print("=" * 84)

    try:
        for ep in range(1, int(args.episodes) + 1):
            obs, _ = env.reset(seed=int(args.seed + ep))
            done = False
            truncated = False

            ep_reward = 0.0
            ep_len = 0
            ep_op_dmg = 0.0
            ep_self_dmg = 0.0
            ep_goal_error_sum = 0.0
            ep_goal_success_sum = 0.0
            ep_goal_steps = 0

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=deterministic)
                if isinstance(action, np.ndarray) and action.ndim > 1:
                    action = action[0]
                env_action = _to_env_action(action)
                obs, reward, done, truncated, info = cast(Any, env).step(env_action)
                ep_reward += float(reward)
                ep_len += 1
                ep_op_dmg += float(info.get("op_delta_damage", 0.0))
                ep_self_dmg += float(info.get("self_delta_damage", 0.0))
                ep_goal_error_sum += float(info.get("goal_error", 0.0))
                ep_goal_success_sum += float(info.get("goal_success", 0.0))
                ep_goal_steps += 1

            base = _get_base_env(env)
            self_stocks = float(base.memory.self_stocks_left)
            op_stocks = float(base.memory.op_stocks_left)
            outcome = _resolve_outcome(self_stocks, op_stocks, bool(truncated))

            rewards.append(ep_reward)
            lengths.append(ep_len)
            outcomes.append(outcome)
            self_stocks_list.append(self_stocks)
            op_stocks_list.append(op_stocks)
            op_dmg_totals.append(ep_op_dmg)
            self_dmg_totals.append(ep_self_dmg)

            goal_err_mean = float(ep_goal_error_sum / max(1, ep_goal_steps))
            goal_success_ratio = float(ep_goal_success_sum / max(1, ep_goal_steps))
            goal_error_means.append(goal_err_mean)
            goal_success_ratios.append(goal_success_ratio)

            print(
                f"Ep {ep:02d} | {outcome:10s} | reward={ep_reward:+9.3f} | steps={ep_len:5d} | "
                f"stocks(self/op)={self_stocks:.1f}/{op_stocks:.1f} | "
                f"dmg(self/op)={ep_self_dmg:.3f}/{ep_op_dmg:.3f} | "
                f"goal_err={goal_err_mean:.4f} | goal_succ={goal_success_ratio:.3f}"
            )
    finally:
        env.close()

    rewards_np = np.asarray(rewards, dtype=np.float32)
    lengths_np = np.asarray(lengths, dtype=np.float32)
    self_stocks_np = np.asarray(self_stocks_list, dtype=np.float32)
    op_stocks_np = np.asarray(op_stocks_list, dtype=np.float32)
    op_dmg_np = np.asarray(op_dmg_totals, dtype=np.float32)
    self_dmg_np = np.asarray(self_dmg_totals, dtype=np.float32)
    goal_err_np = np.asarray(goal_error_means, dtype=np.float32)
    goal_succ_np = np.asarray(goal_success_ratios, dtype=np.float32)

    wins = sum(1 for x in outcomes if x == "WIN")
    losses = sum(1 for x in outcomes if x == "LOSS")
    draws = sum(1 for x in outcomes if x == "DRAW")
    truncs = sum(1 for x in outcomes if x == "TRUNC")

    print("\n" + "=" * 84)
    print("Evaluation summary")
    print("=" * 84)
    print(f"Episodes           : {len(outcomes)}")
    print(f"W/L/D/TRUNC        : {wins}/{losses}/{draws}/{truncs}")
    print(f"Win rate           : {wins / max(1, len(outcomes)):.3f}")
    print(f"Avg reward         : {float(rewards_np.mean()):+.3f} +- {float(rewards_np.std()):.3f}")
    print(f"Avg episode length : {float(lengths_np.mean()):.1f} steps")
    print(f"Avg stocks self/op : {float(self_stocks_np.mean()):.2f}/{float(op_stocks_np.mean()):.2f}")
    print(f"Avg dmg self/op    : {float(self_dmg_np.mean()):.3f}/{float(op_dmg_np.mean()):.3f}")
    print(f"Avg stock diff     : {float((self_stocks_np - op_stocks_np).mean()):+.3f}")
    print(f"Avg goal error     : {float(goal_err_np.mean()):.4f}")
    print(f"Avg goal success   : {float(goal_succ_np.mean()):.3f}")


if __name__ == "__main__":
    main()
