#!/usr/bin/env python
from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import BrawlDeepEnv, EnvConfig
from train.curriculum_config import PHASES, build_phase_spec
from train.llc_stage_common import StageGoalEnv, train_stage_model


def _has_cli_flag(flag: str) -> bool:
    for token in sys.argv[1:]:
        if token == flag or token.startswith(f"{flag}="):
            return True
    return False


class GoalMouseTrackerWrapper(gym.Wrapper):
    """Move mouse cursor to target_x,target_y whenever goal is sampled."""

    def __init__(self, env: StageGoalEnv):
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

    def _set_cursor_to_goal(self, goal_target) -> None:
        if self._user32 is None:
            return
        if goal_target is None:
            return
        goal_target = np.asarray(goal_target, dtype=np.float32).reshape(-1)
        if goal_target.shape[0] < 2:
            return

        w, h = self._screen_size()
        x = int(np.clip(float(goal_target[0]), 0.0, 1.0) * max(1, w - 1))
        y = int(np.clip(float(goal_target[1]), 0.0, 1.0) * max(1, h - 1))
        self._user32.SetCursorPos(x, y)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._set_cursor_to_goal(info.get("goal_target"))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if bool(info.get("goal_new_sampled", False)):
            self._set_cursor_to_goal(info.get("goal_target"))
        return obs, reward, terminated, truncated, info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO curriculum phase")
    p.add_argument("--phase", type=str, required=True, choices=list(PHASES))
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--max-episode-steps", type=int, default=1200)
    p.add_argument("--save-dir", type=str, default="train/models")
    p.add_argument("--model-name", type=str, default="")
    p.add_argument("--resume", type=str, default=None)

    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delay", type=float, default=3.0)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.15)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)

    p.add_argument("--plot-every", type=int, default=2500)
    p.add_argument("--log-csv", action="store_true", help="Write step/episode CSV logs")
    p.add_argument("--diag-report-every", type=int, default=0, help="Diagnostic print period in steps (0 disables)")
    p.add_argument("--moving-avg", type=int, default=300)

    p.add_argument("--death-penalty", type=float, default=1.0)
    p.add_argument("--no-terminate-on-death", action="store_true")
    p.add_argument("--move-mouse-to-goal", action="store_true", default=False)
    p.add_argument("--yolo-every", type=int, default=2, help="Run YOLO every N env steps (1 = max YOLO authority)")
    p.add_argument("--yolo-blend-alpha", type=float, default=0.90, help="YOLO fusion weight in memory update [0,1]")
    p.add_argument("--tracker-max-missing", type=int, default=2, help="Tracker persistence in missed frames")
    p.add_argument("--tracker-smooth-alpha", type=float, default=0.85, help="Tracker smoothing alpha [0,1], higher = closer to YOLO")

    return p.parse_args()


def make_env(
    max_episode_steps: int,
    spec,
    move_mouse_to_goal: bool = False,
    yolo_every: int = 2,
    yolo_blend_alpha: float = 0.90,
    tracker_max_missing: int = 4,
    tracker_smooth_alpha: float = 0.75,
) -> gym.Env:
    config = EnvConfig(
        terminate_on_stock_out=False,
        max_episode_steps=max_episode_steps,
        yolo_infer_every_n_steps=max(1, int(yolo_every)),
        yolo_obs_blend_alpha=float(np.clip(yolo_blend_alpha, 0.0, 1.0)),
        tracker_max_missing=max(1, int(tracker_max_missing)),
        tracker_smooth_alpha=float(np.clip(tracker_smooth_alpha, 0.0, 1.0)),
        action_repeat_steps=1,
        action_repeat_min_steps=1,
        action_repeat_max_steps=1,
        tap_latch_steps=1,
    )
    base = BrawlDeepEnv(config=config)
    env = StageGoalEnv(base, spec)
    if move_mouse_to_goal:
        env = GoalMouseTrackerWrapper(env)
    return env


def main() -> None:
    args = parse_args()

    if not args.model_name:
        args.model_name = f"llc_{args.phase}"

    spec = build_phase_spec(
        phase=args.phase,
        death_penalty=args.death_penalty,
        terminate_on_death=not args.no_terminate_on_death,
    )

    if args.phase.startswith("locomotion"):
        if not _has_cli_flag("--learning-rate"):
            args.learning_rate = 3e-4
        if not _has_cli_flag("--n-steps"):
            args.n_steps = 1024
        if not _has_cli_flag("--clip-range"):
            args.clip_range = 0.2
        if not _has_cli_flag("--ent-coef"):
            args.ent_coef = 0.03

    if args.move_mouse_to_goal:
        print("[train_curriculum] Mouse goal tracking enabled.")

    train_stage_model(
        args=args,
        make_env=lambda: make_env(
            args.max_episode_steps,
            spec,
            move_mouse_to_goal=args.move_mouse_to_goal,
            yolo_every=args.yolo_every,
            yolo_blend_alpha=args.yolo_blend_alpha,
            tracker_max_missing=args.tracker_max_missing,
            tracker_smooth_alpha=args.tracker_smooth_alpha,
        ),
        stage_spec=spec,
    )


if __name__ == "__main__":
    main()
