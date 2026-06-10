#!/usr/bin/env python
from __future__ import annotations

import ctypes
import sys
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import BrawlDeepEnv, EnvConfig
from train.curriculum_config import PHASES, build_phase_spec
from train.llc_stage_common import StageGoalEnv, train_stage_model
from train.train_config import parse_args, TrainConfig


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


def _make_env(cfg: TrainConfig, spec) -> gym.Env:
    config = EnvConfig(
        terminate_on_stock_out=False,
        max_episode_steps=cfg.max_episode_steps,
        yolo_infer_every_n_steps=max(1, cfg.yolo_every),
        action_repeat_steps=1,
        action_repeat_min_steps=1,
        action_repeat_max_steps=1,
        tap_latch_steps=1,
    )
    base = BrawlDeepEnv(config=config)
    env = StageGoalEnv(base, spec)
    if cfg.move_mouse_to_goal:
        env = GoalMouseTrackerWrapper(env)
    return env


def _make_eval_env(cfg: TrainConfig, spec) -> gym.Env:
    eval_spec = replace(
        spec,
        terminate_on_death=False,
        terminate_on_hit_event=False,
    )
    config = EnvConfig(
        terminate_on_stock_out=True,
        max_episode_steps=cfg.max_episode_steps,
        yolo_infer_every_n_steps=max(1, cfg.yolo_every),
        action_repeat_steps=1,
        action_repeat_min_steps=1,
        action_repeat_max_steps=1,
        tap_latch_steps=1,
    )
    base = BrawlDeepEnv(config=config)
    env = StageGoalEnv(base, eval_spec)
    if cfg.move_mouse_to_goal:
        env = GoalMouseTrackerWrapper(env)
    return env


def main() -> None:
    cfg = parse_args()

    spec = build_phase_spec(
        phase=cfg.phase,
        death_penalty=cfg.death_penalty,
        terminate_on_death=cfg.terminate_on_death,
    )

    if cfg.move_mouse_to_goal:
        print("[train_curriculum] Mouse goal tracking enabled.")

    eval_env_builder = None
    eval_env_builder_for_phase = None
    if cfg.eval_every_steps > 0:
        eval_env_builder = lambda: _make_eval_env(cfg, spec)
        eval_env_builder_for_phase = lambda phase: _make_eval_env(
            cfg,
            build_phase_spec(
                phase=phase,
                death_penalty=cfg.death_penalty,
                terminate_on_death=cfg.terminate_on_death,
            ),
        )

    train_stage_model(
        args=cfg,
        make_env=lambda: _make_env(cfg, spec),
        stage_spec=spec,
        make_eval_env=eval_env_builder,
        make_eval_env_for_phase=eval_env_builder_for_phase,
    )


if __name__ == "__main__":
    main()
