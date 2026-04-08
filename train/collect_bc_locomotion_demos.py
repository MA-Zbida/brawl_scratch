#!/usr/bin/env python
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from dataclasses import replace
from pathlib import Path

import keyboard
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import BrawlDeepEnv, EnvConfig, NullInputController
from train.curriculum_config import PHASES, build_phase_spec
from train.llc_stage_common import StageGoalEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect expert demos for BC (all curriculum phases)")
    p.add_argument("--phase", type=str, default="locomotion", choices=list(PHASES))
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--max-episode-steps", type=int, default=1200)
    p.add_argument("--min-goal-duration", type=int, default=120)
    p.add_argument("--max-goal-duration", type=int, default=220)
    p.add_argument("--delay", type=float, default=3.0)
    p.add_argument("--output", type=str, default="")
    p.add_argument("--move-mouse-to-goal", action="store_true", default=True)
    p.add_argument("--no-move-mouse-to-goal", dest="move_mouse_to_goal", action="store_false")
    p.add_argument("--death-penalty", type=float, default=0.0)
    return p.parse_args()


def read_action_from_keyboard(allow_attack: bool) -> np.ndarray:
    movement = 3
    if keyboard.is_pressed("a"):
        movement = 0
    elif keyboard.is_pressed("d"):
        movement = 1
    elif keyboard.is_pressed("s"):
        movement = 2

    jump = 1 if keyboard.is_pressed("space") else 0
    dodge = 1 if keyboard.is_pressed("e") else 0

    attack = 0
    if allow_attack:
        if keyboard.is_pressed("h"):
            attack = 1
        elif keyboard.is_pressed("k"):
            attack = 2
        elif keyboard.is_pressed("j"):
            attack = 3

    # Match env sanitize behavior: no simultaneous dodge+attack.
    if dodge == 1 and attack != 0:
        attack = 0

    return np.array([movement, jump, dodge, attack], dtype=np.int64)


def _build_env(args: argparse.Namespace) -> StageGoalEnv:
    spec = build_phase_spec(
        phase=args.phase,
        death_penalty=float(args.death_penalty),
        terminate_on_death=True,
    )
    # Demo collection should never inject random reset actions.
    spec = replace(spec, reset_perturb_steps=0)
    if spec.resample_goal_on_timer:
        spec = replace(
            spec,
            min_goal_duration=max(1, int(args.min_goal_duration)),
            max_goal_duration=max(int(args.min_goal_duration), int(args.max_goal_duration)),
        )

    config = EnvConfig(
        terminate_on_stock_out=False,
        max_episode_steps=int(args.max_episode_steps),
        yolo_infer_every_n_steps=3,
        action_repeat_steps=1,
        action_repeat_min_steps=1,
        action_repeat_max_steps=1,
        tap_latch_steps=1,
    )
    # Use NullInputController so env.step does not press/release keys.
    # You keep full manual control while we only record state/action labels.
    base = BrawlDeepEnv(config=config, input_controller=NullInputController())
    return StageGoalEnv(base, spec)


def _screen_size_from_env(env: StageGoalEnv) -> tuple[int, int]:
    base = env.unwrapped
    frame = getattr(base, "_last_frame", None)
    if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 2:
        h, w = int(frame.shape[0]), int(frame.shape[1])
        if w > 0 and h > 0:
            return w, h

    user32 = ctypes.windll.user32
    return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))


def _set_mouse_to_goal(env: StageGoalEnv, goal_target: np.ndarray) -> None:
    if goal_target is None or goal_target.shape[0] < 2:
        return

    user32 = ctypes.windll.user32
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass

    w, h = _screen_size_from_env(env)
    x = int(np.clip(float(goal_target[0]), 0.0, 1.0) * max(1, w - 1))
    y = int(np.clip(float(goal_target[1]), 0.0, 1.0) * max(1, h - 1))
    user32.SetCursorPos(x, y)


def _resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output.strip():
        out = Path(args.output)
    else:
        out = Path("train/models") / f"{args.phase}_demos.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    args = parse_args()
    out_path = _resolve_output_path(args)

    env = _build_env(args)
    allow_attack = args.phase in {"damage_static", "damage_dynamic"}

    print("=" * 68)
    print(f"BC DEMO COLLECTION — {args.phase.upper()}")
    print("Input mode: manual only (env key injection disabled)")
    if allow_attack:
        print("Controls: A/D/S, Space, E, H/K/J")
    else:
        print("Controls: A/D/S, Space, E (attacks disabled by phase)")
    print(f"Episodes: {args.episodes} | Output: {out_path}")
    if args.move_mouse_to_goal:
        print("Mouse guidance: enabled (cursor moves to target_x,target_y)")
    print("Press Ctrl+C to stop and save partial data.")
    print("=" * 68)
    print(f"Starting in {args.delay:.1f}s...")
    time.sleep(max(0.0, float(args.delay)))

    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    done_buf: list[bool] = []
    goal_xy_buf: list[np.ndarray] = []

    step_total = 0

    try:
        for ep in range(1, int(args.episodes) + 1):
            obs, info = env.reset()
            done = False
            ep_steps = 0

            goal_target = np.asarray(info.get("goal_target", np.zeros(2, dtype=np.float32)), dtype=np.float32)
            if args.move_mouse_to_goal:
                _set_mouse_to_goal(env, goal_target)

            while not done:
                action = read_action_from_keyboard(allow_attack=allow_attack)

                obs_buf.append(np.asarray(obs, dtype=np.float32).copy())
                act_buf.append(np.asarray(action, dtype=np.int64).copy())
                goal_xy_buf.append(np.asarray(goal_target[:2], dtype=np.float32).copy())

                next_obs, _reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                done_buf.append(done)

                if bool(info.get("goal_new_sampled", False)):
                    goal_target = np.asarray(info.get("goal_target", goal_target), dtype=np.float32)
                    if args.move_mouse_to_goal:
                        _set_mouse_to_goal(env, goal_target)

                obs = next_obs
                ep_steps += 1
                step_total += 1

                if step_total % 100 == 0:
                    print(
                        f"steps={step_total} ep={ep}/{args.episodes} ep_steps={ep_steps} "
                        f"goal_xy=({float(goal_target[0]):.3f}, {float(goal_target[1]):.3f})"
                    )

                if ep_steps >= int(args.max_episode_steps):
                    break

            print(f"Episode {ep}/{args.episodes} collected steps={ep_steps}")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial dataset...")
    finally:
        env.close()

    if len(obs_buf) == 0:
        print("No samples collected; nothing to save.")
        return

    obs_arr = np.stack(obs_buf).astype(np.float32)
    act_arr = np.stack(act_buf).astype(np.int64)
    done_arr = np.asarray(done_buf, dtype=bool)
    goal_xy_arr = np.stack(goal_xy_buf).astype(np.float32)

    np.savez_compressed(
        str(out_path),
        obs=obs_arr,
        actions=act_arr,
        dones=done_arr,
        goal_xy=goal_xy_arr,
        phase=np.asarray([args.phase]),
    )

    print(f"Saved {args.phase} demos")
    print(f"  path   : {out_path}")
    print(f"  obs    : {obs_arr.shape}")
    print(f"  actions: {act_arr.shape}")


if __name__ == "__main__":
    main()
