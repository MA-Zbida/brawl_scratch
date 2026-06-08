#!/usr/bin/env python
"""Record gameplay + phase success diagnostics while you play manually.

Output video layout:
- Left panel: observations, success YES/NO, and active success elements.
- Right panel: raw gameplay frame (no YOLO boxes).

This script does not control your character. You play manually in-game; the script
only reads keyboard state so action-dependent observation features stay consistent.

Usage:
    python tools/debug_observation_overlay.py --phase movement_fluency --show --max-steps 1000
    python tools/debug_observation_overlay.py --phase combat_execution --show --max-steps 4000
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import keyboard
import numpy as np

# Ensure project root is on sys.path when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import UI_REGIONS
from env import BrawlDeepEnv, EnvConfig, NullInputController
from feature_extractor.memory.state_spec import StateSpec
from train.curriculum_config import PHASES, build_phase_spec
from train.llc_stage_common import StageGoalEnv, StageSpec


MOVEMENT_KEYS = {"a": 0, "d": 1, "s": 2}
JUMP_KEY = "space"
DODGE_KEY = "e"
# keyboard scan codes for numpad 4/5/6.
ATTACK_KEYS = {75: 1, 77: 2, 76: 3}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record phase-aware observation debug video while playing")
    p.add_argument("--phase", type=str, required=True, choices=list(PHASES), help="Phase used for success logic")
    p.add_argument("--output", type=str, default="", help="Output video path (.mp4). Auto-generated if empty")
    p.add_argument("--out-dir", type=str, default="debug", help="Directory for auto-generated output")
    p.add_argument("--target-fps", type=int, default=20, help="Environment step rate")
    p.add_argument("--yolo-every", type=int, default=3, help="Run YOLO every N steps")
    p.add_argument("--yolo-conf", type=float, default=0.15, help="YOLO confidence threshold")
    p.add_argument("--max-steps", type=int, default=0, help="Stop after this many steps (0 = unlimited)")
    p.add_argument("--delay", type=float, default=3.0, help="Seconds before capture starts")
    p.add_argument("--panel-width", type=int, default=1100, help="Left text panel width in pixels")
    p.add_argument("--font-scale", type=float, default=0.62, help="Observation text font scale")
    p.add_argument("--line-height", type=int, default=24, help="Observation text line height")
    p.add_argument("--death-penalty", type=float, default=1.0, help="Phase spec death penalty")
    p.add_argument("--no-terminate-on-death", action="store_true")
    p.add_argument("--show", action="store_true", help="Show live preview window while recording")
    p.add_argument("--window-name", type=str, default="obs-debug", help="Preview window title")
    return p.parse_args()


def read_keyboard_action() -> tuple[int, int, int, int]:
    movement = 3
    for key, idx in MOVEMENT_KEYS.items():
        if keyboard.is_pressed(key):
            movement = idx
            break

    jump = 1 if keyboard.is_pressed(JUMP_KEY) else 0
    dodge = 1 if keyboard.is_pressed(DODGE_KEY) else 0

    attack = 0
    for key, idx in ATTACK_KEYS.items():
        if keyboard.is_pressed(key):
            attack = idx
            break

    if dodge == 1 and attack != 0:
        attack = 0

    return movement, jump, dodge, attack


def make_env(args: argparse.Namespace) -> tuple[StageGoalEnv, StageSpec]:
    spec = build_phase_spec(
        phase=args.phase,
        death_penalty=float(args.death_penalty),
        terminate_on_death=not bool(args.no_terminate_on_death),
    )
    config = EnvConfig(
        terminate_on_stock_out=False,
        ui_regions=dict(UI_REGIONS),
        yolo_conf=float(args.yolo_conf),
        yolo_infer_every_n_steps=max(1, int(args.yolo_every)),
        yolo_max_det=5,
        yolo_verbose=False,
        yolo_infer_width=640,
        yolo_infer_height=360,
        emit_detailed_info=False,
        profile_step_timing=False,
        action_repeat_steps=1,
        tap_latch_steps=1,
        max_episode_steps=0,
    )
    base = BrawlDeepEnv(config=config, input_controller=NullInputController())
    return StageGoalEnv(base, spec), spec


def _format_base_obs_lines(obs: np.ndarray) -> list[str]:
    names = StateSpec.names()
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    base_dim = int(StateSpec.dim())
    base = arr[:base_dim]
    lines: list[str] = []
    for i, name in enumerate(names):
        value = float(base[i]) if i < base.shape[0] else float("nan")
        lines.append(f"{i:02d} {name}: {value:+0.4f}")
    return lines


def _format_success_lines(info: dict[str, Any], spec: StageSpec) -> tuple[list[str], bool]:
    goal_success = float(info.get("goal_success", 0.0)) > 0.5
    goal_error = float(info.get("goal_error", 0.0))

    names = list(info.get("stage_feature_names", []))
    raw_goal_feats = np.asarray(info.get("raw_goal_feats", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    goal_target = np.asarray(info.get("goal_target", np.zeros_like(raw_goal_feats)), dtype=np.float32).reshape(-1)
    goal_mask = np.asarray(info.get("goal_mask", np.zeros_like(raw_goal_feats)), dtype=np.float32).reshape(-1)

    lines: list[str] = [
        f"phase: {str(info.get('stage_name', spec.name))}",
        f"SUCCESS: {'YES' if goal_success else 'NO'}",
        f"goal_error: {goal_error:.4f}",
        f"success_threshold: {float(spec.success_threshold):.4f}",
        f"goal_new_sampled: {int(bool(info.get('goal_new_sampled', False)))}",
        "active success elements:",
    ]

    active_idx = [i for i, w in enumerate(goal_mask.tolist()) if float(w) > 0.0]
    if not active_idx:
        lines.append("  (none)")
        return lines, goal_success

    for idx in active_idx:
        name = names[idx] if idx < len(names) else f"f{idx}"
        cur = float(raw_goal_feats[idx]) if idx < raw_goal_feats.shape[0] else 0.0
        tgt = float(goal_target[idx]) if idx < goal_target.shape[0] else 0.0
        w = float(goal_mask[idx]) if idx < goal_mask.shape[0] else 0.0
        lines.append(f"  {name:22s} cur={cur:0.3f} tgt={tgt:0.3f} w={w:0.2f} |d|={abs(cur - tgt):0.3f}")

    return lines, goal_success


def draw_obs_panel(
    panel_width: int,
    panel_height: int,
    obs: np.ndarray,
    step_idx: int,
    reward: float,
    action: tuple[int, int, int, int],
    info: dict[str, Any],
    spec: StageSpec,
    font_scale: float,
    line_height: int,
) -> np.ndarray:
    panel = np.full((panel_height, panel_width, 3), 255, dtype=np.uint8)
    text_color = (0, 0, 0)

    header = [
        f"step: {step_idx}",
        f"reward: {float(reward):+.4f}",
        f"action [mv,j,d,atk]: {list(action)}",
    ]

    y = 32
    for line in header:
        cv2.putText(panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale + 0.06, text_color, 2, cv2.LINE_AA)
        y += line_height

    success_lines, success = _format_success_lines(info, spec)
    success_color = (0, 140, 0) if success else (0, 0, 180)
    y += 6
    for i, line in enumerate(success_lines):
        if y >= panel_height - 10:
            break
        color = success_color if i == 1 else text_color
        cv2.putText(panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
        y += line_height

    y += 8
    cv2.line(panel, (12, y), (panel_width - 12, y), (0, 0, 0), 1, cv2.LINE_AA)
    y += line_height

    obs_lines = _format_base_obs_lines(obs)
    n = len(obs_lines)
    left_count = (n + 1) // 2
    left_lines = obs_lines[:left_count]
    right_lines = obs_lines[left_count:]

    x_left = 18
    x_right = panel_width // 2 + 8

    for i, line in enumerate(left_lines):
        yy = y + i * line_height
        if yy >= panel_height - 10:
            break
        cv2.putText(panel, line, (x_left, yy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)

    for i, line in enumerate(right_lines):
        yy = y + i * line_height
        if yy >= panel_height - 10:
            break
        cv2.putText(panel, line, (x_right, yy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)

    return panel


def resolve_output_path(output: str, out_dir: str, phase: str) -> Path:
    if output.strip():
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return d / f"obs_debug_{phase}_{ts}.mp4"


def main() -> None:
    args = parse_args()
    out_path = resolve_output_path(args.output, args.out_dir, args.phase)

    env, spec = make_env(args)
    step_interval = 1.0 / max(1, int(args.target_fps))

    print("=" * 68)
    print("  OBSERVATION + SUCCESS DEBUG RECORDER")
    print("=" * 68)
    print(f"  Phase        : {args.phase}")
    print(f"  Output video : {out_path}")
    print(f"  Target FPS   : {args.target_fps}")
    print(f"  YOLO every N : {args.yolo_every} step(s)")
    print("  Overlay      : observations + SUCCESS signal + active success elements")
    print("  Right panel  : raw gameplay frame (no YOLO boxes)")
    print("  Controls     : Play normally in-game. Press Ctrl+C to stop.")
    if args.show:
        print(f"  Preview      : Enabled ({args.window_name})")
    print("=" * 68)
    print(f"Starting in {args.delay:.1f}s... switch to Brawlhalla.")
    time.sleep(max(0.0, float(args.delay)))

    obs, info = env.reset()

    writer: cv2.VideoWriter | None = None
    step_idx = 0

    try:
        while True:
            t0 = time.perf_counter()
            action = read_keyboard_action()
            obs_step, reward, terminated, truncated, info_step = env.step(action)

            obs_for_draw = obs_step
            info_for_draw = info_step

            if terminated or truncated:
                obs, info = env.reset()
            else:
                obs = obs_step
                info = info_step

            frame = getattr(env.unwrapped, "_last_frame", None)
            if frame is None:
                elapsed = time.perf_counter() - t0
                sleep = step_interval - elapsed
                if sleep > 0:
                    time.sleep(sleep)
                continue

            panel = draw_obs_panel(
                panel_width=int(args.panel_width),
                panel_height=frame.shape[0],
                obs=obs_for_draw,
                step_idx=step_idx,
                reward=float(reward),
                action=action,
                info=info_for_draw,
                spec=spec,
                font_scale=float(args.font_scale),
                line_height=int(args.line_height),
            )

            canvas = np.concatenate([panel, frame], axis=1)

            if writer is None:
                h, w = canvas.shape[:2]
                fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
                if callable(fourcc_fn):
                    fourcc_value: Any = fourcc_fn(*"mp4v")
                    fourcc = int(fourcc_value)
                else:
                    fourcc = int(cv2.VideoWriter.fourcc(*"mp4v"))
                writer = cv2.VideoWriter(str(out_path), fourcc, float(args.target_fps), (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer: {out_path}")

            writer.write(canvas)

            if args.show:
                cv2.imshow(args.window_name, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    print("Stopped by keypress.")
                    break

            step_idx += 1
            if step_idx % 200 == 0:
                print(f"  recorded steps: {step_idx}")

            if args.max_steps > 0 and step_idx >= int(args.max_steps):
                print(f"Reached max_steps={args.max_steps}, stopping.")
                break

            elapsed = time.perf_counter() - t0
            sleep = step_interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.")
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        env.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
