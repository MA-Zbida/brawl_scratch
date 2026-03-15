#!/usr/bin/env python
"""Record a visual observation debug video while you play manually.

Output video layout:
- Left panel: white background with large black text showing observation values.
- Right panel: gameplay frame with YOLO tracked boxes.

This script does not control your character. You play manually in-game; the script
only reads keyboard state so env-side action-dependent features stay consistent.

Usage:
    python train/debug_observation_overlay.py
    python train/debug_observation_overlay.py --show --target-fps 20 --max-steps 4000
    python train/debug_observation_overlay.py --output debug_obs.mp4 --font-scale 0.75
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

# Ensure project root is on sys.path when invoked as python train/...
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import UI_REGIONS
from env import BrawlDeepEnv, EnvConfig, NullInputController
from feature_extractor.memory.structured_memory import HEIGHT, PLATFORM_Y_MIN
from feature_extractor.memory.state_spec import StateSpec


MOVEMENT_KEYS = {"a": 0, "d": 1, "s": 2}
JUMP_KEY = "space"
DODGE_KEY = "e"
ATTACK_KEYS = {"h": 1, "k": 2, "j": 3}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record YOLO+observation debug video while playing")
    p.add_argument("--output", type=str, default="", help="Output video path (.mp4). Auto-generated if empty")
    p.add_argument("--out-dir", type=str, default="debug", help="Directory for auto-generated output")
    p.add_argument("--target-fps", type=int, default=20, help="Environment step rate")
    p.add_argument("--yolo-every", type=int, default=3, help="Run YOLO every N steps (same as training default=3)")
    p.add_argument("--yolo-conf", type=float, default=0.15, help="YOLO confidence threshold")
    p.add_argument("--max-steps", type=int, default=0, help="Stop after this many steps (0 = unlimited)")
    p.add_argument("--delay", type=float, default=3.0, help="Seconds before capture starts")
    p.add_argument("--panel-width", type=int, default=980, help="Left text panel width in pixels")
    p.add_argument("--font-scale", type=float, default=0.72, help="Observation text font scale")
    p.add_argument("--line-height", type=int, default=30, help="Observation text line height")
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


def make_env(target_fps: int, yolo_every: int, yolo_conf: float) -> BrawlDeepEnv:
    config = EnvConfig(
        terminate_on_stock_out=False,
        ui_regions=dict(UI_REGIONS),
        yolo_conf=float(yolo_conf),
        yolo_infer_every_n_steps=max(1, int(yolo_every)),
        yolo_max_det=5,
        yolo_verbose=False,
        yolo_infer_width=640,
        yolo_infer_height=360,
        use_tracker_layer=True,
        tracker_max_missing=8,
        tracker_iou_threshold=0.1,
        tracker_smooth_alpha=0.6,
        emit_detailed_info=False,
        profile_step_timing=False,
        action_repeat_steps=1,
        tap_latch_steps=1,
        max_episode_steps=0,
    )
    _ = target_fps  # target_fps is pacing; camera runs independently in env.
    return BrawlDeepEnv(config=config, input_controller=NullInputController())


def _xywhn_to_xyxy_px(bbox_xywhn: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    x, y, bw, bh = [float(v) for v in bbox_xywhn[:4]]
    x1 = int((x - bw / 2.0) * w)
    y1 = int((y - bh / 2.0) * h)
    x2 = int((x + bw / 2.0) * w)
    y2 = int((y + bh / 2.0) * h)
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    return x1, y1, x2, y2


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    color_map = {
        "agent": (0, 220, 0),
        "op": (0, 180, 255),
        "op1": (0, 180, 255),
        "op2": (0, 180, 255),
        "weapons": (255, 220, 0),
    }

    for det in detections or []:
        bbox = det.get("bbox", [0.0, 0.0, 0.0, 0.0])
        name = str(det.get("class_name", "obj"))
        conf = float(det.get("confidence", 0.0))
        track_id = det.get("track_id", None)

        x1, y1, x2, y2 = _xywhn_to_xyxy_px(bbox, w, h)
        color = color_map.get(name, (200, 200, 200))

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        if track_id is not None:
            label += f" id:{int(track_id)}"

        text_y = y1 - 8 if y1 > 20 else y1 + 18
        cv2.putText(out, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return out


def draw_frame_debug_lines(frame: np.ndarray, player_y: float, player_foot_y: float, platform_y_min: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Horizontal reference lines in normalized Y space.
    guides = [
        ("player_y", player_y, (0, 255, 255)),         # yellow
        ("player_foot_y", player_foot_y, (0, 255, 0)), # green
        ("platform_y_min", platform_y_min, (0, 80, 255)),  # orange/red
    ]

    for name, y_norm, color in guides:
        y_px = int(max(0, min(h - 1, round(float(y_norm) * (h - 1)))))
        cv2.line(out, (0, y_px), (w - 1, y_px), color, 2, cv2.LINE_AA)
        label = f"{name}: {float(y_norm):+0.4f} (y={y_px})"
        text_y = max(18, y_px - 6)
        cv2.putText(out, label, (12, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(out, label, (12, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)

    return out


def _format_obs_lines(obs: np.ndarray) -> list[str]:
    names = StateSpec.names()
    lines = []
    for i, name in enumerate(names):
        value = float(obs[i]) if i < obs.shape[0] else float("nan")
        lines.append(f"{i:02d} {name}: {value:+0.4f}")
    return lines


def draw_obs_panel(
    panel_width: int,
    panel_height: int,
    obs: np.ndarray,
    step_idx: int,
    action: tuple[int, int, int, int],
    detections: list[dict],
    font_scale: float,
    line_height: int,
) -> np.ndarray:
    panel = np.full((panel_height, panel_width, 3), 255, dtype=np.uint8)
    text_color = (0, 0, 0)

    header = [
        f"step: {step_idx}",
        f"action [mv,j,d,atk]: {list(action)}",
        f"detections: {len(detections or [])}",
    ]

    y = 40
    for line in header:
        cv2.putText(panel, line, (22, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale + 0.05, text_color, 2, cv2.LINE_AA)
        y += line_height

    y += 8

    obs_lines = _format_obs_lines(obs)
    n = len(obs_lines)
    left_count = (n + 1) // 2
    left_lines = obs_lines[:left_count]
    right_lines = obs_lines[left_count:]

    x_left = 22
    x_right = panel_width // 2 + 12

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


def resolve_output_path(output: str, out_dir: str) -> Path:
    if output.strip():
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return d / f"obs_debug_{ts}.mp4"


def main() -> None:
    args = parse_args()
    out_path = resolve_output_path(args.output, args.out_dir)

    env = make_env(
        target_fps=int(args.target_fps),
        yolo_every=int(args.yolo_every),
        yolo_conf=float(args.yolo_conf),
    )

    step_interval = 1.0 / max(1, int(args.target_fps))

    print("=" * 68)
    print("  OBSERVATION DEBUG RECORDER")
    print("=" * 68)
    print(f"  Output video : {out_path}")
    print(f"  Target FPS   : {args.target_fps}")
    print(f"  YOLO every N : {args.yolo_every} step(s)")
    print("  Controls     : Play normally in-game. Press Ctrl+C to stop.")
    if args.show:
        print(f"  Preview      : Enabled ({args.window_name})")
    print("=" * 68)
    print(f"Starting in {args.delay:.1f}s... switch to Brawlhalla.")
    time.sleep(max(0.0, float(args.delay)))

    obs, _ = env.reset()

    writer: cv2.VideoWriter | None = None
    step_idx = 0

    try:
        while True:
            t0 = time.perf_counter()
            action = read_keyboard_action()
            obs, _reward, _terminated, _truncated, info = env.step(action)

            frame = getattr(env, "_last_frame", None)
            if frame is None:
                elapsed = time.perf_counter() - t0
                sleep = step_interval - elapsed
                if sleep > 0:
                    time.sleep(sleep)
                continue

            detections = list(info.get("detections", []))
            frame_boxed = draw_detections(frame, detections)
            player_y = float(StateSpec.get(obs, "player_y"))
            player_foot_y = player_y + (HEIGHT / 2.0)
            frame_boxed = draw_frame_debug_lines(
                frame_boxed,
                player_y=player_y,
                player_foot_y=player_foot_y,
                platform_y_min=float(PLATFORM_Y_MIN),
            )

            panel = draw_obs_panel(
                panel_width=int(args.panel_width),
                panel_height=frame_boxed.shape[0],
                obs=obs,
                step_idx=step_idx,
                action=action,
                detections=detections,
                font_scale=float(args.font_scale),
                line_height=int(args.line_height),
            )

            canvas = np.concatenate([panel, frame_boxed], axis=1)

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
