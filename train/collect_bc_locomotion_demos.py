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

from env import BrawlDeepEnv, EnvConfig, NullInputController, PyDirectInputController
from train.curriculum_config import PHASES, build_phase_spec
from train.llc_stage_common import StageGoalEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect expert demos for BC (all curriculum phases)")
    p.add_argument("--phase", type=str, default="weapon_control", choices=list(PHASES))
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--min-goal-duration", type=int, default=120)
    p.add_argument("--max-goal-duration", type=int, default=220)
    p.add_argument("--delay", type=float, default=3.0)
    p.add_argument("--output", type=str, default="")
    p.add_argument("--move-mouse-to-goal", action="store_true", default=True)
    p.add_argument("--no-move-mouse-to-goal", dest="move_mouse_to_goal", action="store_false")
    p.add_argument("--death-penalty", type=float, default=0.0)
    p.add_argument("--force-drop-on-episode-end", action="store_true", default=True)
    p.add_argument("--no-force-drop-on-episode-end", dest="force_drop_on_episode_end", action="store_false")
    p.add_argument("--end-episode-on-first-hit", dest="end_episode_on_first_hit", action="store_true")
    p.add_argument("--no-end-episode-on-first-hit", dest="end_episode_on_first_hit", action="store_false")
    p.set_defaults(end_episode_on_first_hit=None)
    p.add_argument("--hit-damage-threshold", type=float, default=1e-3)
    return p.parse_args()


def read_action_from_keyboard(allowed_attack_values: set[int]) -> np.ndarray:
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
    # keyboard uses scan codes: 75/76/77 correspond to numpad 4/5/6.
    if 1 in allowed_attack_values and keyboard.is_pressed(75):
        attack = 1
    elif 2 in allowed_attack_values and keyboard.is_pressed(77):
        attack = 2
    elif 3 in allowed_attack_values and keyboard.is_pressed(76):
        attack = 3

    # Match env sanitize behavior: no simultaneous dodge+attack.
    if dodge == 1 and attack != 0:
        attack = 0

    return np.array([movement, jump, dodge, attack], dtype=np.int64)


_DAMAGE_PHASES = frozenset({"damage_static_fist", "damage_static_weapon", "damage_dynamic", "damage_static"})

_DEFAULT_MAX_STEPS = {
    # Locomotion / weapon: short episodes suffice.
    "default": 100,
    # Damage phases need much longer episodes to match PPO training (1200).
    "damage": 600,
}


def _build_env(args: argparse.Namespace) -> StageGoalEnv:
    spec = build_phase_spec(
        phase=args.phase,
        death_penalty=float(args.death_penalty),
        terminate_on_death=True,
    )
    # Demo collection should never inject random reset actions.
    spec = replace(spec, reset_perturb_steps=0)

    is_damage = args.phase in _DAMAGE_PHASES

    # For damage phases the PPO spec has resample_goal_on_timer=False, but
    # for BC collection we NEED fresh goals so the human always has a target.
    # Override to timer-based resampling with the CLI durations.
    if is_damage and not spec.resample_goal_on_timer:
        spec = replace(
            spec,
            resample_goal_on_timer=True,
            min_goal_duration=max(1, int(args.min_goal_duration)),
            max_goal_duration=max(int(args.min_goal_duration), int(args.max_goal_duration)),
        )
    elif spec.resample_goal_on_timer:
        spec = replace(
            spec,
            min_goal_duration=max(1, int(args.min_goal_duration)),
            max_goal_duration=max(int(args.min_goal_duration), int(args.max_goal_duration)),
        )

    # Auto-pick max_episode_steps when the user didn't explicitly set it.
    max_ep_steps = int(args.max_episode_steps)
    if max_ep_steps == _DEFAULT_MAX_STEPS["default"] and is_damage:
        max_ep_steps = _DEFAULT_MAX_STEPS["damage"]

    config = EnvConfig(
        terminate_on_stock_out=False,
        max_episode_steps=max_ep_steps,
        yolo_infer_every_n_steps=1,
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


def _force_drop_weapon_num5_key(drop_controller: PyDirectInputController | None = None) -> bool:
    """Press numpad 5 once to force drop between episodes.

    Prefer DirectInput (same path used by env attacks), then keyboard fallback.
    """
    if drop_controller is not None:
        try:
            # Retry a couple of times to survive occasional missed synthetic inputs.
            for _ in range(3):
                drop_controller.tap({"num5"})
                time.sleep(0.02)
            return True
        except Exception:
            pass

    # keyboard scan code fallback for numpad 5.
    try:
        keyboard.press(76)
        time.sleep(0.03)
        keyboard.release(76)
        return True
    except Exception:
        pass

    for key_name in ("num_5", "num 5", "num5"):
        try:
            keyboard.press_and_release(key_name)
            return True
        except Exception:
            continue

    return False


def _phase_uses_player_xy_goal(spec) -> bool:
    feature_names = list(spec.feature_names or [])
    if not feature_names:
        return False

    idx = {name: i for i, name in enumerate(feature_names)}
    x_idx = idx.get("player_x")
    y_idx = idx.get("player_y")
    if x_idx is None or y_idx is None:
        return False

    mask = np.asarray(spec.mask, dtype=np.float32).reshape(-1)
    return bool(mask[x_idx] > 0.0 or mask[y_idx] > 0.0)


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
    hit_damage_threshold = float(max(0.0, args.hit_damage_threshold))
    phase_lower = str(args.phase).strip().lower()
    is_damage = phase_lower in _DAMAGE_PHASES
    phase_requires_weapon = phase_lower in ("damage_static_weapon", "damage_dynamic")

    # Default: never end on first hit — BC episodes should mirror PPO episodes,
    # which continue after hits until goal progress / max steps.
    if args.end_episode_on_first_hit is None:
        end_episode_on_first_hit = False
    else:
        end_episode_on_first_hit = bool(args.end_episode_on_first_hit)

    # Auto-disable weapon drop for phases that require the agent to hold a weapon.
    force_drop = bool(args.force_drop_on_episode_end) and not phase_requires_weapon

    env = _build_env(args)
    drop_controller: PyDirectInputController | None = None
    drop_warning_printed = False
    if force_drop:
        try:
            drop_controller = PyDirectInputController()
        except Exception as exc:
            print(f"Force-drop warning: PyDirectInput unavailable ({exc}); using keyboard fallback.")

    spec = env.stage_spec
    if spec.disable_attack:
        allowed_attack_values = {0}
    elif spec.allowed_attack_actions is None:
        allowed_attack_values = {0, 1, 2, 3}
    else:
        allowed_attack_values = {int(v) for v in spec.allowed_attack_actions if 0 <= int(v) <= 3}
        if not allowed_attack_values:
            allowed_attack_values = {0}
        allowed_attack_values.add(0)

    mouse_guidance_enabled = bool(args.move_mouse_to_goal)
    if mouse_guidance_enabled and not _phase_uses_player_xy_goal(spec):
        mouse_guidance_enabled = False

    attack_keys_map = {1: "NUM4", 2: "NUM6", 3: "NUM5"}
    active_attack_keys = [attack_keys_map[v] for v in sorted(allowed_attack_values) if v in attack_keys_map]

    print("=" * 68)
    print(f"BC DEMO COLLECTION — {args.phase.upper()}")
    print("Input mode: manual only (env key injection disabled)")
    if active_attack_keys:
        print(f"Controls: A/D/S, Space, E, {'/'.join(active_attack_keys)}")
    else:
        print("Controls: A/D/S, Space, E (attacks disabled by phase)")
    print(f"Episodes: {args.episodes} | Output: {out_path}")
    eff_max_steps = int(args.max_episode_steps)
    if eff_max_steps == _DEFAULT_MAX_STEPS["default"] and is_damage:
        eff_max_steps = _DEFAULT_MAX_STEPS["damage"]
    print(f"Max episode steps: {eff_max_steps}")
    if is_damage:
        print(f"Damage phase: goal resampling every {spec.min_goal_duration}-{spec.max_goal_duration} steps")
    if mouse_guidance_enabled:
        print("Mouse guidance: enabled (cursor moves to target_x,target_y)")
    elif args.move_mouse_to_goal:
        print("Mouse guidance: disabled (phase goals are relational/non-XY)")
    if force_drop:
        print("Episode end action: force drop weapon via NUM_5")
    if phase_requires_weapon:
        print("Weapon drop disabled (phase requires weapon)")
    if end_episode_on_first_hit:
        print(f"Episode end action: stop on first hit (op_delta_damage >= {hit_damage_threshold:.4f})")
    print("Press Ctrl+C to stop and save partial data.")
    print("=" * 68)
    print(f"Starting in {args.delay:.1f}s...")
    time.sleep(max(0.0, float(args.delay)))

    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    done_buf: list[bool] = []
    goal_xy_buf: list[np.ndarray] = []
    goal_target_buf: list[np.ndarray] = []
    goal_mask_buf: list[np.ndarray] = []
    op_ko_buf: list[bool] = []

    step_total = 0
    episodes_collected = 0
    episodes_with_hit = 0
    episodes_ended_on_first_hit = 0

    try:
        for ep in range(1, int(args.episodes) + 1):
            obs, info = env.reset()
            done = False
            ep_steps = 0
            ep_had_hit = False
            end_reason = "unknown"

            goal_target = np.asarray(info.get("goal_target", np.zeros(2, dtype=np.float32)), dtype=np.float32)
            goal_mask = np.asarray(info.get("goal_mask", np.zeros_like(goal_target)), dtype=np.float32)
            if mouse_guidance_enabled:
                _set_mouse_to_goal(env, goal_target)

            while not done:
                action = read_action_from_keyboard(allowed_attack_values=allowed_attack_values)
                action_seq = tuple(int(v) for v in action.tolist())

                obs_buf.append(np.asarray(obs, dtype=np.float32).copy())
                act_buf.append(np.asarray(action_seq, dtype=np.int64).copy())
                goal_xy_buf.append(np.asarray(goal_target[:2], dtype=np.float32).copy())
                goal_target_buf.append(np.asarray(goal_target, dtype=np.float32).copy())
                goal_mask_buf.append(np.asarray(goal_mask, dtype=np.float32).copy())

                next_obs, _reward, terminated, truncated, info = env.step(action_seq)
                op_delta_damage = float(max(0.0, info.get("op_delta_damage", 0.0)))
                op_ko_event = float(info.get("op_stock_lost_step", 0.0)) > 0.0
                op_ko_buf.append(op_ko_event)
                hit_event = op_delta_damage >= hit_damage_threshold
                if hit_event:
                    ep_had_hit = True

                terminated_by_hit = bool(end_episode_on_first_hit and hit_event)
                done = bool(terminated or truncated or terminated_by_hit)
                done_buf.append(done)

                if terminated_by_hit:
                    end_reason = "first_hit"
                elif terminated:
                    end_reason = "env_terminated"
                elif truncated:
                    end_reason = "env_truncated"

                if bool(info.get("goal_new_sampled", False)):
                    goal_target = np.asarray(info.get("goal_target", goal_target), dtype=np.float32)
                    goal_mask = np.asarray(info.get("goal_mask", goal_mask), dtype=np.float32)
                    if mouse_guidance_enabled:
                        _set_mouse_to_goal(env, goal_target)

                obs = next_obs
                ep_steps += 1
                step_total += 1

                if step_total % 100 == 0:
                    if is_damage:
                        # Show combat-relevant info for damage phases.
                        op_dd = float(info.get("op_delta_damage", 0.0))
                        g_err = float(info.get("goal_error", 0.0))
                        print(
                            f"steps={step_total} ep={ep}/{args.episodes} ep_steps={ep_steps} "
                            f"op_delta_dmg={op_dd:.4f} goal_err={g_err:.3f}"
                        )
                    else:
                        print(
                            f"steps={step_total} ep={ep}/{args.episodes} ep_steps={ep_steps} "
                            f"goal_xy=({float(goal_target[0]):.3f}, {float(goal_target[1]):.3f})"
                        )

                if ep_steps >= int(args.max_episode_steps):
                    if not done and len(done_buf) > 0:
                        done = True
                        done_buf[-1] = True
                        end_reason = "collector_step_cap"
                    break

            episodes_collected += 1
            if ep_had_hit:
                episodes_with_hit += 1
            if end_reason == "first_hit":
                episodes_ended_on_first_hit += 1

            print(
                f"Episode {ep}/{args.episodes} collected steps={ep_steps} "
                f"reason={end_reason} had_hit={int(ep_had_hit)}"
            )
            if force_drop:
                dropped = _force_drop_weapon_num5_key(drop_controller=drop_controller)
                if not dropped and not drop_warning_printed:
                    print("Force-drop warning: could not send NUM_5 synthetic key press.")
                    drop_warning_printed = True

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial dataset...")
    finally:
        if drop_controller is not None:
            try:
                drop_controller.reset()
            except Exception:
                pass
        env.close()

    if episodes_collected > 0 and end_episode_on_first_hit:
        hit_ratio = float(episodes_with_hit) / float(max(1, episodes_collected))
        first_hit_end_ratio = float(episodes_ended_on_first_hit) / float(max(1, episodes_collected))
        print(
            f"Hit summary: hit_episodes={episodes_with_hit}/{episodes_collected} ({hit_ratio:.2%}), "
            f"ended_on_first_hit={episodes_ended_on_first_hit}/{episodes_collected} ({first_hit_end_ratio:.2%})"
        )

    if len(obs_buf) == 0:
        print("No samples collected; nothing to save.")
        return

    lengths = {
        "obs": len(obs_buf),
        "actions": len(act_buf),
        "dones": len(done_buf),
        "goal_xy": len(goal_xy_buf),
        "goal_target": len(goal_target_buf),
        "goal_mask": len(goal_mask_buf),
        "op_ko_events": len(op_ko_buf),
    }
    aligned_n = min(lengths.values())
    if aligned_n <= 0:
        print("No aligned samples to save after buffer consistency check.")
        return
    if any(v != aligned_n for v in lengths.values()):
        print(f"Buffer mismatch detected ({lengths}); trimming all arrays to {aligned_n} samples.")
        obs_buf = obs_buf[:aligned_n]
        act_buf = act_buf[:aligned_n]
        done_buf = done_buf[:aligned_n]
        goal_xy_buf = goal_xy_buf[:aligned_n]
        goal_target_buf = goal_target_buf[:aligned_n]
        goal_mask_buf = goal_mask_buf[:aligned_n]
        op_ko_buf = op_ko_buf[:aligned_n]
        if len(done_buf) > 0:
            done_buf[-1] = True

    obs_arr = np.stack(obs_buf).astype(np.float32)
    act_arr = np.stack(act_buf).astype(np.int64)
    done_arr = np.asarray(done_buf, dtype=bool)
    goal_xy_arr = np.stack(goal_xy_buf).astype(np.float32)
    goal_target_arr = np.stack(goal_target_buf).astype(np.float32)
    goal_mask_arr = np.stack(goal_mask_buf).astype(np.float32)
    op_ko_arr = np.asarray(op_ko_buf, dtype=bool)

    np.savez_compressed(
        str(out_path),
        obs=obs_arr,
        actions=act_arr,
        dones=done_arr,
        goal_xy=goal_xy_arr,
        goal_target=goal_target_arr,
        goal_mask=goal_mask_arr,
        op_ko_events=op_ko_arr,
        phase=np.asarray([args.phase]),
    )

    print(f"Saved {args.phase} demos")
    print(f"  path   : {out_path}")
    print(f"  obs    : {obs_arr.shape}")
    print(f"  actions: {act_arr.shape}")


if __name__ == "__main__":
    main()
