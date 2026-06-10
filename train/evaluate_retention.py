#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.retention import (
    load_best_scores,
    parse_phase_list,
    retention_and_amnesia,
    save_best_scores,
    skill_score_for_phase,
    update_best_scores,
)
from train.train_config import PHASE_CHOICES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LLC retention/amnesia across curriculum phases")
    p.add_argument("--model", type=str, required=True, help="Path to PPO .zip checkpoint")
    p.add_argument("--phase", type=str, default="all_skills_llc", choices=list(PHASE_CHOICES), help="Current training phase")
    p.add_argument("--phases", type=str, default="all", help="'all' or comma/semicolon phase list")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-episode-steps", type=int, default=240)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--death-penalty", type=float, default=1.0)
    p.add_argument("--yolo-every", type=int, default=1)
    p.add_argument("--amnesia-threshold", type=float, default=0.15)
    p.add_argument("--best-scores", type=str, default="", help="JSON best-score file to load/update")
    p.add_argument("--csv", type=str, default="", help="Optional CSV output path")
    return p.parse_args()


def make_env(args: argparse.Namespace, phase: str) -> Any:
    import gymnasium as gym

    from env import BrawlDeepEnv, EnvConfig
    from train.curriculum_config import build_phase_spec
    from train.llc_stage_common import StageGoalEnv

    config = EnvConfig(
        terminate_on_stock_out=True,
        max_episode_steps=max(0, int(args.max_episode_steps)),
        yolo_infer_every_n_steps=max(1, int(args.yolo_every)),
        action_repeat_steps=1,
        action_repeat_min_steps=1,
        action_repeat_max_steps=1,
        tap_latch_steps=1,
    )
    base = BrawlDeepEnv(config=config)
    spec = build_phase_spec(
        phase=phase,
        death_penalty=float(args.death_penalty),
        terminate_on_death=False,
    )
    spec.terminate_on_death = False
    spec.terminate_on_hit_event = False
    return StageGoalEnv(base, spec)


def _to_env_action(env: Any, action: Any) -> Any:
    import gymnasium as gym

    if isinstance(action, np.ndarray) and action.ndim > 1:
        action = action[0]
    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        return np.asarray(action, dtype=np.int64).reshape(-1)
    if isinstance(env.action_space, gym.spaces.Discrete):
        return int(np.asarray(action, dtype=np.int64).reshape(-1)[0])
    return action


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


def evaluate_phase(model: Any, args: argparse.Namespace, phase: str) -> dict[str, float | int | str]:
    from env import BrawlDeepEnv

    env = make_env(args, phase)
    deterministic = not bool(args.stochastic)
    rewards: list[float] = []
    lengths: list[int] = []
    goal_errors: list[float] = []
    goal_successes: list[float] = []
    episode_successes: list[float] = []
    op_damage: list[float] = []
    self_damage: list[float] = []
    hit_rates: list[float] = []
    whiff_rates: list[float] = []
    idle_rates: list[float] = []
    attack_precisions: list[float] = []
    weapon_pickups: list[float] = []
    outcomes: list[str] = []

    try:
        for ep in range(max(1, int(args.episodes))):
            obs, _ = env.reset(seed=int(args.seed + ep + 1))
            terminated = False
            truncated = False
            ep_reward = 0.0
            ep_len = 0
            ep_goal_error = 0.0
            ep_goal_success = 0.0
            ep_had_success = False
            ep_op_dmg = 0.0
            ep_self_dmg = 0.0
            ep_attacks = 0
            ep_hits = 0
            ep_whiffs = 0
            ep_idle = 0
            ep_weapon_pickup = False
            prev_has_weapon = False

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=deterministic)
                env_action = _to_env_action(env, action)
                obs, reward, terminated, truncated, info = env.step(env_action)
                ep_reward += float(reward)
                ep_len += 1
                op_delta = float(info.get("op_delta_damage", 0.0))
                self_delta = float(info.get("self_delta_damage", 0.0))
                ep_op_dmg += op_delta
                ep_self_dmg += self_delta
                ep_goal_error += float(info.get("goal_error", 0.0))
                succ = float(info.get("goal_success", 0.0))
                ep_goal_success += succ
                ep_had_success = bool(ep_had_success or succ > 0.5 or float(info.get("terminal_success", 0.0)) > 0.5)

                stage_action = np.asarray(info.get("stage_action", env_action), dtype=np.int64).reshape(-1)
                movement = int(stage_action[0]) if stage_action.shape[0] > 0 else 3
                jump = int(stage_action[1]) if stage_action.shape[0] > 1 else 0
                dodge = int(stage_action[2]) if stage_action.shape[0] > 2 else 0
                attack = int(stage_action[3]) if stage_action.shape[0] > 3 else 0
                ep_idle += int(movement == 3 and jump == 0 and dodge == 0 and attack == 0)
                ep_attacks += int(attack != 0)
                ep_hits += int(op_delta > 1e-6)
                ep_whiffs += int(attack != 0 and op_delta <= 1e-6)

                raw_feats = np.asarray(info.get("raw_goal_feats", []), dtype=np.float32).reshape(-1)
                names = list(info.get("stage_feature_names", []))
                if "player_has_weapon" in names:
                    idx = names.index("player_has_weapon")
                    has_weapon = bool(idx < raw_feats.shape[0] and raw_feats[idx] > 0.5)
                    ep_weapon_pickup = bool(ep_weapon_pickup or (has_weapon and not prev_has_weapon))
                    prev_has_weapon = has_weapon

            base = env.unwrapped
            self_stocks = float(base.memory.self_stocks_left) if isinstance(base, BrawlDeepEnv) else float("nan")
            op_stocks = float(base.memory.op_stocks_left) if isinstance(base, BrawlDeepEnv) else float("nan")
            outcomes.append(_resolve_outcome(self_stocks, op_stocks, bool(truncated)))
            rewards.append(ep_reward)
            lengths.append(ep_len)
            goal_errors.append(ep_goal_error / max(1, ep_len))
            goal_successes.append(ep_goal_success / max(1, ep_len))
            episode_successes.append(1.0 if ep_had_success else 0.0)
            op_damage.append(ep_op_dmg)
            self_damage.append(ep_self_dmg)
            hit_rates.append(ep_hits / max(1, ep_len))
            whiff_rates.append(ep_whiffs / max(1, ep_attacks))
            idle_rates.append(ep_idle / max(1, ep_len))
            attack_precisions.append(ep_hits / max(1, ep_attacks))
            weapon_pickups.append(1.0 if ep_weapon_pickup else 0.0)
    finally:
        env.close()

    op_np = np.asarray(op_damage, dtype=np.float32)
    self_np = np.asarray(self_damage, dtype=np.float32)
    wins = int(sum(1 for x in outcomes if x == "WIN"))
    losses = int(sum(1 for x in outcomes if x == "LOSS"))
    draws = int(sum(1 for x in outcomes if x == "DRAW"))
    truncs = int(sum(1 for x in outcomes if x == "TRUNC"))
    return {
        "phase": phase,
        "episodes": len(outcomes),
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(lengths)),
        "mean_goal_error": float(np.mean(goal_errors)),
        "mean_goal_success": float(np.mean(goal_successes)),
        "episode_success_rate": float(np.mean(episode_successes)),
        "hit_rate": float(np.mean(hit_rates)),
        "whiff_rate": float(np.mean(whiff_rates)),
        "idle_rate": float(np.mean(idle_rates)),
        "attack_precision": float(np.mean(attack_precisions)),
        "weapon_pickup_rate": float(np.mean(weapon_pickups)),
        "mean_op_damage": float(op_np.mean()),
        "mean_self_damage": float(self_np.mean()),
        "mean_damage_trade": float((op_np - self_np).mean()),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "truncs": truncs,
        "win_rate": float(wins / max(1, len(outcomes))),
    }


def main() -> None:
    from stable_baselines3 import PPO

    args = parse_args()
    phases = parse_phase_list(args.phases, args.phase, include_previous=False)
    model = PPO.load(args.model, device=args.device)
    best_path = Path(args.best_scores) if args.best_scores else Path(args.model).parent / "llc_retention_best.json"
    best_scores = load_best_scores(best_path)

    rows: list[dict[str, float | int | str]] = []
    current_scores: dict[str, float] = {}
    for phase in phases:
        summary = evaluate_phase(model, args, phase)
        score = skill_score_for_phase(phase, summary)
        best_ref = max(float(best_scores.get(phase, 0.0)), score)
        retention, amnesia = retention_and_amnesia(score, best_ref)
        summary["skill_score"] = float(score)
        summary["best_skill_score"] = float(best_ref)
        summary["retention"] = float(retention)
        summary["amnesia"] = float(amnesia)
        summary["amnesia_gate_pass"] = int(amnesia <= float(args.amnesia_threshold))
        current_scores[phase] = float(score)
        rows.append(summary)
        gate = "PASS" if int(summary["amnesia_gate_pass"]) else "AMNESIA"
        print(
            f"{phase:20s} score={score:.3f} retain={retention:.3f} amnesia={amnesia:.3f} {gate} "
            f"succ={float(summary['episode_success_rate']):.3f} "
            f"trade={float(summary['mean_damage_trade']):+.3f} win={float(summary['win_rate']):.3f}"
        )

    save_best_scores(best_path, update_best_scores(best_scores, current_scores))
    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
