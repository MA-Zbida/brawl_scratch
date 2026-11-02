from __future__ import annotations


# The live entrypoint eventually imports Stable-Baselines3/torch. Acquire DXGI
# first so the same diagnostic works on hybrid-GPU laptops as training does.


import brawl.capture_first  # noqa: E402,F401

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from brawl.control.action_space import ACTION_DIM, Action
from brawl.calibration import PLATFORM_BOUNDS
from brawl.training.curriculum_goals import GOAL_INDEX, default_goal_target
from brawl.pipelines.llc.retention import phase_score_threshold, skill_score_for_phase
from brawl.compat import load_sb3


@dataclass(frozen=True)
class MovementChallenge:
    challenge_id: str
    slice: str
    player_x: float
    player_y: float


def movement_challenges() -> tuple[MovementChallenge, ...]:
    """Return a short fixed route covering grounded and airborne movement."""
    x_min = float(PLATFORM_BOUNDS["x_min"])
    x_max = float(PLATFORM_BOUNDS["x_max"])
    ground_y = float(np.clip(float(PLATFORM_BOUNDS["y_min"]) + 0.015, 0.0, 1.0))
    center_x = 0.5 * (x_min + x_max)
    return (
        MovementChallenge("left_ground", "ground", x_min + 0.04, ground_y),
        MovementChallenge("right_ground", "ground", x_max - 0.04, ground_y),
        MovementChallenge("center_ground", "ground", center_x, ground_y),
        MovementChallenge("left_air", "air", x_min + 0.08, ground_y - 0.14),
        MovementChallenge("right_air", "air", x_max - 0.08, ground_y - 0.14),
    )


def _target_for_challenge(challenge: MovementChallenge) -> np.ndarray:
    target = default_goal_target()
    target[GOAL_INDEX["player_x"]] = float(challenge.player_x)
    target[GOAL_INDEX["player_y"]] = float(challenge.player_y)
    return target.astype(np.float32, copy=False)


def _summarize_rows(
    rows: Sequence[dict[str, Any]],
    *,
    model_label: str,
    model_path: str,
    deterministic: bool,
) -> dict[str, Any]:
    eligible = [row for row in rows if not bool(row["trivial_start"])]
    success_rate = float(np.mean([float(row["success"]) for row in eligible])) if eligible else 0.0
    mean_error = float(np.mean([float(row["mean_goal_error"]) for row in eligible])) if eligible else 1.0
    step_success = float(np.mean([float(row["mean_goal_success"]) for row in eligible])) if eligible else 0.0
    score_inputs = {
        "episode_success_rate": success_rate,
        "mean_goal_success": step_success,
        "mean_goal_error": mean_error,
    }
    score = skill_score_for_phase("movement_fluency", score_inputs)
    threshold = phase_score_threshold("movement_fluency")
    return {
        "model_label": model_label,
        "model_path": model_path,
        "policy_mode": "deterministic" if deterministic else "stochastic",
        "episodes": len(rows),
        "eligible_episodes": len(eligible),
        "trivial_starts": len(rows) - len(eligible),
        "success_rate": success_rate,
        "mean_goal_error": mean_error,
        "mean_goal_success": step_success,
        "skill_score": score,
        "promotion_threshold": threshold,
        "promotion_pass": int(score >= threshold),
    }


def evaluate_variant(
    model: Any,
    env: Any,
    challenges: Iterable[MovementChallenge],
    *,
    model_label: str,
    model_path: str,
    deterministic: bool,
    seed: int,
    max_episode_steps: int = 200,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate one policy mode on an explicit target route.

    The target coordinates are paired across variants. Brawlhalla cannot restore
    an exact physical starting state, so start error is recorded and trivial
    starts are excluded rather than being misreported as skill successes.
    """
    rows: list[dict[str, Any]] = []
    for episode_index, challenge in enumerate(challenges):
        episode_seed = int(seed + episode_index + 1)
        obs, info = env.reset(seed=episode_seed)
        target = _target_for_challenge(challenge)
        obs, info = env.set_goal_target(obs, info, target)
        start_error = float(env.goal_error(obs, info))
        threshold = float(info.get("goal_success_threshold", 0.04))
        trivial_start = bool(start_error < threshold)

        action_ids: list[int] = []
        errors: list[float] = []
        successes: list[float] = []
        reward_sum = 0.0
        terminated = False
        truncated = False

        if not trivial_start:
            for _ in range(max(1, int(max_episode_steps))):
                action, _ = model.predict(obs, deterministic=deterministic)
                action_id = int(np.asarray(action, dtype=np.int64).reshape(-1)[0])
                action_id = int(np.clip(action_id, 0, ACTION_DIM - 1))
                obs, reward, terminated, truncated, step_info = env.step(action_id)
                executed = int(step_info.get("stage_action", action_id))
                action_ids.append(executed)
                reward_sum += float(reward)
                errors.append(float(step_info.get("goal_error", 0.0)))
                successes.append(float(step_info.get("goal_success", 0.0)))
                if terminated or truncated:
                    break

        success = bool(any(value > 0.5 for value in successes))
        counts = np.bincount(np.asarray(action_ids, dtype=np.int64), minlength=ACTION_DIM)
        target_world = np.asarray(info.get("goal_target_world", target), dtype=np.float32).reshape(-1)
        row = {
            "model_label": model_label,
            "model_path": model_path,
            "policy_mode": "deterministic" if deterministic else "stochastic",
            "challenge_id": challenge.challenge_id,
            "challenge_slice": challenge.slice,
            "seed": episode_seed,
            "target_player_x": float(challenge.player_x),
            "target_player_y": float(challenge.player_y),
            "goal_target_world": json.dumps(target_world.tolist(), separators=(",", ":")),
            "start_goal_error": start_error,
            "trivial_start": int(trivial_start),
            "steps": len(action_ids),
            "success": int(success),
            "total_reward": reward_sum,
            "mean_goal_error": float(np.mean(errors)) if errors else start_error,
            "final_goal_error": float(errors[-1]) if errors else start_error,
            "mean_goal_success": float(np.mean(successes)) if successes else 0.0,
            "terminated": int(terminated),
            "truncated": int(truncated),
            "idle_rate": float(counts[int(Action.NOOP)] / max(1, len(action_ids))),
            "action_ids": json.dumps(action_ids, separators=(",", ":")),
            "action_counts": json.dumps(counts.tolist(), separators=(",", ":")),
        }
        rows.append(row)

    return rows, _summarize_rows(
        rows,
        model_label=model_label,
        model_path=model_path,
        deterministic=deterministic,
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare movement BC and PPO checkpoints on one fixed target route",
    )
    parser.add_argument("--bc-model", required=True, help="Movement BC initializer .zip")
    parser.add_argument("--ppo-model", required=True, help="Movement PPO checkpoint .zip")
    parser.add_argument("--csv", required=True, help="Per-episode diagnostic CSV")
    parser.add_argument("--summary-csv", default="", help="Summary CSV; defaults beside --csv")
    parser.add_argument("--repeats", type=int, default=1, help="Number of five-goal route repetitions")
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--death-penalty", type=float, default=1.0)
    parser.add_argument("--yolo-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    from stable_baselines3 import PPO

    from brawl.pipelines.llc.evaluate_retention import make_env

    args = parse_args()
    models = {
        "bc": (str(args.bc_model), load_sb3(PPO, args.bc_model, device=args.device)),
        "ppo": (str(args.ppo_model), load_sb3(PPO, args.ppo_model, device=args.device)),
    }
    env = make_env(args, "movement_fluency")
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    challenges = movement_challenges()

    print(
        "Fixed targets are paired across variants; exact physical start states "
        "cannot be restored in Training Mode. Start error and trivial starts are logged."
    )
    try:
        for model_label, (model_path, model) in models.items():
            for deterministic in (True, False):
                variant_rows: list[dict[str, Any]] = []
                for repeat in range(max(1, int(args.repeats))):
                    if hasattr(model, "set_random_seed"):
                        model.set_random_seed(int(args.seed + repeat))
                    rows, _ = evaluate_variant(
                        model,
                        env,
                        challenges,
                        model_label=model_label,
                        model_path=model_path,
                        deterministic=deterministic,
                        seed=int(args.seed + repeat * 100),
                        max_episode_steps=int(args.max_episode_steps),
                    )
                    for row in rows:
                        row["repeat"] = repeat
                    variant_rows.extend(rows)
                summary = _summarize_rows(
                    variant_rows,
                    model_label=model_label,
                    model_path=model_path,
                    deterministic=deterministic,
                )
                all_rows.extend(variant_rows)
                summaries.append(summary)
                print(
                    f"{model_label:4s} {summary['policy_mode']:13s} "
                    f"score={float(summary['skill_score']):.3f} "
                    f"success={float(summary['success_rate']):.3f} "
                    f"error={float(summary['mean_goal_error']):.3f} "
                    f"trivial={int(summary['trivial_starts'])} "
                    f"promotion={'PASS' if int(summary['promotion_pass']) else 'STOP'}"
                )
    finally:
        env.close()

    csv_path = Path(args.csv)
    summary_path = (
        Path(args.summary_csv)
        if args.summary_csv
        else csv_path.with_name(f"{csv_path.stem}_summary.csv")
    )
    _write_csv(csv_path, all_rows)
    _write_csv(summary_path, summaries)
    print(f"episodes: {csv_path}")
    print(f"summary : {summary_path}")


if __name__ == "__main__":
    main()
