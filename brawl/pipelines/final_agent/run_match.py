"""Evaluate one typed goal-independent final policy over live stock matches."""

from __future__ import annotations

# Acquire desktop duplication before importing SB3/torch through final_policy.

import brawl.capture_first  # noqa: E402,F401

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from brawl.control.action_space import ACTION_DIM
from brawl.env.live import BrawlDeepEnv, EnvConfig
from brawl.state.state_spec import StateSpec
from brawl.reward.providers import MatchRewardProvider
from brawl.pipelines.final_agent.policy import load_final_model
from brawl.world_model.replay_log import WorldReplayRecorder, WorldReplayWriter
from brawl.paths import LEGACY_RUNS


@dataclass
class MatchResult:
    match: int
    steps: int = 0
    reward: float = 0.0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    stocks_taken: float = 0.0
    stocks_lost: float = 0.0
    weapon_steps: int = 0
    offstage_steps: int = 0
    seconds: float = 0.0
    ended: str = "step_cap"

    @property
    def damage_trade(self) -> float:
        return self.damage_dealt - self.damage_taken

    def row(self) -> dict[str, float | int | str]:
        denom = max(1, self.steps)
        return {
            "match": self.match,
            "ended": self.ended,
            "environment_steps": self.steps,
            "interaction_seconds": round(self.seconds, 3),
            "steps_per_second": round(self.steps / self.seconds, 3) if self.seconds > 0 else 0.0,
            "reward": round(self.reward, 4),
            "damage_dealt": round(self.damage_dealt, 1),
            "damage_taken": round(self.damage_taken, 1),
            "damage_trade": round(self.damage_trade, 1),
            "stocks_taken": self.stocks_taken,
            "stocks_lost": self.stocks_lost,
            "stock_differential": self.stocks_taken - self.stocks_lost,
            "weapon_uptime": round(self.weapon_steps / denom, 4),
            "offstage_rate": round(self.offstage_steps / denom, 4),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one goal-independent final-agent checkpoint"
    )
    parser.add_argument("--model", required=True, help="Typed final-agent .zip checkpoint")
    parser.add_argument("--matches", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--stocks-to-win", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--csv", default=str(LEGACY_RUNS / "final_agent" / "matches.csv"))
    parser.add_argument("--world-replay", default="")
    parser.add_argument("--yolo-every", type=int, default=1)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> int:
    args = parse_args()
    if args.matches <= 0 or args.max_steps <= 0 or args.stocks_to_win <= 0:
        raise SystemExit("--matches, --max-steps, and --stocks-to-win must be positive")

    # Role validation and network deserialisation happen before the live loop.
    # A skill checkpoint cannot accidentally pass because its metadata and
    # 168-dimensional observation contract differ from the final policy.
    model = load_final_model(args.model, device=args.device)
    base = BrawlDeepEnv(
        reward_provider=MatchRewardProvider(),
        config=EnvConfig(
            terminate_on_stock_out=False,
            max_episode_steps=0,
            yolo_infer_every_n_steps=max(1, int(args.yolo_every)),
            action_repeat_steps=1,
            action_repeat_min_steps=1,
            action_repeat_max_steps=1,
            tap_latch_steps=1,
        )
    )
    env: Any = base
    if str(args.world_replay).strip():
        writer = WorldReplayWriter(
            args.world_replay,
            phase="final_agent_match",
            metadata={"source": "run_match", "matches": int(args.matches)},
        )
        env = WorldReplayRecorder(env, writer)
        print(f"World replay log: {writer.dir}")

    print("=" * 70)
    print("FINAL AGENT MATCH EVALUATION -- one policy, no goals or selector")
    print(f"model: {args.model}")
    print(f"observation: {model.observation_space.shape} | action: Discrete({ACTION_DIM})")
    print("=" * 70)
    time.sleep(max(0.0, float(args.delay)))

    deterministic = not bool(args.stochastic)
    stocks_to_win = int(args.stocks_to_win)
    results: list[MatchResult] = []
    try:
        for match in range(1, int(args.matches) + 1):
            obs, _ = env.reset(seed=int(args.seed) + match)
            result = MatchResult(match=match)
            started = time.perf_counter()
            for _ in range(int(args.max_steps)):
                action, _ = model.predict(obs, deterministic=deterministic)
                action_id = int(np.asarray(action, dtype=np.int64).reshape(-1)[0])
                if not 0 <= action_id < ACTION_DIM:
                    raise ValueError(f"final policy returned invalid action {action_id}")
                obs, reward, terminated, truncated, info = env.step(action_id)
                result.steps += 1
                result.reward += float(reward)
                result.damage_dealt += float(info.get("op_delta_damage", 0.0))
                result.damage_taken += float(info.get("self_delta_damage", 0.0))
                result.stocks_taken += float(info.get("op_stock_lost_step", 0.0))
                result.stocks_lost += float(info.get("self_stock_lost_step", 0.0))
                result.weapon_steps += int(StateSpec.get(obs, "player_has_weapon") > 0.5)
                result.offstage_steps += int(StateSpec.get(obs, "player_is_offstage") > 0.5)

                if result.stocks_taken >= stocks_to_win:
                    result.ended = "won"
                    break
                if result.stocks_lost >= stocks_to_win:
                    result.ended = "lost"
                    break
                if terminated or truncated:
                    result.ended = "environment_boundary"
                    break

            result.seconds = time.perf_counter() - started
            results.append(result)
            row = result.row()
            print(
                f"match {match:3d}/{args.matches} {row['ended']:>20s} "
                f"trade={row['damage_trade']:+8.1f} stocks="
                f"{row['stocks_taken']:.0f}-{row['stocks_lost']:.0f} "
                f"steps={row['environment_steps']} hz={row['steps_per_second']:.1f}"
            )
    except KeyboardInterrupt:
        print("\nInterrupted; scoring completed matches.")
    finally:
        env.close()

    if not results:
        print("No matches completed.")
        return 1

    rows = [result.row() for result in results]
    output = Path(args.csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    wins = sum(row["ended"] == "won" for row in rows)
    print("\n" + "=" * 70)
    print(f"matches             {len(rows)}")
    print(f"win rate            {wins / len(rows):8.3f}")
    print(f"stock differential  {_mean(rows, 'stock_differential'):+8.3f}")
    print(f"damage dealt/taken  {_mean(rows, 'damage_dealt'):8.1f} / {_mean(rows, 'damage_taken'):.1f}")
    print(f"offstage rate       {_mean(rows, 'offstage_rate'):8.3f}")
    print(f"environment steps   {sum(int(row['environment_steps']) for row in rows):8d}")
    print(f"interaction seconds {sum(float(row['interaction_seconds']) for row in rows):8.1f}")
    print(f"wrote               {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
