#!/usr/bin/env python
"""Run the frozen LLC under a scripted goal selector and score the result.

This is the milestone harness. It answers one question the per-phase metrics
cannot: does the whole system, driven end to end, produce competent play?

The goal selector is deliberately **scripted, not learned**. That separates two
failure modes which look identical in a video:

  * the selector picks the right skill and the LLC executes it badly
        -> keep working on the LLC
  * each skill executes cleanly but the sequencing is incoherent
        -> the LLC is sufficient; start the HSP

Scoring is matched to training mode, which has no win condition: damage dealt
against damage taken, stocks from the RGB kill-indicator detector, weapon uptime,
and self-destructs. Every match also writes to the world replay log, so a bad
result is diagnosable afterwards rather than merely disappointing.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import capture_first  # noqa: E402,F401  (import order is load-bearing)

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Any, Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from action_space import ACTION_DIM  # noqa: E402
from feature_extractor.memory.state_spec import StateSpec  # noqa: E402

#: Skills the selector can request, in priority order. Recovery outranks
#: everything: being offstage is the only state that ends the match outright.
SKILLS = ("recovery_mastery", "weapon_acquisition", "spacing_neutral", "combat_execution")


@dataclass
class SelectorConfig:
    """Thresholds for the scripted selector, in normalised observation units."""

    #: Below this, the agent is close enough that spacing matters more than approach.
    too_close: float = 0.06
    #: Above this, close the gap rather than trade.
    too_far: float = 0.28
    #: Hysteresis: hold a chosen skill this many steps before reconsidering, so the
    #: agent commits to an approach instead of oscillating on the threshold.
    commit_steps: int = 8


@dataclass
class MatchResult:
    match: int = 0
    steps: int = 0
    seconds: float = 0.0
    env_episodes: int = 1
    ended: str = "step_cap"
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    stocks_taken: float = 0.0
    stocks_lost: float = 0.0
    weapon_steps: int = 0
    offstage_steps: int = 0
    skill_steps: dict[str, int] = field(default_factory=lambda: {s: 0 for s in SKILLS})

    @property
    def damage_trade(self) -> float:
        return self.damage_dealt - self.damage_taken

    def row(self) -> dict[str, Any]:
        out = {
            "match": self.match,
            "steps": self.steps,
            "seconds": round(self.seconds, 1),
            "env_episodes": self.env_episodes,
            "ended": self.ended,
            "damage_dealt": round(self.damage_dealt, 1),
            "damage_taken": round(self.damage_taken, 1),
            "damage_trade": round(self.damage_trade, 1),
            "stocks_taken": self.stocks_taken,
            "stocks_lost": self.stocks_lost,
            "weapon_uptime": round(self.weapon_steps / max(1, self.steps), 3),
            "offstage_rate": round(self.offstage_steps / max(1, self.steps), 3),
        }
        for skill in SKILLS:
            out[f"pct_{skill}"] = round(self.skill_steps[skill] / max(1, self.steps), 3)
        return out


def select_skill(obs: np.ndarray, cfg: SelectorConfig) -> str:
    """Choose a skill from the current observation. Fixed rules, no learning."""
    if StateSpec.get(obs, "player_is_offstage") > 0.5:
        return "recovery_mastery"
    if StateSpec.get(obs, "player_has_weapon") <= 0.5:
        return "weapon_acquisition"

    distance = float(StateSpec.get(obs, "rel_distance"))
    if distance < cfg.too_close or distance > cfg.too_far:
        return "spacing_neutral"
    return "combat_execution"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", type=str, default="train/models",
                   help="Directory holding llc_<phase>.zip or llc_<phase>_bc_init.zip per skill.")
    p.add_argument("--matches", type=int, default=30)
    p.add_argument("--stocks-to-win", type=int, default=1,
                   help="Match ends when either side loses this many stocks. 1 gives fast "
                        "diagnostic matches; a real Brawlhalla match is 3.")
    p.add_argument("--max-steps", type=int, default=3000, help="Step cap per match (~64s at 47 Hz).")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stochastic", action="store_true", help="Sample actions instead of argmax.")
    p.add_argument("--world-replay", type=str, default="data/world_replay")
    p.add_argument("--csv", type=str, default="train/models/match_results.csv")
    p.add_argument("--commit-steps", type=int, default=SelectorConfig.commit_steps)
    return p.parse_args(argv)


def _resolve_models(models_dir: Path_t, device: str) -> dict[str, Any]:  # type: ignore[valid-type]
    from stable_baselines3 import PPO

    root = _Path(models_dir)
    models: dict[str, Any] = {}
    for skill in SKILLS:
        for name in (f"llc_{skill}.zip", f"llc_{skill}_bc_init.zip"):
            path = root / name
            if path.exists():
                models[skill] = PPO.load(str(path), device=device)
                print(f"  {skill:<20} <- {path.name}")
                break
        else:
            print(f"  {skill:<20} <- MISSING (this skill will fall back to combat)")
    if not models:
        raise FileNotFoundError(f"No LLC checkpoints found under {root}")
    return models


Path_t = str  # keeps the annotation above readable without importing Path twice


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    from env import BrawlDeepEnv, EnvConfig
    from train.curriculum_config import build_phase_spec
    from train.llc_stage_common import StageGoalEnv
    from world_model import WorldReplayRecorder, WorldReplayWriter

    cfg = SelectorConfig(commit_steps=max(1, int(args.commit_steps)))
    stocks_to_win = max(1, int(args.stocks_to_win))

    print("=" * 70)
    print("MATCH RUN -- frozen LLC under a scripted goal selector")
    print("=" * 70)
    models = _resolve_models(args.models_dir, args.device)
    fallback = models.get("combat_execution") or next(iter(models.values()))

    base = BrawlDeepEnv(config=EnvConfig(
        terminate_on_stock_out=False,
        max_episode_steps=0,               # the match loop owns termination
        action_repeat_steps=1,
        tap_latch_steps=1,
    ))
    # One spec drives the observation layout; the selector swaps which policy acts,
    # not which environment runs. Keeping a single env avoids re-acquiring capture.
    spec = build_phase_spec(phase="combat_execution", death_penalty=1.0, terminate_on_death=False)
    # A match is not an episode. The game runs continuously, so every termination
    # the training wrapper produces -- goal reached, hit landed, death -- is a
    # training construct that must not end a match. Silence what we can here; the
    # match loop absorbs the rest by resetting and continuing.
    spec.terminate_on_death = False
    spec.terminate_on_hit_event = False
    env: Any = StageGoalEnv(base, spec)

    if str(args.world_replay).strip():
        writer = WorldReplayWriter(args.world_replay, phase="match_run",
                                   metadata={"source": "run_match", "matches": int(args.matches)})
        print(f"World replay log: {writer.dir}")
        env = WorldReplayRecorder(env, writer)

    results: list[MatchResult] = []
    deterministic = not bool(args.stochastic)

    try:
        for match in range(1, int(args.matches) + 1):
            obs, _ = env.reset(seed=int(args.seed) + match)
            result = MatchResult(match=match)
            started = time.perf_counter()
            held_skill, held_for = "combat_execution", 0

            for _ in range(int(args.max_steps)):
                if held_for <= 0:
                    held_skill = select_skill(obs, cfg)
                    held_for = cfg.commit_steps
                held_for -= 1

                policy = models.get(held_skill, fallback)
                action, _ = policy.predict(obs, deterministic=deterministic)
                action_id = int(np.asarray(action, dtype=np.int64).reshape(-1)[0])
                if not 0 <= action_id < ACTION_DIM:
                    raise ValueError(f"policy for {held_skill} returned action {action_id}")

                obs, _reward, terminated, truncated, info = env.step(action_id)

                result.steps += 1
                result.skill_steps[held_skill] = result.skill_steps.get(held_skill, 0) + 1
                result.damage_dealt += float(info.get("op_delta_damage", 0.0))
                result.damage_taken += float(info.get("self_delta_damage", 0.0))
                result.stocks_taken += float(info.get("op_stock_lost_step", 0.0))
                result.stocks_lost += float(info.get("self_stock_lost_step", 0.0))
                result.weapon_steps += int(StateSpec.get(obs, "player_has_weapon") > 0.5)
                result.offstage_steps += int(StateSpec.get(obs, "player_is_offstage") > 0.5)

                # The match ends on stocks, and nothing else.
                if result.stocks_taken >= stocks_to_win:
                    result.ended = "won"
                    break
                if result.stocks_lost >= stocks_to_win:
                    result.ended = "lost"
                    break

                if terminated or truncated:
                    # A training-wrapper episode boundary, not the end of the match.
                    # Reset to get a fresh episode and keep playing; the game never
                    # stopped. Without this a match lasted 2-3 steps.
                    obs, _ = env.reset(seed=int(args.seed) + match * 1000 + result.env_episodes)
                    result.env_episodes += 1
                    held_for = 0

            result.seconds = time.perf_counter() - started
            results.append(result)
            r = result.row()
            print(f"match {match:3d}/{args.matches} {r['ended']:>9s}  trade={r['damage_trade']:+8.1f} "
                  f"(dealt {r['damage_dealt']:.0f} / took {r['damage_taken']:.0f})  "
                  f"stocks {r['stocks_taken']:.0f}-{r['stocks_lost']:.0f}  "
                  f"weapon={r['weapon_uptime']:.2f}  steps={r['steps']} "
                  f"env_eps={r['env_episodes']}")
    except KeyboardInterrupt:
        print("\nInterrupted; scoring what completed.")
    finally:
        env.close()

    if not results:
        print("No matches completed.")
        return 1

    rows = [r.row() for r in results]
    out = _Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer_csv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    print("\n" + "=" * 70)
    print(f"{len(rows)} matches")
    print(f"  damage trade      {mean('damage_trade'):+8.1f} per match "
          f"({sum(1 for r in rows if r['damage_trade'] > 0)}/{len(rows)} positive)")
    print(f"  dealt / taken     {mean('damage_dealt'):8.1f} / {mean('damage_taken'):.1f}")
    print(f"  stocks taken/lost {mean('stocks_taken'):8.2f} / {mean('stocks_lost'):.2f}")
    print(f"  weapon uptime     {mean('weapon_uptime'):8.3f}")
    print(f"  offstage rate     {mean('offstage_rate'):8.3f}")
    print(f"  steps per match   {mean('steps'):8.1f}   env resets absorbed: {mean('env_episodes') - 1:.1f}")
    outcomes = {}
    for row in rows:
        outcomes[row["ended"]] = outcomes.get(row["ended"], 0) + 1
    print(f"  outcomes          {outcomes}")
    print("\n  skill selection (fraction of steps):")
    for skill in SKILLS:
        print(f"    {skill:<22}{mean(f'pct_{skill}'):.3f}")
    print(f"\n  wrote {out}")
    print("\nReading this: if selection looks sane but the trade is negative, the LLC")
    print("needs work. If individual skills look clean and selection is incoherent,")
    print("the LLC is sufficient and the HSP is the next thing to build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
