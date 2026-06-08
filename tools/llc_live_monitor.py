#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.check_llc_phase_gate import evaluate_gate
from train.retention import PHASE_ORDER, phase_score_threshold


COMBAT_PHASES = {"combat_execution", "all_skills_llc"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor LLC training CSVs for collapse and retention signals")
    p.add_argument("--phase", required=True, choices=list(PHASE_ORDER), help="Current LLC phase")
    p.add_argument("--models-dir", type=str, default="train/models")
    p.add_argument("--steps-csv", type=str, default="")
    p.add_argument("--episodes-csv", type=str, default="")
    p.add_argument("--eval-csv", type=str, default="")
    p.add_argument("--interval", type=float, default=10.0, help="Seconds between refreshes when not using --once")
    p.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    p.add_argument("--tail-steps", type=int, default=500)
    p.add_argument("--tail-episodes", type=int, default=20)
    p.add_argument("--max-idle-rate", type=float, default=0.45)
    p.add_argument("--max-whiff-rate", type=float, default=0.80)
    p.add_argument("--min-action-entropy", type=float, default=0.15)
    p.add_argument("--min-combat-damage-trade", type=float, default=0.0)
    p.add_argument("--min-retention", type=float, default=0.85)
    p.add_argument("--amnesia-threshold", type=float, default=0.15)
    p.add_argument("--fail-on-alert", action="store_true", help="Exit with code 2 when hard alerts are present")
    return p.parse_args()


def _default_csvs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    prefix = f"llc_{str(args.phase).strip().lower()}"
    models_dir = Path(str(args.models_dir))
    steps = Path(args.steps_csv) if str(args.steps_csv).strip() else models_dir / f"{prefix}_steps.csv"
    episodes = Path(args.episodes_csv) if str(args.episodes_csv).strip() else models_dir / f"{prefix}_episodes.csv"
    eval_csv = Path(args.eval_csv) if str(args.eval_csv).strip() else models_dir / f"{prefix}_eval.csv"
    return steps, episodes, eval_csv


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def _as_float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        text = str(row.get(key, "")).strip()
        return float(text) if text else float(default)
    except Exception:
        return float(default)


def _mean(rows: list[dict[str, str]], key: str, default: float = float("nan")) -> float:
    values = [_as_float(row, key) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return float(default)
    return float(sum(values) / len(values))


def _latest_by_phase(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for row in rows:
        phase = str(row.get("phase", "")).strip().lower()
        if phase:
            latest[phase] = row
    return latest


def _fmt(value: float, digits: int = 3, *, signed: bool = False) -> str:
    if not math.isfinite(float(value)):
        return "n/a"
    prefix = "+" if signed else ""
    return f"{float(value):{prefix}.{digits}f}"


def _status_from_metric(value: float, *, high_bad: bool, threshold: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    failed = value > threshold if high_bad else value < threshold
    return "STOP" if failed else "OK"


def summarize_steps(rows: list[dict[str, str]], tail: int) -> dict[str, Any]:
    recent = rows[-max(1, int(tail)) :]
    goal_counts = Counter(str(row.get("goal_type", "unknown")).strip() or "unknown" for row in recent)
    top_goals = ", ".join(f"{goal}:{count}" for goal, count in goal_counts.most_common(4))
    latest_step = 0
    if rows:
        try:
            latest_step = int(float(rows[-1].get("step", 0)))
        except Exception:
            latest_step = len(rows)
    return {
        "rows": len(rows),
        "recent": len(recent),
        "latest_step": latest_step,
        "reward": _mean(recent, "reward"),
        "goal_error": _mean(recent, "goal_error"),
        "goal_success": _mean(recent, "goal_success"),
        "idle": _mean(recent, "idle"),
        "hit": _mean(recent, "hit"),
        "whiff": _mean(recent, "whiff"),
        "damage_trade": _mean(recent, "damage_trade"),
        "top_goals": top_goals or "n/a",
    }


def summarize_episodes(rows: list[dict[str, str]], tail: int) -> dict[str, Any]:
    recent = rows[-max(1, int(tail)) :]
    latest_episode = 0
    if rows:
        try:
            latest_episode = int(float(rows[-1].get("episode", 0)))
        except Exception:
            latest_episode = len(rows)
    return {
        "rows": len(rows),
        "recent": len(recent),
        "latest_episode": latest_episode,
        "return": _mean(recent, "return"),
        "success": _mean(recent, "episode_success"),
        "success_ratio": _mean(recent, "success_ratio"),
        "goal_error": _mean(recent, "mean_goal_error"),
        "time_to_success": _mean(recent, "time_to_success"),
        "action_entropy": _mean(recent, "action_entropy"),
        "idle": _mean(recent, "idle_rate"),
        "whiff": _mean(recent, "whiff_rate"),
        "attack_precision": _mean(recent, "attack_precision"),
        "damage_trade": _mean(recent, "damage_trade"),
    }


def summarize_eval(rows: list[dict[str, str]], current_phase: str) -> list[dict[str, Any]]:
    latest = _latest_by_phase(rows)
    summaries: list[dict[str, Any]] = []
    for phase in PHASE_ORDER:
        row = latest.get(phase)
        if row is None:
            continue
        skill = _as_float(row, "skill_score", 0.0)
        threshold = phase_score_threshold(phase)
        retention = _as_float(row, "retention", 1.0 if skill > 0 else 0.0)
        amnesia = _as_float(row, "amnesia", max(0.0, 1.0 - retention))
        idle = _as_float(row, "idle_rate", 0.0)
        whiff = _as_float(row, "whiff_rate", 0.0)
        trade = _as_float(row, "mean_damage_trade", 0.0)
        status = "OK"
        if phase == current_phase and skill < threshold:
            status = "STOP"
        if phase != current_phase and (retention < 0.85 or amnesia > 0.15):
            status = "STOP"
        if phase in COMBAT_PHASES and trade < 0.0:
            status = "STOP"
        summaries.append(
            {
                "phase": phase,
                "skill": skill,
                "threshold": threshold,
                "retention": retention,
                "amnesia": amnesia,
                "idle": idle,
                "whiff": whiff,
                "trade": trade,
                "status": status,
            }
        )
    return summaries


def _gate_failures(eval_csv: Path, args: argparse.Namespace) -> list[str]:
    if not eval_csv.exists():
        return []
    gate_args = argparse.Namespace(
        eval_csv=str(eval_csv),
        phase=str(args.phase),
        phases="all" if str(args.phase) == "all_skills_llc" else "",
        amnesia_threshold=float(args.amnesia_threshold),
        min_retention=float(args.min_retention),
        min_current_score=None,
        max_idle_rate=float(args.max_idle_rate),
        max_combat_whiff_rate=float(args.max_whiff_rate),
        min_combat_damage_trade=float(args.min_combat_damage_trade),
        no_combat_trade_gate=False,
    )
    try:
        passed, failures, _ = evaluate_gate(gate_args)
    except Exception:
        return []
    return [] if passed else failures


def build_snapshot(args: argparse.Namespace) -> tuple[str, list[str]]:
    phase = str(args.phase).strip().lower()
    steps_csv, episodes_csv, eval_csv = _default_csvs(args)
    step_rows = _read_csv(steps_csv)
    episode_rows = _read_csv(episodes_csv)
    eval_rows = _read_csv(eval_csv)
    step = summarize_steps(step_rows, int(args.tail_steps))
    episode = summarize_episodes(episode_rows, int(args.tail_episodes))
    eval_summary = summarize_eval(eval_rows, phase)
    hard_alerts: list[str] = []
    soft_notes: list[str] = []

    if not step_rows:
        soft_notes.append(f"Step CSV missing or empty: {steps_csv.as_posix()}")
    if not episode_rows:
        soft_notes.append(f"Episode CSV missing or empty: {episodes_csv.as_posix()}")
    if not eval_rows:
        soft_notes.append(f"Eval CSV missing or empty: {eval_csv.as_posix()}")

    if math.isfinite(episode["idle"]) and episode["idle"] > float(args.max_idle_rate):
        hard_alerts.append(f"Idle collapse risk: episode idle_rate={episode['idle']:.3f} > {float(args.max_idle_rate):.3f}")
    if math.isfinite(episode["action_entropy"]) and episode["action_entropy"] < float(args.min_action_entropy):
        hard_alerts.append(
            f"Action collapse risk: action_entropy={episode['action_entropy']:.3f} < {float(args.min_action_entropy):.3f}"
        )
    if math.isfinite(episode["whiff"]) and episode["whiff"] > float(args.max_whiff_rate):
        hard_alerts.append(f"Attack spam risk: whiff_rate={episode['whiff']:.3f} > {float(args.max_whiff_rate):.3f}")
    if phase in COMBAT_PHASES and math.isfinite(episode["damage_trade"]) and episode["damage_trade"] < float(args.min_combat_damage_trade):
        hard_alerts.append(
            f"Combat trade risk: episode damage_trade={episode['damage_trade']:+.3f} < {float(args.min_combat_damage_trade):+.3f}"
        )

    hard_alerts.extend(_gate_failures(eval_csv, args))

    lines: list[str] = []
    lines.append(f"LLC LIVE MONITOR phase={phase} time={time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"steps_csv={steps_csv.as_posix()}")
    lines.append(f"episodes_csv={episodes_csv.as_posix()}")
    lines.append(f"eval_csv={eval_csv.as_posix()}")
    lines.append("")
    lines.append(
        "Steps recent={recent}/{rows} latest={latest_step} reward={reward} "
        "goal_error={goal_error} goal_success={goal_success} idle={idle} "
        "hit={hit} whiff={whiff} trade={trade} goals={top_goals}".format(
            recent=step["recent"],
            rows=step["rows"],
            latest_step=step["latest_step"],
            reward=_fmt(step["reward"], signed=True),
            goal_error=_fmt(step["goal_error"]),
            goal_success=_fmt(step["goal_success"]),
            idle=_fmt(step["idle"]),
            hit=_fmt(step["hit"]),
            whiff=_fmt(step["whiff"]),
            trade=_fmt(step["damage_trade"], signed=True),
            top_goals=step["top_goals"],
        )
    )
    lines.append(
        "Episodes recent={recent}/{rows} latest={latest_episode} return={return_} "
        "success={success} success_ratio={success_ratio} mean_error={goal_error} "
        "tts={tts} entropy={entropy} idle={idle} whiff={whiff} precision={precision} "
        "trade={trade}".format(
            recent=episode["recent"],
            rows=episode["rows"],
            latest_episode=episode["latest_episode"],
            return_=_fmt(episode["return"], signed=True),
            success=_fmt(episode["success"]),
            success_ratio=_fmt(episode["success_ratio"]),
            goal_error=_fmt(episode["goal_error"]),
            tts=_fmt(episode["time_to_success"], digits=1),
            entropy=_fmt(episode["action_entropy"]),
            idle=_fmt(episode["idle"]),
            whiff=_fmt(episode["whiff"]),
            precision=_fmt(episode["attack_precision"]),
            trade=_fmt(episode["damage_trade"], signed=True),
        )
    )

    if eval_summary:
        lines.append("")
        lines.append("Eval latest:")
        lines.append("phase                  skill  min    retain amnesia idle   whiff  trade   gate")
        lines.append("----------------------------------------------------------------------------")
        for row in eval_summary:
            lines.append(
                f"{row['phase']:20s} "
                f"{_fmt(row['skill']):>6s} "
                f"{_fmt(row['threshold']):>6s} "
                f"{_fmt(row['retention']):>6s} "
                f"{_fmt(row['amnesia']):>7s} "
                f"{_fmt(row['idle']):>6s} "
                f"{_fmt(row['whiff']):>6s} "
                f"{_fmt(row['trade'], signed=True):>7s} "
                f"{row['status']:>4s}"
            )

    if soft_notes:
        lines.append("")
        lines.append("Waiting:")
        for note in soft_notes:
            lines.append(f"- {note}")

    lines.append("")
    if hard_alerts:
        lines.append("STOP SIGNALS:")
        for alert in hard_alerts:
            lines.append(f"- {alert}")
        lines.append("Action: pause advancement; fix demos/perception/reward or run rehearsal before continuing.")
    else:
        lines.append("Status: no hard collapse signal in the available CSV window.")

    return "\n".join(lines), hard_alerts


def main() -> None:
    args = parse_args()
    while True:
        snapshot, alerts = build_snapshot(args)
        print(snapshot)
        if args.once:
            if alerts and bool(args.fail_on_alert):
                raise SystemExit(2)
            raise SystemExit(0)
        print("")
        time.sleep(max(1.0, float(args.interval)))


if __name__ == "__main__":
    main()
