#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.retention import parse_phase_list, phase_score_threshold, previous_phases


COMBAT_PHASES = {"combat_execution", "all_skills_llc"}
DEMO_PATHS: dict[str, str] = {
    "recovery_mastery": "train/models/recovery_mastery_demos.npz",
    "movement_fluency": "train/models/movement_fluency_demos.npz",
    "weapon_acquisition": "train/models/weapon_acquisition_demos.npz",
    "spacing_neutral": "train/models/spacing_neutral_demos.npz",
    "combat_execution": "train/models/combat_execution_demos.npz",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check whether an LLC phase can advance without skill collapse")
    p.add_argument("--eval-csv", type=str, required=True, help="Path to *_eval.csv or *_retention_eval.csv")
    p.add_argument("--phase", type=str, required=True, help="Current phase being gated")
    p.add_argument("--phases", type=str, default="", help="'all' or comma/semicolon phase list. Default: current + previous")
    p.add_argument("--amnesia-threshold", type=float, default=0.15)
    p.add_argument("--min-retention", type=float, default=0.85)
    p.add_argument(
        "--min-current-score",
        type=float,
        default=None,
        help="Override current phase skill threshold. Default: phase-specific threshold.",
    )
    p.add_argument("--max-idle-rate", type=float, default=0.45)
    p.add_argument("--max-combat-whiff-rate", type=float, default=0.80)
    p.add_argument("--min-combat-damage-trade", type=float, default=0.0)
    p.add_argument("--no-combat-trade-gate", action="store_true", help="Do not require non-negative damage trade in combat phases")
    return p.parse_args()


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {p}")
    with p.open("r", newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        text = str(row.get(key, "")).strip()
        return float(text) if text else float(default)
    except Exception:
        return float(default)


def _latest_by_phase(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for row in rows:
        phase = str(row.get("phase", "")).strip().lower()
        if not phase:
            continue
        latest[phase] = row
    return latest


def _required_phases(raw: str, current_phase: str) -> list[str]:
    if raw.strip():
        return parse_phase_list(raw, current_phase, include_previous=False)
    return previous_phases(current_phase, include_current=True)


def evaluate_gate(args: argparse.Namespace) -> tuple[bool, list[str], list[dict[str, Any]]]:
    rows = _read_rows(args.eval_csv)
    latest = _latest_by_phase(rows)
    current_phase = str(args.phase).strip().lower()
    required = _required_phases(str(args.phases or ""), current_phase)

    failures: list[str] = []
    table: list[dict[str, Any]] = []
    missing = [phase for phase in required if phase not in latest]
    if missing:
        failures.append("missing eval rows for: " + ", ".join(missing))

    for phase in required:
        row = latest.get(phase, {})
        skill = _as_float(row, "skill_score")
        best = _as_float(row, "best_skill_score", skill)
        retention = _as_float(row, "retention", 1.0 if skill > 0.0 else 0.0)
        amnesia = _as_float(row, "amnesia", max(0.0, 1.0 - retention))
        idle = _as_float(row, "idle_rate")
        whiff = _as_float(row, "whiff_rate")
        trade = _as_float(row, "mean_damage_trade")

        row_failures: list[str] = []
        if phase != current_phase:
            if amnesia > float(args.amnesia_threshold):
                row_failures.append(f"amnesia {amnesia:.3f} > {float(args.amnesia_threshold):.3f}")
            if retention < float(args.min_retention):
                row_failures.append(f"retention {retention:.3f} < {float(args.min_retention):.3f}")

        current_threshold = phase_score_threshold(phase, getattr(args, "min_current_score", None))
        if phase == current_phase and skill < current_threshold:
            row_failures.append(f"current score {skill:.3f} < {current_threshold:.3f}")

        if idle > float(args.max_idle_rate):
            row_failures.append(f"idle {idle:.3f} > {float(args.max_idle_rate):.3f}")

        if phase in COMBAT_PHASES:
            if whiff > float(args.max_combat_whiff_rate):
                row_failures.append(f"whiff {whiff:.3f} > {float(args.max_combat_whiff_rate):.3f}")
            if not bool(args.no_combat_trade_gate) and trade < float(args.min_combat_damage_trade):
                row_failures.append(f"damage trade {trade:+.3f} < {float(args.min_combat_damage_trade):+.3f}")

        if row_failures:
            failures.append(f"{phase}: " + "; ".join(row_failures))

        table.append(
            {
                "phase": phase,
                "skill": skill,
                "best": best,
                "retention": retention,
                "amnesia": amnesia,
                "idle": idle,
                "whiff": whiff,
                "trade": trade,
                "current_threshold": current_threshold if phase == current_phase else 0.0,
                "status": "FAIL" if row_failures or phase in missing else "PASS",
            }
        )

    return len(failures) == 0, failures, table


def _demo_chain(phases: list[str]) -> str:
    demos = [DEMO_PATHS[phase] for phase in phases if phase in DEMO_PATHS]
    return ";".join(demos)


def recommend_actions(
    failures: list[str],
    table: list[dict[str, Any]],
    *,
    current_phase: str,
) -> list[str]:
    current_phase = str(current_phase).strip().lower()
    recommendations: list[str] = []
    failed_phases = [str(row["phase"]) for row in table if str(row.get("status")) == "FAIL"]
    prior_failed = [phase for phase in failed_phases if phase != current_phase]

    if any("missing eval rows" in failure for failure in failures):
        recommendations.append(
            "Rerun retention evaluation with all required phases before trusting the gate."
        )

    if prior_failed:
        demo_chain = _demo_chain(previous_phases(current_phase, include_current=True))
        phase_list = ",".join(previous_phases(current_phase, include_current=True))
        recommendations.append(
            "Skill collapse detected: stop advancing and run a rehearsal fine-tune with all demos so far."
        )
        recommendations.append(
            "Suggested rehearse command template: "
            f"python -m train.train_curriculum --phase {current_phase} "
            f"--resume train/models/llc_{current_phase}.zip "
            f"--timesteps 100000 --model-name llc_{current_phase}_rehearsal "
            f"--bc-demos-path \"{demo_chain}\" --anchor-kl-coef 0.06 --bc-loss-coef 0.12 "
            f"--eval-every-steps 25000 --eval-episodes 5 --eval-phases {phase_list} "
            "--retention-scores-path train/models/llc_retention_best.json --log-csv"
        )

    if any("idle " in failure for failure in failures):
        recommendations.append(
            "High idle rate: inspect the overlay, then add/refresh movement demos and raise entropy during the next short rehearsal."
        )

    if any("whiff " in failure for failure in failures):
        recommendations.append(
            "High whiff rate: rehearse spacing before combat, keep attack grounding active, and add combat demos with deliberate punish timing."
        )

    if any("damage trade" in failure for failure in failures):
        recommendations.append(
            "Negative damage trade: do not progress to HSP; collect better combat demos, verify damage extraction, and rehearse spacing/combat together."
        )

    if any("current score" in failure for failure in failures):
        recommendations.append(
            "Current phase score is below threshold: continue this phase with shorter eval intervals before adding new goals."
        )

    deduped: list[str] = []
    for item in recommendations:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _print_table(table: list[dict[str, Any]]) -> None:
    header = "phase                  skill   best    retain  amnesia idle    whiff   trade   cur_min gate"
    print(header)
    print("-" * len(header))
    for row in table:
        print(
            f"{str(row['phase']):20s} "
            f"{float(row['skill']):6.3f} "
            f"{float(row['best']):6.3f} "
            f"{float(row['retention']):7.3f} "
            f"{float(row['amnesia']):7.3f} "
            f"{float(row['idle']):6.3f} "
            f"{float(row['whiff']):7.3f} "
            f"{float(row['trade']):+7.3f} "
            f"{float(row.get('current_threshold', 0.0)):7.3f} "
            f"{str(row['status']):>4s}"
        )


def main() -> None:
    args = parse_args()
    passed, failures, table = evaluate_gate(args)
    _print_table(table)
    if passed:
        print("ADVANCE: retention gate passed.")
        raise SystemExit(0)
    print("STOP: rehearse/fix before advancing.")
    for failure in failures:
        print(f"- {failure}")
    recommendations = recommend_actions(failures, table, current_phase=str(args.phase))
    if recommendations:
        print("Recommended next actions:")
        for item in recommendations:
            print(f"- {item}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
