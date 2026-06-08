#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.check_llc_phase_gate import evaluate_gate, recommend_actions
from tools.validate_llc_demos import default_paths_for_phase, validate_demo_archive


PLOT_SUFFIXES = (
    "retention_amnesia",
    "goal_family_errors",
    "goal_feature_traces",
    "goal_phase_spaces",
    "episode_health",
    "combat_precision",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write a Markdown report for an LLC phase run")
    p.add_argument("--phase", required=True, help="Current phase")
    p.add_argument("--eval-csv", required=True, help="Retention/eval CSV used for gate checking")
    p.add_argument("--phases", type=str, default="", help="'all' or comma/semicolon phase list. Default: current + previous")
    p.add_argument("--models-dir", type=str, default="train/models")
    p.add_argument("--prefix", type=str, default="", help="Plot/model prefix. Default: llc_<phase>")
    p.add_argument("--out", type=str, default="", help="Markdown output path")
    p.add_argument("--include-demo-check", action="store_true", default=True)
    p.add_argument("--no-include-demo-check", dest="include_demo_check", action="store_false")
    p.add_argument("--amnesia-threshold", type=float, default=0.15)
    p.add_argument("--min-retention", type=float, default=0.85)
    p.add_argument("--min-current-score", type=float, default=None, help="Override current phase skill threshold")
    p.add_argument("--max-idle-rate", type=float, default=0.45)
    p.add_argument("--max-combat-whiff-rate", type=float, default=0.80)
    p.add_argument("--min-combat-damage-trade", type=float, default=0.0)
    p.add_argument("--no-combat-trade-gate", action="store_true")
    return p.parse_args()


def _fmt_float(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def _plot_status(models_dir: str, prefix: str) -> list[tuple[str, Path, bool]]:
    base = Path(models_dir)
    rows: list[tuple[str, Path, bool]] = []
    for suffix in PLOT_SUFFIXES:
        path = base / f"{prefix}_{suffix}.png"
        rows.append((suffix, path, path.exists()))
    return rows


def _demo_phase_arg(phase: str) -> str:
    return "all" if str(phase).strip().lower() == "all_skills_llc" else str(phase).strip().lower()


def _demo_results(phase: str, models_dir: str) -> list[dict[str, Any]]:
    demo_phase = _demo_phase_arg(phase)
    paths = default_paths_for_phase(demo_phase, models_dir)
    results: list[dict[str, Any]] = []
    for path in paths:
        expected = "" if demo_phase == "all" else demo_phase
        results.append(validate_demo_archive(path, expected_phase=expected, min_samples=100))
    return results


def build_report(args: argparse.Namespace) -> str:
    phase = str(args.phase).strip().lower()
    prefix = str(args.prefix or f"llc_{phase}").strip()
    gate_args = argparse.Namespace(
        eval_csv=str(args.eval_csv),
        phase=phase,
        phases=str(args.phases or ""),
        amnesia_threshold=float(args.amnesia_threshold),
        min_retention=float(args.min_retention),
        min_current_score=args.min_current_score,
        max_idle_rate=float(args.max_idle_rate),
        max_combat_whiff_rate=float(args.max_combat_whiff_rate),
        min_combat_damage_trade=float(args.min_combat_damage_trade),
        no_combat_trade_gate=bool(args.no_combat_trade_gate),
    )
    passed, failures, table = evaluate_gate(gate_args)
    recommendations = recommend_actions(failures, table, current_phase=phase)

    lines: list[str] = []
    lines.append(f"# LLC Phase Report: `{phase}`")
    lines.append("")
    lines.append(f"Gate: **{'ADVANCE' if passed else 'STOP'}**")
    lines.append(f"Eval CSV: `{args.eval_csv}`")
    lines.append("")
    lines.append("## Phase Metrics")
    lines.append("")
    lines.append("| Phase | Skill | Current Min | Best | Retention | Amnesia | Idle | Whiff | Damage Trade | Gate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in table:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['phase']}`",
                    _fmt_float(row["skill"]),
                    _fmt_float(row.get("current_threshold", 0.0)),
                    _fmt_float(row["best"]),
                    _fmt_float(row["retention"]),
                    _fmt_float(row["amnesia"]),
                    _fmt_float(row["idle"]),
                    _fmt_float(row["whiff"]),
                    _fmt_float(row["trade"]),
                    str(row["status"]),
                ]
            )
            + " |"
        )

    if failures:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        for failure in failures:
            lines.append(f"- {failure}")

    if recommendations:
        lines.append("")
        lines.append("## Recommended Next Actions")
        lines.append("")
        for item in recommendations:
            lines.append(f"- {item}")

    if bool(args.include_demo_check):
        lines.append("")
        lines.append("## Demo Archive Check")
        lines.append("")
        demo_results = _demo_results(phase, str(args.models_dir))
        if not demo_results:
            lines.append("- No demo archives selected.")
        else:
            lines.append("| Status | Phase | Samples | Entropy | Idle | Attack | Goal Active | Path |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---|")
            for item in demo_results:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(item["status"]),
                            f"`{item['phase'] or 'unknown'}`",
                            str(int(item["samples"])),
                            _fmt_float(item["action_entropy"]),
                            _fmt_float(item["idle_rate"]),
                            _fmt_float(item["attack_rate"]),
                            _fmt_float(item["goal_active_ratio"]),
                            f"`{item['path']}`",
                        ]
                    )
                    + " |"
                )
                for err in item["errors"]:
                    lines.append(f"- Demo error `{item['path']}`: {err}")
                for warning in item["warnings"]:
                    lines.append(f"- Demo warning `{item['path']}`: {warning}")

    lines.append("")
    lines.append("## Plot Evidence")
    lines.append("")
    lines.append("| Plot | Found | Path |")
    lines.append("|---|---:|---|")
    for name, path, exists in _plot_status(str(args.models_dir), prefix):
        lines.append(f"| `{name}` | {'yes' if exists else 'no'} | `{path.as_posix()}` |")

    lines.append("")
    lines.append("## Manual Observation Notes")
    lines.append("")
    lines.append(
        f"Record approval with: `python tools/record_llc_observation.py --phase {phase} --approved yes --notes \"...\"`"
    )
    lines.append("")
    lines.append("- Movement collapse visible? `yes/no`")
    lines.append("- Recovery returns to stage reliably? `yes/no`")
    lines.append("- Weapon pickup looks intentional? `yes/no`")
    lines.append("- Neutral spacing avoids self-damage? `yes/no`")
    lines.append("- Combat produces positive damage trade without attack spam? `yes/no`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = build_report(args)
    out = Path(args.out) if args.out else Path("outputs") / f"llc_{str(args.phase).strip().lower()}_run_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
