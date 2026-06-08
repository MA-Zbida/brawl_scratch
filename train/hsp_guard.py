from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tools.check_llc_phase_gate import evaluate_gate


def hsp_readiness_failure(args: Any) -> str:
    if bool(getattr(args, "allow_legacy_hsp", False)):
        return ""

    csv_path = str(getattr(args, "llc_retention_csv", "") or "").strip()
    if not csv_path:
        return (
            "HSP is deferred until the LLC passes all retention gates. "
            "Pass --llc-retention-csv train/models/llc_all_skills_llc_retention_eval.csv "
            "after all_skills_llc succeeds, or use --allow-legacy-hsp to override knowingly."
        )

    if not Path(csv_path).exists():
        return f"LLC retention CSV not found: {csv_path}"

    gate_args = argparse.Namespace(
        eval_csv=csv_path,
        phase="all_skills_llc",
        phases="all",
        amnesia_threshold=float(getattr(args, "amnesia_threshold", 0.15)),
        min_retention=float(getattr(args, "min_retention", 0.85)),
        min_current_score=getattr(args, "min_current_score", None),
        max_idle_rate=float(getattr(args, "max_idle_rate", 0.45)),
        max_combat_whiff_rate=float(getattr(args, "max_combat_whiff_rate", 0.80)),
        min_combat_damage_trade=float(getattr(args, "min_combat_damage_trade", 0.0)),
        no_combat_trade_gate=bool(getattr(args, "no_combat_trade_gate", False)),
    )
    passed, failures, _ = evaluate_gate(gate_args)
    if passed:
        return ""
    return "LLC all-skills retention gate has not passed: " + "; ".join(failures)


def require_hsp_readiness(args: Any) -> None:
    failure = hsp_readiness_failure(args)
    if failure:
        raise SystemExit("[HSP guard] " + failure)
