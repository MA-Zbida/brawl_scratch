#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.retention import PHASE_ORDER


TRI_STATES = ("yes", "no", "na")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record manual visual approval for an LLC phase")
    p.add_argument("--phase", required=True, choices=list(PHASE_ORDER))
    p.add_argument("--approved", required=True, choices=["yes", "no"], help="Your visual approval for advancing")
    p.add_argument("--outputs-dir", type=str, default="outputs")
    p.add_argument("--notes", type=str, default="")
    p.add_argument("--movement-collapse-visible", choices=TRI_STATES, default="na")
    p.add_argument("--recovery-reliable", choices=TRI_STATES, default="na")
    p.add_argument("--weapon-intentional", choices=TRI_STATES, default="na")
    p.add_argument("--spacing-safe", choices=TRI_STATES, default="na")
    p.add_argument("--combat-clean", choices=TRI_STATES, default="na")
    return p.parse_args()


def observation_path(outputs_dir: str, phase: str) -> Path:
    return Path(outputs_dir) / f"llc_{str(phase).strip().lower()}_manual_observation.json"


def observation_markdown_path(outputs_dir: str, phase: str) -> Path:
    return Path(outputs_dir) / f"llc_{str(phase).strip().lower()}_manual_observation.md"


def _bad_visual_flags(checks: dict[str, str]) -> list[str]:
    bad: list[str] = []
    if checks.get("movement_collapse_visible") == "yes":
        bad.append("movement collapse visible")
    if checks.get("recovery_reliable") == "no":
        bad.append("recovery is not reliable")
    if checks.get("weapon_intentional") == "no":
        bad.append("weapon pickup does not look intentional")
    if checks.get("spacing_safe") == "no":
        bad.append("spacing does not look safe")
    if checks.get("combat_clean") == "no":
        bad.append("combat shows attack spam or bad trade")
    return bad


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    checks = {
        "movement_collapse_visible": str(args.movement_collapse_visible),
        "recovery_reliable": str(args.recovery_reliable),
        "weapon_intentional": str(args.weapon_intentional),
        "spacing_safe": str(args.spacing_safe),
        "combat_clean": str(args.combat_clean),
    }
    approved = str(args.approved).strip().lower() == "yes"
    bad_flags = _bad_visual_flags(checks)
    return {
        "phase": str(args.phase).strip().lower(),
        "observed_at": datetime.now().isoformat(timespec="seconds"),
        "approved": bool(approved),
        "checks": checks,
        "bad_visual_flags": bad_flags,
        "notes": str(args.notes or "").strip(),
    }


def build_markdown(payload: dict[str, Any]) -> str:
    checks = payload.get("checks", {})
    lines: list[str] = []
    lines.append(f"# Manual LLC Observation: `{payload.get('phase', 'unknown')}`")
    lines.append("")
    lines.append(f"Observed at: `{payload.get('observed_at', '')}`")
    lines.append(f"Approved to advance: **{'yes' if payload.get('approved') else 'no'}**")
    lines.append("")
    lines.append("| Check | Value |")
    lines.append("|---|---|")
    for key in (
        "movement_collapse_visible",
        "recovery_reliable",
        "weapon_intentional",
        "spacing_safe",
        "combat_clean",
    ):
        lines.append(f"| `{key}` | `{checks.get(key, 'na')}` |")
    bad_flags = list(payload.get("bad_visual_flags", []))
    if bad_flags:
        lines.append("")
        lines.append("## Bad Visual Flags")
        lines.append("")
        for item in bad_flags:
            lines.append(f"- {item}")
    notes = str(payload.get("notes", "")).strip()
    if notes:
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.append(notes)
    lines.append("")
    return "\n".join(lines)


def write_observation(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    payload = build_payload(args)
    json_path = observation_path(str(args.outputs_dir), str(args.phase))
    md_path = observation_markdown_path(str(args.outputs_dir), str(args.phase))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(build_markdown(payload), encoding="utf-8")
    return json_path, md_path, payload


def load_observation(outputs_dir: str, phase: str) -> dict[str, Any] | None:
    path = observation_path(outputs_dir, phase)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def observation_status(outputs_dir: str, phase: str) -> tuple[str, list[str]]:
    data = load_observation(outputs_dir, phase)
    path = observation_path(outputs_dir, phase)
    if data is None:
        if path.exists():
            return "FAIL", [f"Manual observation file is unreadable: {path.as_posix()}"]
        return "MISSING", [f"Manual observation missing: {path.as_posix()}"]
    if str(data.get("phase", "")).strip().lower() != str(phase).strip().lower():
        return "FAIL", [f"Manual observation phase mismatch in {path.as_posix()}"]
    if not bool(data.get("approved", False)):
        details = [f"Manual observation says do not advance: {path.as_posix()}"]
        for item in list(data.get("bad_visual_flags", [])):
            details.append(f"Visual issue: {item}")
        notes = str(data.get("notes", "")).strip()
        if notes:
            details.append(f"Notes: {notes}")
        return "FAIL", details
    bad_flags = list(data.get("bad_visual_flags", []))
    if bad_flags:
        return "WARN", [f"Approved, but visual flags were recorded: {', '.join(str(x) for x in bad_flags)}"]
    return "PASS", [f"Manual observation approved: {path.as_posix()}"]


def main() -> None:
    args = parse_args()
    json_path, md_path, payload = write_observation(args)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if not bool(payload.get("approved", False)):
        print("STOP: manual observation did not approve advancement.")
        raise SystemExit(2)
    bad_flags = list(payload.get("bad_visual_flags", []))
    if bad_flags:
        print("CHECK: approved, but visual flags were recorded:")
        for item in bad_flags:
            print(f"- {item}")
        raise SystemExit(1)
    print("PASS: manual observation approved advancement.")


if __name__ == "__main__":
    main()
