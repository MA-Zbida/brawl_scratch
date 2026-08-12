#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PHASES: tuple[str, ...] = (
    "combat_execution",
    "movement_fluency",
    "weapon_acquisition",
    "spacing_neutral",
    "recovery_mastery",
    "all_skills_llc",
)
CORE_PHASES: tuple[str, ...] = tuple(phase for phase in PHASES if phase != "all_skills_llc")

DEFAULT_MAX_STEPS: dict[str, int] = {
    "combat_execution": 120,
    "movement_fluency": 90,
    "weapon_acquisition": 140,
    "spacing_neutral": 180,
    "recovery_mastery": 240,
    "all_skills_llc": 240,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect heuristic BC demos for multiple LLC phases in one run")
    p.add_argument(
        "--phases",
        type=str,
        default="all",
        help="'all', 'core', or comma/semicolon-separated phase names",
    )
    p.add_argument("--episodes-per-phase", type=int, default=50)
    p.add_argument("--output-dir", type=str, default="train/models")
    p.add_argument("--delay", type=float, default=3.0)
    p.add_argument("--python", type=str, default=sys.executable or "python")
    p.add_argument("--max-collection-attempts", type=int, default=0)
    p.add_argument("--weapon-hold-steps", type=int, default=20)
    p.add_argument("--weapon-reset-max-steps", type=int, default=30)
    p.add_argument("--weapon-drop-grace-steps", type=int, default=3)
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    p.add_argument("--keep-going", action="store_true", help="Continue after a phase command fails")
    return p.parse_args()


def parse_phase_list(raw: str) -> list[str]:
    key = str(raw or "").strip().lower()
    if key in ("", "all", "curriculum"):
        return list(PHASES)
    if key == "core":
        return list(CORE_PHASES)

    phases = [part.strip().lower() for part in key.replace(";", ",").split(",") if part.strip()]
    unknown = [phase for phase in phases if phase not in PHASES]
    if unknown:
        raise ValueError(f"Unknown phase(s): {', '.join(unknown)}. Expected one of: {', '.join(PHASES)}")
    return phases


def command_for_phase(args: argparse.Namespace, phase: str) -> list[str]:
    output_dir = Path(str(args.output_dir))
    output_path = output_dir / f"{phase}_demos.npz"
    cmd = [
        str(args.python),
        "-m",
        "train.collect_bc_locomotion_demos",
        "--phase",
        phase,
        "--teacher",
        "heuristic",
        "--episodes",
        str(max(1, int(args.episodes_per_phase))),
        "--max-episode-steps",
        str(DEFAULT_MAX_STEPS[phase]),
        "--delay",
        str(max(0.0, float(args.delay))),
        "--output",
        str(output_path),
    ]
    if int(args.max_collection_attempts) > 0:
        cmd.extend(["--max-collection-attempts", str(int(args.max_collection_attempts))])
    if phase == "weapon_acquisition":
        cmd.extend(
            [
                "--weapon-hold-steps",
                str(max(1, int(args.weapon_hold_steps))),
                "--weapon-reset-max-steps",
                str(max(0, int(args.weapon_reset_max_steps))),
                "--weapon-drop-grace-steps",
                str(max(0, int(args.weapon_drop_grace_steps))),
            ]
        )
    return cmd


def main() -> int:
    args = parse_args()
    phases = parse_phase_list(args.phases)

    print("=" * 72)
    print("HEURISTIC CURRICULUM DEMO COLLECTION")
    print(f"Phases: {', '.join(phases)}")
    print(f"Episodes per phase: {max(1, int(args.episodes_per_phase))}")
    print(f"Output dir: {Path(str(args.output_dir)).as_posix()}")
    print("=" * 72)

    for idx, phase in enumerate(phases, start=1):
        cmd = command_for_phase(args, phase)
        printable = " ".join(cmd)
        print(f"\n[{idx}/{len(phases)}] {phase}")
        print(printable)

        if bool(args.dry_run):
            continue

        result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent)
        if result.returncode != 0:
            print(f"[{phase}] failed with exit code {result.returncode}")
            if not bool(args.keep_going):
                return int(result.returncode)

    print("\nHeuristic curriculum collection finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
