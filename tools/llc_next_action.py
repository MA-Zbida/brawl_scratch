#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.check_llc_phase_gate import evaluate_gate, recommend_actions
from tools.print_llc_phase_commands import (
    DEMO_BY_PHASE,
    _bc_init_path,
    _eval_phases_arg,
    _model_path,
    commands_for_phase,
)
from tools.record_llc_observation import observation_status
from tools.validate_llc_demos import validate_demo_archive
from train.retention import PHASE_ORDER, previous_phases


COLLECT_COMMANDS: dict[str, str] = {
    "recovery_mastery": (
        "python -m train.collect_bc_locomotion_demos --phase recovery_mastery "
        "--episodes 20 --max-episode-steps 120"
    ),
    "movement_fluency": (
        "python -m train.collect_bc_locomotion_demos --phase movement_fluency "
        "--episodes 40 --max-episode-steps 90 --move-mouse-to-goal"
    ),
    "weapon_acquisition": (
        "python -m train.collect_bc_locomotion_demos --phase weapon_acquisition "
        "--episodes 30 --max-episode-steps 140"
    ),
    "spacing_neutral": (
        "python -m train.collect_bc_locomotion_demos --phase spacing_neutral "
        "--episodes 30 --max-episode-steps 180"
    ),
    "combat_execution": (
        "python -m train.collect_bc_locomotion_demos --phase combat_execution "
        "--episodes 50 --max-episode-steps 240"
    ),
}

PERCEPTION_COMMAND = (
    "python tools/debug_observation_overlay.py --phase movement_fluency "
    "--show --max-steps 1000 --yolo-every 1"
)


@dataclass
class PhaseState:
    phase: str
    demo: str = "NA"
    bc_init: str = "NA"
    model: str = "MISSING"
    retention_eval: str = "MISSING"
    gate: str = "MISSING"
    plots: str = "MISSING"
    report: str = "MISSING"
    manual: str = "MISSING"


@dataclass
class Advice:
    title: str
    next_phase: str
    next_kind: str
    command: str
    details: list[str]
    states: list[PhaseState]
    exit_code: int = 0

    def render(self) -> str:
        lines: list[str] = []
        lines.append(f"NEXT: {self.title}")
        if self.next_phase:
            lines.append(f"Phase: {self.next_phase}")
        if self.command:
            lines.append("")
            lines.append("Command:")
            lines.append(self.command)
        if self.details:
            lines.append("")
            lines.append("Details:")
            for item in self.details:
                lines.append(f"- {item}")
        if self.states:
            lines.append("")
            lines.append("Status:")
            lines.append("phase                  demo    bc_init model   eval    gate    plots   report  manual")
            lines.append("------------------------------------------------------------------------------------")
            for state in self.states:
                lines.append(
                    f"{state.phase:20s} "
                    f"{state.demo:7s} "
                    f"{state.bc_init:7s} "
                    f"{state.model:7s} "
                    f"{state.retention_eval:7s} "
                    f"{state.gate:7s} "
                    f"{state.plots:7s} "
                    f"{state.report:7s} "
                    f"{state.manual:7s}"
                )
        return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect LLC artifacts and print the next safe manual training action")
    p.add_argument("--phase", type=str, default="auto", choices=["auto", *list(PHASE_ORDER)])
    p.add_argument("--models-dir", type=str, default="train/models")
    p.add_argument("--outputs-dir", type=str, default="outputs")
    p.add_argument("--device", type=str, default="$device")
    p.add_argument("--best-scores", type=str, default="train/models/llc_retention_best.json")
    p.add_argument("--python", type=str, default="python")
    p.add_argument("--bc-epochs", type=int, default=20)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--all-skills-eval-episodes", type=int, default=10)
    p.add_argument("--timesteps", type=int, default=0)
    p.add_argument("--min-samples", type=int, default=100)
    p.add_argument("--allow-demo-warnings", action="store_true")
    p.add_argument("--no-require-plots", dest="require_plots", action="store_false")
    p.add_argument("--no-require-report", dest="require_report", action="store_false")
    p.add_argument("--no-require-manual-approval", dest="require_manual_approval", action="store_false")
    p.set_defaults(require_plots=True, require_report=True, require_manual_approval=True)
    return p.parse_args()


def _phase_commands(args: argparse.Namespace, phase: str) -> list[str]:
    command_args = argparse.Namespace(
        phase=phase,
        models_dir=str(args.models_dir),
        device=str(args.device),
        best_scores=str(args.best_scores),
        python=str(args.python),
        bc_epochs=int(args.bc_epochs),
        eval_episodes=int(args.eval_episodes),
        all_skills_eval_episodes=int(args.all_skills_eval_episodes),
        timesteps=int(args.timesteps),
        plot=True,
    )
    return commands_for_phase(command_args, phase)


def _find_command(commands: list[str], token: str) -> str:
    for command in commands:
        if token in command:
            return command
    return ""


def _demo_path(models_dir: str, phase: str) -> Path:
    return Path(models_dir) / DEMO_BY_PHASE[phase]


def _retention_csv(models_dir: str, phase: str) -> Path:
    return Path(models_dir) / f"llc_{phase}_retention_eval.csv"


def _report_path(outputs_dir: str, phase: str) -> Path:
    return Path(outputs_dir) / f"llc_{phase}_run_report.md"


def _plot_paths(models_dir: str, phase: str) -> list[Path]:
    prefix = f"llc_{phase}"
    suffixes = (
        "retention_amnesia",
        "goal_family_errors",
        "goal_feature_traces",
        "goal_phase_spaces",
        "episode_health",
        "combat_precision",
    )
    return [Path(models_dir) / f"{prefix}_{suffix}.png" for suffix in suffixes]


def _plots_status(models_dir: str, phase: str) -> str:
    paths = _plot_paths(models_dir, phase)
    found = sum(1 for path in paths if path.exists())
    if found == len(paths):
        return "PASS"
    if found > 0:
        return "PARTIAL"
    return "MISSING"


def _demo_status(args: argparse.Namespace, phase: str) -> tuple[str, list[str]]:
    path = _demo_path(str(args.models_dir), phase)
    if not path.exists():
        return "MISSING", [f"Demo archive missing: {path.as_posix()}"]
    result = validate_demo_archive(
        path,
        expected_phase=phase,
        min_samples=int(args.min_samples),
        strict_warnings=False,
    )
    status = str(result["status"])
    details = [f"Demo {path.as_posix()}: {status}, samples={int(result['samples'])}"]
    for err in result["errors"]:
        details.append(f"Demo error: {err}")
    for warning in result["warnings"]:
        details.append(f"Demo warning: {warning}")
    return status, details


def _gate_status(args: argparse.Namespace, phase: str) -> tuple[str, list[str], list[dict[str, Any]]]:
    csv_path = _retention_csv(str(args.models_dir), phase)
    if not csv_path.exists():
        return "MISSING", [f"Retention eval CSV missing: {csv_path.as_posix()}"], []
    gate_args = argparse.Namespace(
        eval_csv=str(csv_path),
        phase=phase,
        phases=_eval_phases_arg(phase) if phase in ("recovery_mastery", "all_skills_llc") else "",
        amnesia_threshold=0.15,
        min_retention=0.85,
        min_current_score=None,
        max_idle_rate=0.45,
        max_combat_whiff_rate=0.80,
        min_combat_damage_trade=0.0,
        no_combat_trade_gate=False,
    )
    passed, failures, table = evaluate_gate(gate_args)
    if passed:
        return "PASS", [f"Gate passed for {phase}."], table
    return "FAIL", failures, table


def _state_for_phase(args: argparse.Namespace, phase: str) -> PhaseState:
    models_dir = str(args.models_dir)
    state = PhaseState(phase=phase)
    if phase in DEMO_BY_PHASE:
        demo_status, _ = _demo_status(args, phase)
        state.demo = demo_status
        state.bc_init = "PASS" if Path(_bc_init_path(models_dir, phase)).exists() else "MISSING"
    else:
        state.demo = "PASS" if all(_demo_path(models_dir, p).exists() for p in DEMO_BY_PHASE) else "MISSING"
        state.bc_init = "NA"
    state.model = "PASS" if Path(_model_path(models_dir, phase)).exists() else "MISSING"
    state.retention_eval = "PASS" if _retention_csv(models_dir, phase).exists() else "MISSING"
    state.gate, _, _ = _gate_status(args, phase) if state.retention_eval == "PASS" else ("MISSING", [], [])
    state.plots = _plots_status(models_dir, phase)
    state.report = "PASS" if _report_path(str(args.outputs_dir), phase).exists() else "MISSING"
    state.manual, _ = observation_status(str(args.outputs_dir), phase)
    return state


def _scan_phases(args: argparse.Namespace) -> list[str]:
    if str(args.phase) == "auto":
        return list(PHASE_ORDER)
    return previous_phases(str(args.phase), include_current=True)


def build_advice(args: argparse.Namespace) -> Advice:
    phases = _scan_phases(args)
    states: list[PhaseState] = []

    any_demo_exists = any(_demo_path(str(args.models_dir), p).exists() for p in DEMO_BY_PHASE)
    if not any_demo_exists and str(args.phase) == "auto":
        first_phase = PHASE_ORDER[0]
        state = _state_for_phase(args, first_phase)
        states.append(state)
        return Advice(
            title="sanity-check perception before collecting demos",
            next_phase=first_phase,
            next_kind="perception",
            command=PERCEPTION_COMMAND,
            details=[
                "If this is a fresh environment, run: python -m pip install -r requirements-llc.txt",
                "First run: python tools/llc_preflight.py --device cuda",
                "No demo archives were found. Verify boxes, damage, stocks, weapon state, and relative positions before recording anchors.",
                "After perception looks correct, collect the first demo archive with: "
                + COLLECT_COMMANDS[first_phase],
            ],
            states=states,
        )

    for phase in phases:
        state = _state_for_phase(args, phase)
        states.append(state)
        commands = _phase_commands(args, phase)

        if phase in DEMO_BY_PHASE:
            demo_status, demo_details = _demo_status(args, phase)
            if demo_status == "MISSING":
                return Advice(
                    title="collect missing BC demos",
                    next_phase=phase,
                    next_kind="collect_demos",
                    command=COLLECT_COMMANDS[phase],
                    details=demo_details,
                    states=states,
                )
            if demo_status == "FAIL" or (demo_status == "WARN" and not bool(args.allow_demo_warnings)):
                return Advice(
                    title="fix weak BC demo archive before training",
                    next_phase=phase,
                    next_kind="validate_demos",
                    command=_find_command(commands, "tools/validate_llc_demos.py"),
                    details=demo_details + ["Recollect this phase if the warning reflects bad behavior, high idle, or missing active goals."],
                    states=states,
                    exit_code=2,
                )

            bc_path = Path(_bc_init_path(str(args.models_dir), phase))
            if not bc_path.exists():
                return Advice(
                    title="BC-pretrain current phase",
                    next_phase=phase,
                    next_kind="bc_pretrain",
                    command=_find_command(commands, "train.pretrain_bc_locomotion"),
                    details=["Use the validated demo archive to initialize the LLC before PPO fine-tuning."],
                    states=states,
                )

        model_path = Path(_model_path(str(args.models_dir), phase))
        if not model_path.exists():
            return Advice(
                title="PPO fine-tune current phase with replay, BC, and anchoring",
                next_phase=phase,
                next_kind="ppo_train",
                command=_find_command(commands, "train.train_curriculum"),
                details=[
                    "This command includes CSV logging, periodic eval, replay anchoring, and all demos trained so far.",
                    f"While it runs, monitor collapse signals in another PowerShell window: python tools/llc_live_monitor.py --phase {phase}",
                ],
                states=states,
            )

        retention_path = _retention_csv(str(args.models_dir), phase)
        if not retention_path.exists():
            return Advice(
                title="run retention evaluation",
                next_phase=phase,
                next_kind="evaluate_retention",
                command=_find_command(commands, "train.evaluate_retention"),
                details=["Evaluate current plus previous phases before deciding whether to advance."],
                states=states,
            )

        gate_status, failures, table = _gate_status(args, phase)
        if gate_status != "PASS":
            recommendations = recommend_actions(failures, table, current_phase=phase)
            return Advice(
                title="STOP: retention gate failed",
                next_phase=phase,
                next_kind="gate_failed",
                command=_find_command(commands, "tools/check_llc_phase_gate.py"),
                details=failures + recommendations,
                states=states,
                exit_code=2,
            )

        if bool(args.require_plots) and _plots_status(str(args.models_dir), phase) != "PASS":
            return Advice(
                title="plot diagnostics for visual inspection",
                next_phase=phase,
                next_kind="plot",
                command=_find_command(commands, "tools/plot_llc_diagnostics.py"),
                details=["Inspect retention/amnesia, goal-family errors, episode health, and combat precision plots before advancing."],
                states=states,
            )

        if bool(args.require_report) and not _report_path(str(args.outputs_dir), phase).exists():
            return Advice(
                title="write phase report and manual observation notes",
                next_phase=phase,
                next_kind="report",
                command=_find_command(commands, "tools/summarize_llc_run.py"),
                details=["Use the report to record your visual judgment next to the numeric gate."],
                states=states,
            )

        if bool(args.require_manual_approval):
            manual_status, manual_details = observation_status(str(args.outputs_dir), phase)
            if manual_status != "PASS":
                return Advice(
                    title="record manual visual approval",
                    next_phase=phase,
                    next_kind="manual_observation",
                    command=(
                        f"python tools/record_llc_observation.py --phase {phase} --approved yes "
                        "--notes \"Describe what you saw\""
                    ),
                    details=manual_details
                    + [
                        "Only approve if your live observation agrees with the numeric gate: no collapsed movement, reliable recovery, intentional weapon pickup, safe spacing, and clean combat for the relevant phase.",
                    ],
                    states=states,
                    exit_code=2 if manual_status == "FAIL" else 0,
                )

    return Advice(
        title="LLC ladder artifacts are complete; inspect final report before HSP",
        next_phase="all_skills_llc",
        next_kind="llc_complete",
        command=(
            "python -m train.train_phase3_hsp --llc train/models/llc_all_skills_llc.zip "
            "--llc-retention-csv train/models/llc_all_skills_llc_retention_eval.csv"
        ),
        details=[
            "Run HSP only if all reports and your live observation agree that movement, recovery, weapon pickup, spacing, and combat did not collapse.",
        ],
        states=states,
    )


def main() -> None:
    args = parse_args()
    advice = build_advice(args)
    print(advice.render())
    raise SystemExit(int(advice.exit_code))


if __name__ == "__main__":
    main()
