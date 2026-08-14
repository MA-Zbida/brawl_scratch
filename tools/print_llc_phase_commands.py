#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.retention import PHASE_ORDER, previous_phases


DEMO_BY_PHASE: dict[str, str] = {
    "recovery_mastery": "recovery_mastery_demos.npz",
    "movement_fluency": "movement_fluency_demos.npz",
    "weapon_acquisition": "weapon_acquisition_demos.npz",
    "spacing_neutral": "spacing_neutral_demos.npz",
    "combat_execution": "combat_execution_demos.npz",
}

DEFAULT_TIMESTEPS: dict[str, int] = {
    "recovery_mastery": 300_000,
    "movement_fluency": 300_000,
    "weapon_acquisition": 350_000,
    "spacing_neutral": 350_000,
    "combat_execution": 500_000,
    "all_skills_llc": 500_000,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print the exact LLC phase commands for the mastery ladder")
    p.add_argument("--phase", required=True, choices=list(PHASE_ORDER) + ["all"], help="Phase to print, or 'all'")
    p.add_argument("--models-dir", type=str, default="train/models")
    p.add_argument("--device", type=str, default="$device", help="Device literal to put in commands, e.g. cuda, cpu, or $device")
    p.add_argument(
        "--best-scores",
        type=str,
        default="train/models/llc_retention_best.json",
        help="Retention JSON path or Powershell variable",
    )
    p.add_argument("--python", type=str, default="python", help="Python command literal")
    p.add_argument("--bc-epochs", type=int, default=20)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--all-skills-eval-episodes", type=int, default=10)
    p.add_argument("--timesteps", type=int, default=0, help="Override timesteps for the printed phase(s)")
    p.add_argument("--plot", action="store_true", default=True)
    p.add_argument("--no-plot", dest="plot", action="store_false")
    return p.parse_args()


def _path(models_dir: str, filename: str) -> str:
    return str(Path(models_dir) / filename).replace("\\", "/")


def _model_path(models_dir: str, phase: str) -> str:
    return _path(models_dir, f"llc_{phase}.zip")


def _bc_init_path(models_dir: str, phase: str) -> str:
    return _path(models_dir, f"llc_{phase}_bc_init.zip")


def _demo_path(models_dir: str, phase: str) -> str:
    return _path(models_dir, DEMO_BY_PHASE[phase])


def _demo_chain(models_dir: str, phase: str) -> str:
    phases = previous_phases(phase, include_current=True)
    phases = [p for p in phases if p in DEMO_BY_PHASE]
    return ";".join(_demo_path(models_dir, p) for p in phases)


def _eval_phases_arg(phase: str) -> str:
    if phase == "all_skills_llc":
        return "all"
    return ",".join(previous_phases(phase, include_current=True))


def _previous_final_model(models_dir: str, phase: str) -> str:
    phases = list(PHASE_ORDER)
    idx = phases.index(phase)
    if idx <= 0:
        return ""
    return _model_path(models_dir, phases[idx - 1])


def _timesteps(phase: str, override: int) -> int:
    return int(override) if int(override) > 0 else int(DEFAULT_TIMESTEPS[phase])


def _report_path(phase: str) -> str:
    return str(Path("outputs") / f"llc_{phase}_run_report.md").replace("\\", "/")


def commands_for_phase(args: argparse.Namespace, phase: str) -> list[str]:
    phase = str(phase).strip().lower()
    py = str(args.python)
    models_dir = str(args.models_dir)
    device = str(args.device)
    best = str(args.best_scores)
    model_name = f"llc_{phase}"
    eval_phases = _eval_phases_arg(phase)
    eval_episodes = int(args.all_skills_eval_episodes if phase == "all_skills_llc" else args.eval_episodes)
    retention_csv = _path(models_dir, f"{model_name}_retention_eval.csv")
    commands: list[str] = []

    if phase != "all_skills_llc":
        commands.append(
            " ".join(
                [
                    py,
                    "tools/validate_llc_demos.py",
                    f"--phase {phase}",
                    "--min-samples 100",
                ]
            )
        )
        pretrain_parts = [
            py,
            "-m train.pretrain_bc_locomotion",
            f"--phase {phase}",
        ]
        prev_model = _previous_final_model(models_dir, phase)
        if prev_model:
            pretrain_parts.append(f"--resume {prev_model}")
        pretrain_parts.append(f"--demos {_demo_path(models_dir, phase)}")
        pretrain_parts.extend(
            [
                f"--epochs {int(args.bc_epochs)}",
                f"--output {_bc_init_path(models_dir, phase)}",
                f"--device {device}",
            ]
        )
        commands.append(" ".join(pretrain_parts))
        train_resume = _bc_init_path(models_dir, phase)
    else:
        commands.append(
            " ".join(
                [
                    py,
                    "tools/validate_llc_demos.py",
                    "--phase all",
                    "--min-samples 100",
                ]
            )
        )
        train_resume = _previous_final_model(models_dir, phase)

    train_parts = [
        py,
        "-m train.train_curriculum",
        f"--phase {phase}",
        f"--resume {train_resume}",
        f"--timesteps {_timesteps(phase, int(args.timesteps))}",
        f"--model-name {model_name}",
        f"--bc-demos-path \"{_demo_chain(models_dir, phase)}\"",
        "--log-csv",
        "--plot-every 10",
        "--eval-every-steps 25000",
        f"--eval-episodes {int(args.eval_episodes)}",
    ]
    if phase == "all_skills_llc":
        train_parts.append("--eval-phases all")
    else:
        train_parts.append("--eval-include-previous")
    train_parts.extend(
        [
            f"--retention-scores-path {best}",
            f"--device {device}",
        ]
    )
    commands.append(" ".join(train_parts))

    commands.append(
        " ".join(
            [
                py,
                "-m train.evaluate_retention",
                f"--model {_model_path(models_dir, phase)}",
                f"--phase {phase}",
                f"--phases {eval_phases}",
                f"--best-scores {best}",
                f"--episodes {eval_episodes}",
                f"--device {device}",
                f"--csv {retention_csv}",
            ]
        )
    )

    gate_parts = [
        py,
        "tools/check_llc_phase_gate.py",
        f"--eval-csv {retention_csv}",
        f"--phase {phase}",
    ]
    if phase in ("recovery_mastery", "all_skills_llc"):
        gate_parts.append(f"--phases {eval_phases}")
    commands.append(" ".join(gate_parts))

    if bool(args.plot):
        commands.append(
            " ".join(
                [
                    py,
                    "tools/plot_llc_diagnostics.py",
                    f"--steps-csv {_path(models_dir, f'{model_name}_steps.csv')}",
                    f"--episodes-csv {_path(models_dir, f'{model_name}_episodes.csv')}",
                    f"--eval-csv {_path(models_dir, f'{model_name}_eval.csv')}",
                    f"--prefix {model_name}",
                ]
            )
        )

    report_parts = [
        py,
        "tools/summarize_llc_run.py",
        f"--phase {phase}",
        f"--phases {eval_phases}",
        f"--eval-csv {retention_csv}",
        f"--models-dir {models_dir}",
        f"--prefix {model_name}",
        f"--out {_report_path(phase)}",
    ]
    commands.append(" ".join(report_parts))

    commands.append(
        " ".join(
            [
                py,
                "tools/record_llc_observation.py",
                f"--phase {phase}",
                "--approved yes",
                "--notes \"Describe what you saw\"",
            ]
        )
    )

    return commands


def main() -> None:
    args = parse_args()
    command_args = argparse.Namespace(**vars(args))
    phases = list(PHASE_ORDER) if args.phase == "all" else [str(args.phase).strip().lower()]
    print("$device = \"cuda\"  # or \"cpu\"")
    best_literal = str(args.best_scores).strip()
    if best_literal.startswith("$"):
        print(f"# {best_literal} is expected to be set already")
    else:
        print(f"$best = \"{best_literal}\"")
        command_args.best_scores = "$best"
    for phase in phases:
        print()
        print(f"# {phase}")
        for command in commands_for_phase(command_args, phase):
            print(command)
            print()


if __name__ == "__main__":
    main()
