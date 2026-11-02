#!/usr/bin/env python
"""Run the recorded four-phase LLC curriculum as one fail-closed chain."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence
from brawl.paths import DATA, LEGACY_RUNS


CORE_PHASES: tuple[str, ...] = (
    "movement_fluency",
    "weapon_acquisition",
    "spacing_neutral",
    "combat_execution",
)

DEMO_FILENAMES: dict[str, str] = {
    "movement_fluency": "movement_fluency_demos.npz",
    "weapon_acquisition": "weapon_acquisition_demos.npz",
    "spacing_neutral": "spacing_neutral_demos_v2.npz",
    "combat_execution": "combat_execution_demos.npz",
}

MODEL_NAMES: dict[str, str] = {
    phase: f"llc_{phase}_3h" for phase in CORE_PHASES
}


@dataclass(frozen=True)
class RunStep:
    """One subprocess in the sequential curriculum."""

    name: str
    kind: str
    phase: str
    argv: tuple[str, ...]


def _bc_command(
    *,
    python: str,
    phase: str,
    demo: Path,
    resume: Path,
    output: Path,
    epochs: int,
) -> tuple[str, ...]:
    return (
        python,
        "-m",
        "brawl.pipelines.llc.pretrain_bc",
        "--phase",
        phase,
        "--demos",
        str(demo),
        "--resume",
        str(resume),
        "--epochs",
        str(epochs),
        "--batch-size",
        "512",
        "--learning-rate",
        "0.0002",
        "--entropy-coef",
        "0.001",
        "--goal-relabel",
        "--output",
        str(output),
        "--device",
        "cpu",
    )


def _ppo_command(
    *,
    python: str,
    phase: str,
    resume: Path,
    models_dir: Path,
    model_name: str,
    timesteps: int,
    demos: Sequence[Path],
    world_replay: Path,
    opponent_setup: str,
) -> tuple[str, ...]:
    # Every live phase records independently. The shared root groups the four
    # immutable sessions without making one corruptible multi-process log.
    return (
        python,
        "-m",
        "brawl.pipelines.llc.train_curriculum",
        "--phase",
        phase,
        "--resume",
        str(resume),
        "--save-dir",
        str(models_dir),
        "--model-name",
        model_name,
        "--timesteps",
        str(timesteps),
        "--n-steps",
        "512",
        "--batch-size",
        "256",
        "--max-episode-steps",
        "200",
        "--bc-demos-path",
        ";".join(str(path) for path in demos),
        "--seed",
        "42",
        "--device",
        "cpu",
        "--ent-coef",
        "0.01",
        "--log-csv",
        "--plot-every",
        "50",
        "--diag-report-every",
        "2048",
        "--eval-every-steps",
        "0",
        "--world-replay",
        str(world_replay),
        "--world-replay-shard-size",
        "4096",
        "--opponent-setup",
        opponent_setup,
    )


def _eval_command(
    *,
    python: str,
    phase: str,
    model: Path,
    phases: Sequence[str],
    models_dir: Path,
    episodes: int,
) -> tuple[str, ...]:
    return (
        python,
        "-m",
        "brawl.pipelines.llc.evaluate_retention",
        "--model",
        str(model),
        "--phase",
        phase,
        "--phases",
        ",".join(phases),
        "--episodes",
        str(episodes),
        "--max-episode-steps",
        "200",
        "--best-scores",
        str(models_dir / "retention_best_3h.json"),
        "--csv",
        str(models_dir / f"{MODEL_NAMES[phase]}_retention_eval.csv"),
        "--seed",
        "42",
        "--device",
        "cpu",
    )


def build_run_plan(
    *,
    python: str,
    models_dir: Path,
    demos_dir: Path,
    movement_init: Path,
    world_replay: Path,
    opponent_setup: str,
    movement_steps: int,
    weapon_steps: int,
    spacing_steps: int,
    combat_steps: int,
    bc_epochs: int = 8,
    eval_episodes: int = 2,
) -> list[RunStep]:
    """Build the immutable handoff chain without touching the filesystem."""
    step_budget = {
        "movement_fluency": movement_steps,
        "weapon_acquisition": weapon_steps,
        "spacing_neutral": spacing_steps,
        "combat_execution": combat_steps,
    }
    demos = {
        phase: demos_dir / DEMO_FILENAMES[phase] for phase in CORE_PHASES
    }
    final_models = {
        phase: models_dir / f"{MODEL_NAMES[phase]}.zip" for phase in CORE_PHASES
    }
    bc_models = {
        phase: models_dir / f"{MODEL_NAMES[phase]}_bc_init.zip"
        for phase in CORE_PHASES[1:]
    }

    plan = [
        RunStep(
            name="validate_demos",
            kind="validate",
            phase="all",
            argv=(
                python,
                "-m",
                "scripts.llc.validate_llc_demos",
                *(str(demos[phase]) for phase in CORE_PHASES),
                "--min-samples",
                "1",
                "--strict-warnings",
            ),
        )
    ]

    cumulative_demos: list[Path] = []
    resume = movement_init
    evaluated_phases: list[str] = []
    for index, phase in enumerate(CORE_PHASES):
        cumulative_demos.append(demos[phase])
        short_name = phase.split("_", maxsplit=1)[0]
        if index > 0:
            bc_output = bc_models[phase]
            plan.append(
                RunStep(
                    name=f"{short_name}_bc",
                    kind="bc",
                    phase=phase,
                    argv=_bc_command(
                        python=python,
                        phase=phase,
                        demo=demos[phase],
                        resume=resume,
                        output=bc_output,
                        epochs=bc_epochs,
                    ),
                )
            )
            resume = bc_output

        plan.append(
            RunStep(
                name=f"{short_name}_ppo",
                kind="ppo",
                phase=phase,
                argv=_ppo_command(
                    python=python,
                    phase=phase,
                    resume=resume,
                    models_dir=models_dir,
                    model_name=MODEL_NAMES[phase],
                    timesteps=step_budget[phase],
                    demos=cumulative_demos,
                    world_replay=world_replay,
                    opponent_setup=opponent_setup,
                ),
            )
        )
        resume = final_models[phase]
        evaluated_phases.append(phase)
        plan.append(
            RunStep(
                name=f"{short_name}_eval",
                kind="eval",
                phase=phase,
                argv=_eval_command(
                    python=python,
                    phase=phase,
                    model=resume,
                    phases=evaluated_phases,
                    models_dir=models_dir,
                    episodes=eval_episodes,
                ),
            )
        )
    return plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run movement -> weapon -> spacing -> combat as one recorded, "
            "fail-closed LLC checkpoint chain."
        )
    )
    parser.add_argument("--models-dir", type=Path, default=LEGACY_RUNS / "llc" / "easy_bot_v0")
    parser.add_argument("--demos-dir", type=Path, default=LEGACY_RUNS / "llc" / "recollected")
    parser.add_argument(
        "--movement-init",
        type=Path,
        default=LEGACY_RUNS / "llc" / "easy_bot_v0" / "llc_movement_fluency_masked_pilot.zip",
    )
    parser.add_argument(
        "--world-replay",
        type=Path,
        default=DATA / "world_replay" / "easy_bot_v0_3h",
        help="Root for four phase-specific world-model replay sessions.",
    )
    parser.add_argument("--opponent-setup", default="moving_non_attacking")
    parser.add_argument("--movement-steps", type=int, default=60_000)
    parser.add_argument("--weapon-steps", type=int, default=60_000)
    parser.add_argument("--spacing-steps", type=int, default=120_000)
    parser.add_argument("--combat-steps", type=int, default=120_000)
    parser.add_argument("--bc-epochs", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--start-delay", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _expected_outputs(plan: Sequence[RunStep]) -> list[Path]:
    outputs: list[Path] = []
    for step in plan:
        if step.kind == "ppo":
            save_dir = Path(step.argv[step.argv.index("--save-dir") + 1])
            name = step.argv[step.argv.index("--model-name") + 1]
            outputs.extend(
                [
                    save_dir / f"{name}.zip",
                    save_dir / f"{name}_steps.csv",
                    save_dir / f"{name}_episodes.csv",
                ]
            )
        elif step.kind == "bc":
            outputs.append(Path(step.argv[step.argv.index("--output") + 1]))
        elif step.kind == "eval":
            outputs.append(Path(step.argv[step.argv.index("--csv") + 1]))
    return outputs


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _display_command(argv: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    budgets = (
        args.movement_steps,
        args.weapon_steps,
        args.spacing_steps,
        args.combat_steps,
    )
    if any(value <= 0 for value in budgets):
        raise SystemExit("All phase step budgets must be positive.")
    if args.bc_epochs <= 0 or args.eval_episodes <= 0:
        raise SystemExit("BC epochs and evaluation episodes must be positive.")
    if not str(args.opponent_setup).strip():
        raise SystemExit("--opponent-setup must be a non-empty provenance label.")

    plan = build_run_plan(
        python=sys.executable,
        models_dir=args.models_dir,
        demos_dir=args.demos_dir,
        movement_init=args.movement_init,
        world_replay=args.world_replay,
        opponent_setup=args.opponent_setup,
        movement_steps=args.movement_steps,
        weapon_steps=args.weapon_steps,
        spacing_steps=args.spacing_steps,
        combat_steps=args.combat_steps,
        bc_epochs=args.bc_epochs,
        eval_episodes=args.eval_episodes,
    )

    required = [args.movement_init]
    required.extend(args.demos_dir / DEMO_FILENAMES[phase] for phase in CORE_PHASES)
    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"Required input artifacts are missing:\n{formatted}")

    manifest_path = args.models_dir / "unattended_core_3h_manifest.json"
    collisions = [path for path in _expected_outputs(plan) if path.exists()]
    if not args.dry_run and (manifest_path.exists() or collisions):
        existing = ([manifest_path] if manifest_path.exists() else []) + collisions
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise SystemExit(
            "Refusing to overwrite an earlier unattended run. Choose a new "
            f"--models-dir (and replay root). Existing artifacts:\n{formatted}"
        )

    print("=" * 72)
    print("UNATTENDED CORE LLC CURRICULUM")
    print("Phases   : movement -> weapon -> spacing -> combat (recovery deferred)")
    print(f"PPO steps: {sum(budgets):,}")
    print(f"Replay   : {args.world_replay} (enabled for every PPO phase)")
    print(f"Opponent : {args.opponent_setup}")
    print("Combat note: a non-attacking bot trains chase/offense, not defense.")
    print("Setup    : Wu Shang vs Mordex on Small Brawlhaven; disable system sleep.")
    print("=" * 72)
    for index, step in enumerate(plan, start=1):
        print(f"[{index:02d}/{len(plan):02d}] {step.name}: {_display_command(step.argv)}")

    if args.dry_run:
        print("DRY RUN: no subprocesses started and no files written.")
        return 0

    args.models_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "started_at_unix": started_at,
        "python": sys.executable,
        "world_replay": str(args.world_replay),
        "opponent_setup": args.opponent_setup,
        "step_budgets": dict(zip(CORE_PHASES, budgets, strict=True)),
        "steps": [],
    }
    _write_manifest(manifest_path, manifest)

    if args.start_delay > 0:
        print(f"Starting in {args.start_delay:.0f}s; focus Brawlhalla now.")
        time.sleep(args.start_delay)

    records = manifest["steps"]
    assert isinstance(records, list)
    try:
        for index, step in enumerate(plan, start=1):
            print(f"\n[{index:02d}/{len(plan):02d}] START {step.name}", flush=True)
            step_started = time.time()
            completed = subprocess.run(step.argv, check=False)
            record = {
                **asdict(step),
                "argv": list(step.argv),
                "started_at_unix": step_started,
                "finished_at_unix": time.time(),
                "returncode": completed.returncode,
            }
            records.append(record)
            _write_manifest(manifest_path, manifest)
            if completed.returncode != 0:
                manifest["status"] = "failed"
                manifest["failed_step"] = step.name
                manifest["finished_at_unix"] = time.time()
                _write_manifest(manifest_path, manifest)
                print(
                    f"STOP: {step.name} exited with code {completed.returncode}. "
                    "Completed checkpoints and replay shards were preserved.",
                    file=sys.stderr,
                )
                return completed.returncode
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["finished_at_unix"] = time.time()
        _write_manifest(manifest_path, manifest)
        print("Interrupted; completed checkpoints and replay shards were preserved.")
        return 130

    manifest["status"] = "complete"
    manifest["finished_at_unix"] = time.time()
    _write_manifest(manifest_path, manifest)
    print(f"COMPLETE: all four phases finished. Manifest: {manifest_path}")
    print(f"World-model replay root: {args.world_replay}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
