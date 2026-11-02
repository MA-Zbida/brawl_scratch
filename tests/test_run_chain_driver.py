"""Regression tests for the resumable PowerShell chain driver."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT = Path("scripts/llc/easy_bot_v0/run_chain.ps1").resolve()


def _run_driver(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    state = tmp_path / "state"
    logs = tmp_path / "logs"
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-StateRoot",
            str(state),
            "-LogRoot",
            str(logs),
            *args,
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )


def test_driver_uses_exit_codes_and_utf8_native_logging() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert '$ErrorActionPreference = "Continue"' in text
    assert "$global:LASTEXITCODE = 0" in text
    assert "UTF8Encoding" in text
    assert "RedirectStandardOutput" in text
    assert "RedirectStandardError" in text
    assert "Tee-Object" not in text


def test_checkpoint_sidecars_are_bound_as_complete_paths() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    # In PowerShell argument mode, `$model + ".metadata.json"` was split while
    # binding the string-array parameter. The postcondition consequently looked
    # for the bare relative path `.metadata.json` after BC had saved successfully.
    assert '+ ".metadata.json"' not in text
    assert "$SpacingBcArtifacts" in text
    assert "$CombatBcArtifacts" in text


def test_checkpoint_artifact_harness_records_zip_and_full_sidecar_path(tmp_path: Path) -> None:
    result = _run_driver(tmp_path, "-HarnessProbe", "artifacts")

    assert result.returncode == 0, result.stdout + result.stderr
    marker = tmp_path / "state" / "production" / "artifact_binding.done.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    names = [Path(path).name for path in payload["artifacts"]]
    assert names == ["probe.zip", "probe.zip.metadata.json"]


def test_postcondition_recovery_adopts_only_outputs_newer_than_dependency(tmp_path: Path) -> None:
    result = _run_driver(tmp_path, "-HarnessProbe", "recovery")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "HARNESS_RECOVERY_READY" in result.stdout
    marker = tmp_path / "state" / "production" / "recovered_stage.done.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["source"] == "recovered_after_postcondition_failure"


def test_stderr_warning_does_not_abort_and_is_logged(tmp_path: Path) -> None:
    result = _run_driver(tmp_path, "-HarnessProbe", "warning")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "HARNESS_WARNING_CONTINUED" in result.stdout
    log = (tmp_path / "logs" / "harness.log").read_text(encoding="utf-8")
    assert "PROBE_WARNING_ON_STDERR" in log
    assert "HARNESS_WARNING_CONTINUED" in log


def test_nonzero_exit_aborts_with_stage_and_traceback_in_log(tmp_path: Path) -> None:
    result = _run_driver(tmp_path, "-HarnessProbe", "failure")

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "fault_nonzero" in combined
    log = (tmp_path / "logs" / "harness.log").read_text(encoding="utf-8")
    assert "TRACEBACK_SENTINEL" in log
    assert "fault_nonzero" in log


def test_from_spacing_skips_weapon_and_recovery_collection(tmp_path: Path) -> None:
    result = _run_driver(tmp_path, "--from", "spacing", "-DryRun")

    assert result.returncode == 0, result.stdout + result.stderr
    plan = [line for line in result.stdout.splitlines() if line.startswith("[PLAN]")]
    assert any("spacing_collect" in line for line in plan)
    assert any("combat_collect" in line for line in plan)
    assert all("weapon_" not in line for line in plan)
    assert all("recovery_collect" not in line for line in plan)


def test_smoke_plan_uses_tiny_budgets_and_never_collects_recovery(tmp_path: Path) -> None:
    result = _run_driver(tmp_path, "--smoke", "-DryRun")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "episodes=2 bc_epochs=1 ppo_steps=512 retention_episodes=1" in result.stdout
    plan = [line for line in result.stdout.splitlines() if line.startswith("[PLAN]")]
    assert any("recovery_bc" in line for line in plan)
    assert all("recovery_collect" not in line for line in plan)


def test_spacing_only_range_stops_before_combat_and_overrides_timesteps(tmp_path: Path) -> None:
    result = _run_driver(
        tmp_path,
        "--from",
        "spacing_bc",
        "--to",
        "spacing",
        "--timesteps",
        "1234",
        "-DryRun",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ppo_steps=1234" in result.stdout
    plan = [line for line in result.stdout.splitlines() if line.startswith("[PLAN]")]
    assert plan == [
        "[PLAN] preflight",
        "[PLAN] spacing_bc",
        "[PLAN] spacing_ppo",
        "[PLAN] spacing_retention",
    ]


def test_combat_only_range_contains_the_complete_combat_phase(tmp_path: Path) -> None:
    result = _run_driver(tmp_path, "--from", "combat", "--to", "combat", "-DryRun")

    assert result.returncode == 0, result.stdout + result.stderr
    plan = [line for line in result.stdout.splitlines() if line.startswith("[PLAN]")]
    assert plan == [
        "[PLAN] preflight",
        "[PLAN] combat_collect",
        "[PLAN] combat_bc",
        "[PLAN] combat_ppo",
        "[PLAN] combat_retention",
    ]


def test_reversed_stage_range_is_rejected(tmp_path: Path) -> None:
    result = _run_driver(tmp_path, "--from", "combat", "--to", "spacing", "-DryRun")

    assert result.returncode != 0
    assert "comes after --to" in result.stdout + result.stderr
