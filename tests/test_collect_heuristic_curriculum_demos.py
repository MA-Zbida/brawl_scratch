from __future__ import annotations

import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from train.collect_heuristic_curriculum_demos import PHASES, command_for_phase, parse_phase_list


def test_direct_script_help_resolves_repo_imports() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "train" / "collect_heuristic_curriculum_demos.py"),
            "--help",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _args(**overrides) -> Namespace:
    data = {
        "python": "python",
        "episodes_per_phase": 50,
        "output_dir": "train/models",
        "delay": 0.0,
        "max_collection_attempts": 0,
        "min_episode_steps": 4,
        "weapon_hold_steps": 20,
        "weapon_reset_max_steps": 30,
        "weapon_drop_grace_steps": 3,
    }
    data.update(overrides)
    return Namespace(**data)


def test_parse_phase_list_all_returns_curriculum_phases() -> None:
    assert parse_phase_list("all") == list(PHASES)


def test_default_collection_order_defers_recovery_until_after_combat() -> None:
    assert PHASES == (
        "movement_fluency",
        "weapon_acquisition",
        "spacing_neutral",
        "combat_execution",
        "recovery_mastery",
        "all_skills_llc",
    )


def test_parse_phase_list_core_skips_consolidation_phase() -> None:
    phases = parse_phase_list("core")

    assert "all_skills_llc" not in phases
    assert "combat_execution" in phases


def test_parse_phase_list_accepts_comma_and_semicolon() -> None:
    phases = parse_phase_list("movement_fluency; weapon_acquisition,combat_execution")

    assert phases == ["movement_fluency", "weapon_acquisition", "combat_execution"]


def test_command_for_phase_runs_heuristic_single_phase_collector() -> None:
    cmd = command_for_phase(_args(), "weapon_acquisition")
    joined = " ".join(cmd)

    assert "python -m train.collect_bc_locomotion_demos" in joined
    assert "--phase weapon_acquisition" in joined
    assert "--teacher heuristic" in joined
    assert "--episodes 50" in joined
    assert "--output train\\models\\weapon_acquisition_demos.npz" in joined or "--output train/models/weapon_acquisition_demos.npz" in joined
    assert "--weapon-hold-steps 20" in joined
    assert "--weapon-reset-max-steps 30" in joined
    assert "--weapon-drop-grace-steps 3" in joined
    assert "--min-episode-steps 4" in joined


def test_command_for_phase_can_add_attempt_budget() -> None:
    cmd = command_for_phase(_args(max_collection_attempts=123), "combat_execution")
    joined = " ".join(cmd)

    assert "--max-collection-attempts 123" in joined
