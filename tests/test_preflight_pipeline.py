"""Fail-fast checks for the live LLC pipeline contract."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.llc.preflight_pipeline import (
    PreflightFailure,
    exercise_environment_matrix,
    exercise_sequential_lifecycle,
    validate_allowed_actions,
    validate_checkpoint_references,
    validate_demo_references,
)
from brawl.pipelines.llc.stage_common import default_env_config


class _ClosableEnv:
    def __init__(self, *, fail_close: bool = False) -> None:
        self.closed = False
        self.fail_close = fail_close

    def close(self) -> None:
        self.closed = True
        if self.fail_close:
            raise RuntimeError("close exploded")


def test_live_training_defaults_keep_the_skewed_classifier_disabled() -> None:
    assert default_env_config(max_episode_steps=200).weapon_classifier_enabled is False


def test_environment_matrix_builds_and_closes_every_phase_in_both_modes() -> None:
    built: list[tuple[str, bool, _ClosableEnv]] = []

    def factory(phase: str, classifier_enabled: bool) -> _ClosableEnv:
        env = _ClosableEnv()
        built.append((phase, classifier_enabled, env))
        return env

    exercise_environment_matrix(
        ("movement_fluency", "weapon_acquisition"),
        classifier_modes=(False, True),
        environment_factory=factory,
    )

    assert [(phase, enabled) for phase, enabled, _ in built] == [
        ("movement_fluency", False),
        ("movement_fluency", True),
        ("weapon_acquisition", False),
        ("weapon_acquisition", True),
    ]
    assert all(env.closed for _, _, env in built)


def test_environment_failure_names_phase_and_classifier_mode() -> None:
    def factory(phase: str, classifier_enabled: bool) -> _ClosableEnv:
        raise RuntimeError("camera singleton collision")

    with pytest.raises(PreflightFailure, match=r"spacing_neutral.*classifier=off.*camera singleton"):
        exercise_environment_matrix(
            ("spacing_neutral",),
            classifier_modes=(False,),
            environment_factory=factory,
        )


def test_three_sequential_environments_are_closed() -> None:
    built: list[_ClosableEnv] = []

    def factory() -> _ClosableEnv:
        env = _ClosableEnv()
        built.append(env)
        return env

    exercise_sequential_lifecycle(factory, count=3)

    assert len(built) == 3
    assert all(env.closed for env in built)


def test_allowed_actions_fail_closed_on_empty_or_out_of_range() -> None:
    specs = {
        "empty": SimpleNamespace(allowed_actions=()),
        "bad": SimpleNamespace(allowed_actions=(0, 27)),
    }

    with pytest.raises(PreflightFailure, match=r"allowed_actions:empty.*non-empty"):
        validate_allowed_actions(("empty",), lambda phase: specs[phase], action_dim=27)
    with pytest.raises(PreflightFailure, match=r"allowed_actions:bad.*27"):
        validate_allowed_actions(("bad",), lambda phase: specs[phase], action_dim=27)


def test_checkpoint_contract_loads_archive_and_checks_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "skill.zip"
    checkpoint.write_bytes(b"archive")
    sidecar = checkpoint.with_name(checkpoint.name + ".metadata.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy_type": "goal_conditioned_skill",
                "goal_conditioning": True,
                "observation_dim": 168,
                "action_dim": 27,
                "goal_dim": 11,
            }
        ),
        encoding="utf-8",
    )
    loaded: list[Path] = []

    def loader(path: Path):
        loaded.append(path)
        return SimpleNamespace(
            observation_space=SimpleNamespace(shape=(168,)),
            action_space=SimpleNamespace(n=27),
        )

    validate_checkpoint_references(
        [
            {
                "name": "movement_base",
                "path": str(checkpoint),
                "role": "goal_conditioned_skill",
                "observation_dim": 168,
            }
        ],
        repo_root=tmp_path,
        checkpoint_loader=loader,
    )

    assert loaded == [checkpoint]


def test_checkpoint_contract_reports_the_named_reference(tmp_path: Path) -> None:
    with pytest.raises(PreflightFailure, match=r"checkpoint:movement_base.*missing"):
        validate_checkpoint_references(
            [
                {
                    "name": "movement_base",
                    "path": "missing.zip",
                    "role": "goal_conditioned_skill",
                    "observation_dim": 168,
                }
            ],
            repo_root=tmp_path,
            checkpoint_loader=lambda path: None,
        )


def test_demo_contract_validates_each_named_archive(tmp_path: Path) -> None:
    demo = tmp_path / "recovery.npz"
    demo.write_bytes(b"demo")
    calls: list[tuple[Path, str]] = []

    def validator(path: Path, *, expected_phase: str, min_samples: int):
        calls.append((path, expected_phase))
        return {"status": "PASS", "errors": [], "warnings": []}

    validate_demo_references(
        [
            {
                "phase": "recovery_mastery",
                "path": str(demo),
                "min_samples": 100,
            }
        ],
        repo_root=tmp_path,
        demo_validator=validator,
    )

    assert calls == [(demo, "recovery_mastery")]


def test_demo_contract_rejects_validator_failures(tmp_path: Path) -> None:
    demo = tmp_path / "spacing.npz"
    demo.write_bytes(b"demo")

    def validator(path: Path, *, expected_phase: str, min_samples: int):
        return {"status": "FAIL", "errors": ["bad goal rows"], "warnings": []}

    with pytest.raises(PreflightFailure, match=r"demo:spacing_neutral.*bad goal rows"):
        validate_demo_references(
            [{"phase": "spacing_neutral", "path": str(demo)}],
            repo_root=tmp_path,
            demo_validator=validator,
        )
