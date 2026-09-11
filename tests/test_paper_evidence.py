from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.paper_evidence import (
    PRIMARY_PHASES,
    build_evidence,
    build_retention_matrices,
    load_learning_series,
    summarize_demo,
)


EPISODE_FIELDS = (
    "episode",
    "return",
    "length",
    "mean_goal_error",
    "episode_success",
)

RETENTION_FIELDS = (
    "phase",
    "episodes",
    "skill_score",
    "retention",
)


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _episode_rows() -> list[dict[str, object]]:
    return [
        {"episode": 1, "return": 1.0, "length": 2, "mean_goal_error": 0.4, "episode_success": 0},
        {"episode": 2, "return": 2.0, "length": 3, "mean_goal_error": 0.2, "episode_success": 1},
        {"episode": 3, "return": 3.0, "length": 4, "mean_goal_error": 0.1, "episode_success": 1},
    ]


def _write_demo(path: Path, *, phase: str = "movement_fluency") -> None:
    actions = np.asarray([0, 1, 10, 18, 26, 10], dtype=np.int64)
    dones = np.asarray([False, True, False, False, False, True])
    goal_mask = np.zeros((6, 3), dtype=np.float32)
    goal_mask[:, 0] = 1.0
    goal_target = np.ones((6, 3), dtype=np.float32)
    goal_target[2] = 0.0
    np.savez_compressed(
        path,
        obs=np.zeros((6, 8), dtype=np.float32),
        actions=actions,
        dones=dones,
        goal_mask=goal_mask,
        goal_target=goal_target,
        phase=np.asarray([phase]),
        episodes_collected=np.asarray([2]),
        episodes_attempted=np.asarray([3]),
        episodes_rejected_trivial=np.asarray([1]),
    )


def test_learning_series_uses_cumulative_transitions_and_aligned_rolling_window(tmp_path: Path) -> None:
    path = tmp_path / "llc_movement_fluency_episodes.csv"
    _write_csv(path, EPISODE_FIELDS, _episode_rows())

    series = load_learning_series(path, phase="movement_fluency", window=2)

    np.testing.assert_array_equal(series.transitions, [5, 9])
    np.testing.assert_allclose(series.success, [0.5, 1.0])
    np.testing.assert_allclose(series.goal_error, [0.3, 0.15])
    assert series.total_transitions == 9


def test_retention_loader_never_falls_back_to_superseded_3h_files(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "llc_weapon_acquisition_3h_retention_eval.csv",
        RETENTION_FIELDS,
        [{"phase": "weapon_acquisition", "episodes": 99, "skill_score": 0.99, "retention": 0.99}],
    )

    matrices = build_retention_matrices(tmp_path)

    weapon_row = PRIMARY_PHASES.index("weapon_acquisition")
    assert np.isnan(matrices.skill_score[weapon_row]).all()
    assert matrices.source_paths[weapon_row] is None


def test_retention_matrix_preserves_missing_cells_and_episode_counts(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "llc_spacing_neutral_retention_eval.csv",
        RETENTION_FIELDS,
        [
            {"phase": "movement_fluency", "episodes": 3, "skill_score": 0.71, "retention": 0.90},
            {"phase": "spacing_neutral", "episodes": 3, "skill_score": 0.74, "retention": 1.00},
        ],
    )

    matrices = build_retention_matrices(tmp_path)

    row = PRIMARY_PHASES.index("spacing_neutral")
    movement = PRIMARY_PHASES.index("movement_fluency")
    spacing = PRIMARY_PHASES.index("spacing_neutral")
    assert matrices.skill_score[row, movement] == pytest.approx(0.71)
    assert matrices.retention[row, movement] == pytest.approx(0.90)
    assert matrices.episodes[row, movement] == 3
    assert matrices.skill_score[row, spacing] == pytest.approx(0.74)
    assert np.isnan(matrices.skill_score[row, PRIMARY_PHASES.index("combat_execution")])


def test_demo_summary_recomputes_quality_metrics_from_archive(tmp_path: Path) -> None:
    path = tmp_path / "movement_fluency_demos.npz"
    _write_demo(path)

    summary = summarize_demo(path)

    assert summary.samples == 6
    assert summary.episodes == 2
    assert summary.attempts == 3
    assert summary.rejected_trivial == 1
    assert summary.distinct_actions == 5
    assert summary.idle_rate == pytest.approx(1 / 6)
    assert summary.dodge_rate == pytest.approx(2 / 6)
    assert summary.attack_rate == pytest.approx(1 / 6)
    assert summary.pickup_rate == pytest.approx(1 / 6)
    assert summary.single_step_rate == 0.0
    assert summary.median_episode_steps == pytest.approx(3.0)
    assert summary.invalid_goal_rows == 1


def test_build_evidence_emits_manifest_figures_tables_and_readiness_gaps(tmp_path: Path) -> None:
    models = tmp_path / "models"
    demos = tmp_path / "demos"
    output = tmp_path / "paper"
    models.mkdir()
    demos.mkdir()

    for phase in PRIMARY_PHASES:
        _write_csv(models / f"llc_{phase}_episodes.csv", EPISODE_FIELDS, _episode_rows())
        _write_demo(demos / f"{phase}_demos.npz", phase=phase)

    _write_csv(
        models / "llc_movement_fluency_retention_eval.csv",
        RETENTION_FIELDS,
        [{"phase": "movement_fluency", "episodes": 2, "skill_score": 0.6, "retention": 1.0}],
    )

    result = build_evidence(
        models_dir=models,
        demos_dir=demos,
        output_dir=output,
        window=2,
        min_eval_episodes=30,
    )

    expected = {
        "evidence_manifest.json",
        "evidence_readiness.md",
        "paper_metrics.csv",
        "paper_tables.tex",
        "demo_quality.png",
        "demo_quality.pdf",
        "corrected_learning_success.png",
        "corrected_learning_success.pdf",
        "corrected_goal_error.png",
        "corrected_goal_error.pdf",
        "corrected_skill_score_matrix.png",
        "corrected_skill_score_matrix.pdf",
        "corrected_retention_matrix.png",
        "corrected_retention_matrix.pdf",
    }
    assert expected.issubset({path.name for path in result.outputs})

    manifest = json.loads((output / "evidence_manifest.json").read_text(encoding="utf-8"))
    assert manifest["publication_ready"] is False
    assert manifest["source_policy"] == (
        "exact current canonical filenames only; "
        "no *_3h, pilot, failed-run, or alternate-version fallback"
    )
    assert any("weapon_acquisition retention evaluation is missing" in issue for issue in manifest["issues"])
    assert any("2 evaluation episodes" in issue for issue in manifest["issues"])
    assert manifest["sources"]
    assert all(len(source["sha256"]) == 64 for source in manifest["sources"])
