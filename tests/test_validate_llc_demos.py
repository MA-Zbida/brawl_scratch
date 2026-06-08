from __future__ import annotations

import numpy as np

from tools.validate_llc_demos import split_paths, validate_demo_archive


def _write_demo(path, *, phase: str = "movement_fluency", samples: int = 12, idle: bool = False) -> None:
    obs = np.ones((samples, 77), dtype=np.float32)
    if idle:
        actions = np.tile(np.array([[3, 0, 0, 0]], dtype=np.int64), (samples, 1))
    else:
        pattern = np.array(
            [
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [2, 0, 1, 0],
                [3, 0, 0, 1],
            ],
            dtype=np.int64,
        )
        actions = np.vstack([pattern[i % len(pattern)] for i in range(samples)]).astype(np.int64)
    goal_mask = np.ones((samples, 11), dtype=np.float32)
    np.savez_compressed(
        path,
        obs=obs,
        actions=actions,
        actions_multidiscrete=actions,
        goal_mask=goal_mask,
        episodes_collected=np.asarray([3], dtype=np.int64),
        phase=np.asarray([phase]),
    )


def test_split_paths_accepts_semicolon_and_comma_lists() -> None:
    paths = split_paths(["a.npz;b.npz", "c.npz,d.npz"])
    assert [str(path) for path in paths] == ["a.npz", "b.npz", "c.npz", "d.npz"]


def test_validate_demo_archive_passes_well_formed_demo(tmp_path) -> None:
    path = tmp_path / "movement_fluency_demos.npz"
    _write_demo(path)

    result = validate_demo_archive(
        path,
        expected_phase="movement_fluency",
        min_samples=8,
        min_action_entropy=0.01,
        max_idle_rate=0.8,
    )

    assert result["status"] == "PASS"
    assert result["samples"] == 12
    assert result["obs_dim"] == 77
    assert result["action_dim"] == 4
    assert result["goal_active_ratio"] == 1.0


def test_validate_demo_archive_fails_phase_mismatch(tmp_path) -> None:
    path = tmp_path / "movement_fluency_demos.npz"
    _write_demo(path, phase="recovery_mastery")

    result = validate_demo_archive(path, expected_phase="movement_fluency", min_samples=8)

    assert result["status"] == "FAIL"
    assert any("phase metadata mismatch" in err for err in result["errors"])


def test_validate_demo_archive_warns_on_idle_action_collapse(tmp_path) -> None:
    path = tmp_path / "movement_fluency_demos.npz"
    _write_demo(path, idle=True)

    result = validate_demo_archive(
        path,
        expected_phase="movement_fluency",
        min_samples=8,
        min_action_entropy=0.01,
        max_idle_rate=0.5,
    )

    assert result["status"] == "WARN"
    assert any("high idle rate" in warning for warning in result["warnings"])
    assert any("low action entropy" in warning for warning in result["warnings"])

