from __future__ import annotations

import numpy as np

from action_space import ACTION_DIM, Action
from feature_extractor.memory.state_spec import StateSpec

from tools.validate_llc_demos import split_paths, validate_demo_archive


#: StageGoalEnv observation = stacked env observation + goal target + goal mask.
_DEMO_OBS_DIM = StateSpec.observation_dim((2, 4, 8)) + (2 * 11)


def _write_demo(path, *, phase: str = "movement_fluency", samples: int = 12, idle: bool = False) -> None:
    obs = np.ones((samples, _DEMO_OBS_DIM), dtype=np.float32)
    if idle:
        actions = np.full((samples,), int(Action.NOOP), dtype=np.int64)
    else:
        actions = (np.arange(samples) % ACTION_DIM).astype(np.int64)
    goal_mask = np.ones((samples, 11), dtype=np.float32)
    goal_target = np.full((samples, 11), 0.5, dtype=np.float32)
    np.savez_compressed(
        path,
        obs=obs,
        actions=actions,
        actions_discrete=actions,
        goal_mask=goal_mask,
        goal_target=goal_target,
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
    assert result["obs_dim"] == _DEMO_OBS_DIM
    assert result["action_dim"] == ACTION_DIM
    assert result["goal_active_ratio"] == 1.0


def test_validate_demo_archive_fails_phase_mismatch(tmp_path) -> None:
    path = tmp_path / "movement_fluency_demos.npz"
    _write_demo(path, phase="recovery_mastery")

    result = validate_demo_archive(path, expected_phase="movement_fluency", min_samples=8)

    assert result["status"] == "FAIL"
    assert any("phase metadata mismatch" in err for err in result["errors"])


def test_validate_demo_archive_rejects_legacy_multidiscrete_actions(tmp_path) -> None:
    """An (N,4) archive is from MultiDiscrete([4,2,2,4]) and cannot be converted.

    That space had no way to express a direction-modified attack, so there is no
    id in the 27-action space to map its rows onto. Silently reinterpreting the
    columns would train BC on labels that never matched the demonstrator.
    """
    path = tmp_path / "movement_fluency_demos.npz"
    samples = 12
    np.savez_compressed(
        path,
        obs=np.ones((samples, _DEMO_OBS_DIM), dtype=np.float32),
        actions=np.zeros((samples, 4), dtype=np.int64),
        goal_mask=np.ones((samples, 11), dtype=np.float32),
        episodes_collected=np.asarray([3], dtype=np.int64),
        phase=np.asarray(["movement_fluency"]),
    )

    result = validate_demo_archive(path, expected_phase="movement_fluency", min_samples=8)

    assert result["status"] == "FAIL"
    assert any("Recollect" in err for err in result["errors"])


def test_validate_demo_archive_rejects_actions_outside_the_space(tmp_path) -> None:
    path = tmp_path / "movement_fluency_demos.npz"
    _write_demo(path)
    with np.load(path) as data:
        payload = {key: data[key] for key in data.files}
    payload["actions"] = np.full((12,), ACTION_DIM, dtype=np.int64)
    payload["actions_discrete"] = payload["actions"]
    np.savez_compressed(path, **payload)

    result = validate_demo_archive(path, expected_phase="movement_fluency", min_samples=8)

    assert result["status"] == "FAIL"
    assert any("outside the" in err for err in result["errors"])


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


def test_validate_demo_archive_rejects_active_mask_with_zero_goal_target(
    tmp_path,
) -> None:
    path = tmp_path / "recovery_mastery_demos.npz"
    _write_demo(path, phase="recovery_mastery")
    with np.load(path) as data:
        payload = {key: data[key] for key in data.files}
    payload["goal_target"] = payload["goal_target"].copy()
    payload["goal_target"][3] = 0.0
    np.savez_compressed(path, **payload)

    result = validate_demo_archive(
        path,
        expected_phase="recovery_mastery",
        min_samples=8,
    )

    assert result["status"] == "FAIL"
    assert any("active goal mask with an all-zero target" in err for err in result["errors"])

