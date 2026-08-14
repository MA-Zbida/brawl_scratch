"""Quality gates for heuristic behaviour-cloning collection."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np

from feature_extractor.memory.state_spec import StateSpec
from train import collect_bc_locomotion_demos as collector
from train.collect_bc_locomotion_demos import (
    CollectionRejections,
    RecoveryArming,
    prepare_nontrivial_goal,
)
from train.curriculum_config import build_phase_spec


def _obs(*, offstage: bool) -> np.ndarray:
    obs = np.zeros((StateSpec.dim(),), dtype=np.float32)
    obs[StateSpec.index("player_is_offstage")] = float(offstage)
    return obs


class _GoalEnv:
    def __init__(self, errors: list[float]) -> None:
        self.errors = iter(errors)
        self.current_error = next(self.errors)
        self.resamples = 0
        self.stage_spec = Namespace(success_threshold=0.08)

    def goal_error(self, obs: np.ndarray, info: dict) -> float:
        del obs, info
        return self.current_error

    def resample_goal(self, obs: np.ndarray, info: dict) -> tuple[np.ndarray, dict]:
        self.resamples += 1
        self.current_error = next(self.errors)
        return obs, info


def test_satisfied_reset_goal_is_rejected_and_counted_as_trivial() -> None:
    env = _GoalEnv([0.01] * 9)
    rejections = CollectionRejections()

    _obs_out, _info_out, ready, retries = prepare_nontrivial_goal(
        env,
        _obs(offstage=False),
        {},
        max_retries=8,
    )
    if not ready:
        rejections.reject_trivial()

    assert ready is False
    assert retries == 8
    assert rejections.episodes_rejected_trivial == 1
    assert rejections.as_metadata()["episodes_rejected_trivial"].tolist() == [1]


def test_satisfied_reset_goal_is_resampled_until_unsatisfied() -> None:
    env = _GoalEnv([0.01, 0.02, 0.20])

    _obs_out, _info_out, ready, retries = prepare_nontrivial_goal(
        env,
        _obs(offstage=False),
        {},
        max_retries=8,
    )

    assert ready is True
    assert retries == 2
    assert env.resamples == 2


def test_recovery_arming_buffers_only_offstage_and_later_frames() -> None:
    arming = RecoveryArming(required=True)
    scripted = [
        _obs(offstage=False),
        _obs(offstage=False),
        _obs(offstage=True),
        _obs(offstage=True),
    ]

    info = _goal_info()
    buffered = [obs for obs in scripted if arming.should_record(obs, info)]

    assert arming.armed is True
    assert len(buffered) == 2
    assert all(StateSpec.get(obs, "player_is_offstage") > 0.5 for obs in buffered)


def test_recovery_arming_rejects_stale_offstage_respawn_frame() -> None:
    arming = RecoveryArming(required=True)
    offstage = _obs(offstage=True)

    assert arming.should_record(
        offstage,
        _goal_info(goal_active=False, player_exists=False),
    ) is False
    assert arming.armed is False

    assert arming.should_record(offstage, _goal_info()) is True
    assert arming.armed is True


def _collector_args(tmp_path: Path, phase: str, **overrides) -> Namespace:
    values = {
        "phase": phase,
        "episodes": 1,
        "max_collection_attempts": 2,
        "max_episode_steps": 8,
        "min_episode_steps": 4,
        "delay": 0.0,
        "output": str(tmp_path / f"{phase}.npz"),
        "teacher": "heuristic",
        "move_mouse_to_goal": False,
        "death_penalty": 0.0,
        "end_episode_on_first_hit": None,
        "hit_damage_threshold": 1e-3,
        "enforce_recovery_sequence": phase == "recovery_mastery",
        "weapon_hold_steps": 2,
        "weapon_reset_max_steps": 0,
        "weapon_reset_retry_interval": 8,
        "weapon_drop_grace_steps": 1,
    }
    values.update(overrides)
    return Namespace(**values)


def _goal_info(
    *,
    success: bool = False,
    goal_active: bool = True,
    player_exists: bool = True,
) -> dict:
    spec = build_phase_spec("recovery_mastery")
    return {
        "goal_target": (
            spec.target_sampler(_obs(offstage=False))
            if goal_active
            else np.zeros_like(spec.mask)
        ),
        "goal_mask": spec.mask.copy(),
        "goal_type": "recovery",
        "goal_type_index": 0,
        "goal_success": float(success),
        "terminal_success": float(success),
        "goal_new_sampled": False,
        "goal_active": float(goal_active),
        "player_exists": float(player_exists),
        "player_respawn_timer": 0.0 if player_exists else 1.0,
    }


class _TrivialThenUsefulEnv:
    def __init__(self) -> None:
        self.stage_spec = build_phase_spec("spacing_neutral")
        self.attempt = 0
        self.step_count = 0
        self.current_error = 0.0

    def reset(self):
        self.attempt += 1
        self.step_count = 0
        self.current_error = 0.01 if self.attempt == 1 else 0.20
        info = _goal_info()
        info["goal_type"] = "spacing"
        info["goal_type_index"] = 3
        info["goal_mask"] = self.stage_spec.mask.copy()
        info["goal_target"] = self.stage_spec.target_sampler(_obs(offstage=False))
        return _obs(offstage=False), info

    def goal_error(self, obs, info):
        del obs, info
        return self.current_error

    def resample_goal(self, obs, info):
        return obs, info

    def step(self, action):
        del action
        self.step_count += 1
        done = self.step_count >= 4
        info = _goal_info(success=done)
        info["goal_type"] = "spacing"
        info["goal_type_index"] = 3
        info["goal_mask"] = self.stage_spec.mask.copy()
        info["goal_target"] = self.stage_spec.target_sampler(_obs(offstage=False))
        return _obs(offstage=False), 0.0, False, done, info

    def close(self):
        pass


class _InactiveThenSpacingEnv:
    """Reset starts during respawn, so no goal exists for two environment steps."""

    def __init__(self) -> None:
        self.stage_spec = build_phase_spec("spacing_neutral")
        self.step_count = 0
        self.actions: list[int] = []

    def _info(self, *, active: bool, success: bool = False) -> dict:
        target = (
            self.stage_spec.target_sampler(_obs(offstage=False))
            if active
            else np.zeros_like(self.stage_spec.mask)
        )
        return {
            "goal_target": target,
            "goal_mask": self.stage_spec.mask.copy(),
            "goal_type": "spacing",
            "goal_type_index": 3,
            "goal_success": float(success),
            "terminal_success": float(success),
            "goal_new_sampled": bool(active and self.step_count == 2),
            "goal_active": float(active),
            "player_exists": float(active),
            "player_respawn_timer": 0.0 if active else 1.0,
        }

    def reset(self):
        self.step_count = 0
        self.actions.clear()
        return _obs(offstage=False), self._info(active=False)

    def goal_error(self, obs, info):
        del obs, info
        # The inactive all-zero sentinel looks unsatisfied numerically. Recording
        # readiness must be checked before goal-error-based triviality checks.
        return 0.20

    def resample_goal(self, obs, info):
        return obs, info

    def step(self, action):
        self.actions.append(int(action))
        self.step_count += 1
        active = self.step_count >= 2
        success = self.step_count >= 6
        return (
            _obs(offstage=False),
            0.0,
            False,
            success,
            self._info(active=active, success=success),
        )

    def close(self):
        pass


class _ScriptedRecoveryEnv:
    def __init__(self) -> None:
        self.stage_spec = build_phase_spec("recovery_mastery")
        self.step_count = 0
        self.actions: list[int] = []

    @staticmethod
    def _state(*, offstage: bool, dx_to_ledge: float) -> np.ndarray:
        obs = _obs(offstage=offstage)
        obs[StateSpec.index("signed_dx_to_ledge")] = dx_to_ledge
        obs[StateSpec.index("dy_to_ledge")] = 0.08 if offstage else 0.0
        obs[StateSpec.index("player_grounded")] = float(not offstage)
        obs[StateSpec.index("player_jumps_norm")] = 1.0
        return obs

    def reset(self):
        self.step_count = 0
        self.actions.clear()
        return self._state(offstage=False, dx_to_ledge=0.10), _goal_info(success=False)

    def step(self, action):
        self.actions.append(int(action))
        self.step_count += 1
        offstage = self.step_count in (2, 3)
        success = self.step_count == 4
        dx_to_ledge = 0.0 if self.step_count == 1 else 0.08
        return (
            self._state(offstage=offstage, dx_to_ledge=dx_to_ledge),
            0.0,
            False,
            success,
            _goal_info(success=success),
        )

    def close(self):
        pass


class _RespawnThenRecoveryEnv:
    """A stale offstage position survives reset while the goal is inactive."""

    def __init__(self) -> None:
        self.stage_spec = build_phase_spec("recovery_mastery")
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return _obs(offstage=True), _goal_info(
            goal_active=False,
            player_exists=False,
        )

    def step(self, action):
        del action
        self.step_count += 1
        if self.step_count == 1:
            return (
                _obs(offstage=True),
                0.0,
                False,
                False,
                _goal_info(goal_active=False, player_exists=False),
            )
        if self.step_count in (2, 3):
            return _obs(offstage=True), 0.0, False, False, _goal_info()
        return _obs(offstage=False), 0.0, False, True, _goal_info(success=True)

    def close(self):
        pass


def _run_fake_collection(monkeypatch, args: Namespace, env) -> Path:
    monkeypatch.setattr(collector, "parse_args", lambda: args)
    monkeypatch.setattr(collector, "_build_env", lambda _args: env)
    collector.main()
    return Path(args.output)


def test_collector_persists_trivial_rejection_metadata(monkeypatch, tmp_path) -> None:
    args = _collector_args(tmp_path, "spacing_neutral")

    archive_path = _run_fake_collection(monkeypatch, args, _TrivialThenUsefulEnv())

    with np.load(archive_path, allow_pickle=False) as archive:
        assert archive["episodes_attempted"].tolist() == [2]
        assert archive["episodes_collected"].tolist() == [1]
        assert archive["episodes_rejected_trivial"].tolist() == [1]


def test_spacing_collector_discards_inactive_goal_respawn_frames(
    monkeypatch,
    tmp_path,
) -> None:
    args = _collector_args(
        tmp_path,
        "spacing_neutral",
        max_collection_attempts=1,
    )
    env = _InactiveThenSpacingEnv()

    archive_path = _run_fake_collection(monkeypatch, args, env)

    with np.load(archive_path, allow_pickle=False) as archive:
        assert archive["obs"].shape[0] == 4
        assert not np.any(np.isclose(archive["goal_target"], 0.0).all(axis=1))
        assert np.all(np.any(archive["goal_mask"] > 0.0, axis=1))
    assert env.actions[:2] == [int(collector.Action.NOOP)] * 2


def test_recovery_collector_archive_starts_at_first_offstage_frame(
    monkeypatch,
    tmp_path,
) -> None:
    args = _collector_args(
        tmp_path,
        "recovery_mastery",
        max_collection_attempts=1,
        min_episode_steps=2,
    )
    env = _ScriptedRecoveryEnv()

    archive_path = _run_fake_collection(monkeypatch, args, env)

    with np.load(archive_path, allow_pickle=False) as archive:
        recorded_obs = archive["obs"]
        assert recorded_obs.shape[0] == 2
        assert np.all(
            recorded_obs[:, StateSpec.index("player_is_offstage")] > 0.5
        )
    assert env.actions[:2] == [
        int(collector.Action.MOVE_TOWARD),
        int(collector.Action.JUMP_TOWARD),
    ]


def test_recovery_collector_discards_inactive_goal_respawn_frames(
    monkeypatch,
    tmp_path,
) -> None:
    args = _collector_args(
        tmp_path,
        "recovery_mastery",
        max_collection_attempts=1,
        min_episode_steps=2,
    )

    archive_path = _run_fake_collection(
        monkeypatch,
        args,
        _RespawnThenRecoveryEnv(),
    )

    with np.load(archive_path, allow_pickle=False) as archive:
        assert archive["obs"].shape[0] == 2
        assert not np.any(np.isclose(archive["goal_target"], 0.0).all(axis=1))
        assert np.all(np.any(archive["goal_mask"] > 0.0, axis=1))
