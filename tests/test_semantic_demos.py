from __future__ import annotations

import numpy as np
import pytest

from brawl.control.semantic_actions import BY_NAME
from brawl.state.state_spec import StateSpec
from brawl.pipelines.semantic57.contract import ACTION_DIM, OBSERVATION_DIM
from brawl.pipelines.semantic57.demos import DemoBatch, load_demos, save_demos, split_episodes, validate_rows
from brawl.pipelines.semantic57.teacher import IntegratedTeacher


def observation(**features):
    obs = np.zeros(OBSERVATION_DIM, np.float32)
    for key, value in features.items():
        obs[StateSpec.index(key)] = value
    obs[-ACTION_DIM:] = 1
    return obs


def test_demo_contract_roundtrip_and_episode_disjoint_split(tmp_path):
    obs = np.stack([observation(player_x=.5)] * 12)
    data = DemoBatch(obs, np.ones(12, np.int64), np.repeat(np.arange(6), 2),
                     np.array(["weapon", "recovery", "combat"] * 4))
    path = tmp_path / "demos.npz"
    save_demos(path, data, {"accepted": 6})
    loaded = load_demos(path)
    train, validation = split_episodes(loaded, seed=42)
    assert len(train) and len(validation)
    assert not set(loaded.episodes[train]) & set(loaded.episodes[validation])
    np.testing.assert_array_equal(loaded.obs, obs)
    with pytest.raises(FileExistsError):
        save_demos(path, data, {})


def test_masked_labels_and_legacy_observations_are_rejected():
    obs = np.stack([observation()])
    obs[0, -ACTION_DIM + 1] = 0
    with pytest.raises(ValueError, match="masked"):
        validate_rows(obs, np.array([1]))
    with pytest.raises(ValueError, match="shape"):
        validate_rows(np.zeros((1, 168)), np.array([1]))


def test_previous_executor_contract_is_rejected():
    from brawl.pipelines.semantic57.contract import metadata, validate_metadata
    previous = metadata()
    previous['executor_version'] = 'semantic_sequence_candidate_v1'
    with pytest.raises(ValueError, match='executor_version'):
        validate_metadata(previous)


def test_integrated_teacher_recovers_toward_ledge_not_opponent():
    obs = observation(player_is_offstage=1, signed_dx_to_ledge=-.2,
                      player_jumps_norm=0, dodge_cooldown_norm=1)
    decision = IntegratedTeacher().action(obs)
    assert decision.family == "recovery"
    assert decision.action == BY_NAME["RECOVERY_AWAY"].id


def test_integrated_teacher_prioritises_visible_weapon():
    obs = observation(player_grounded=1, weapon_on_ground=1, weapon_dx=.01)
    decision = IntegratedTeacher().action(obs)
    assert decision.family == "weapon"
    assert decision.action == BY_NAME["PICKUP"].id


def test_teacher_respects_busy_mask():
    obs = observation(player_grounded=1, weapon_on_ground=1)
    obs[-ACTION_DIM:] = 0
    obs[-ACTION_DIM] = 1
    assert IntegratedTeacher().action(obs).action == 0
