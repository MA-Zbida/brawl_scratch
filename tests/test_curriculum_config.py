from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
pytest.importorskip("ultralytics")

from train.curriculum_config import PHASES, build_phase_spec
from train.curriculum_goals import GOAL_INDEX
from train.retention import PHASE_ORDER


def test_phase_registry_and_retention_use_the_same_order() -> None:
    assert PHASES == PHASE_ORDER


def test_all_skills_phase_is_available() -> None:
    assert "all_skills_llc" in PHASES
    spec = build_phase_spec("all_skills_llc")
    assert spec.goal_family_sampler is not None
    assert spec.name == "stage6_all_skills_llc"


def test_all_skills_sampler_returns_dynamic_family_masks() -> None:
    spec = build_phase_spec("all_skills_llc")
    assert spec.goal_family_sampler is not None

    seen: set[str] = set()
    masks: list[np.ndarray] = []
    obs = np.zeros((55,), dtype=np.float32)
    for _ in range(200):
        target, mask, goal_type = spec.goal_family_sampler(obs)
        seen.add(goal_type)
        masks.append(mask)
        assert target.shape == mask.shape == spec.mask.shape
        assert np.all(target >= 0.0) and np.all(target <= 1.0)
        assert np.any(mask > 0.0)

    assert seen == {"recovery", "movement", "weapon_acquisition", "spacing", "combat"}
    assert any(mask[GOAL_INDEX["player_x"]] > 0.0 for mask in masks)
    assert any(mask[GOAL_INDEX["player_has_weapon"]] > 0.0 for mask in masks)
    assert any(mask[GOAL_INDEX["in_strike_range"]] > 0.0 for mask in masks)
