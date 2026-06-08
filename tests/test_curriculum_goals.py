from __future__ import annotations

import numpy as np

from feature_extractor.memory.state_spec import StateSpec
from train.curriculum_goals import CURRICULUM_GOAL_FEATURES, GOAL_DIM, GOAL_INDEX, extract_curriculum_goal_features


def test_curriculum_feature_extraction_matches_goal_space() -> None:
    obs = np.zeros((StateSpec.dim(),), dtype=np.float32)
    obs[StateSpec.index("signed_dx_to_ledge")] = -1.0
    obs[StateSpec.index("dy_to_ledge")] = 1.0
    obs[StateSpec.index("player_x")] = 0.25
    obs[StateSpec.index("player_y")] = 0.75
    obs[StateSpec.index("player_has_weapon")] = 1.0
    obs[StateSpec.index("weapon_dx")] = 0.0
    obs[StateSpec.index("weapon_dy")] = -1.0
    obs[StateSpec.index("rel_distance")] = 0.5
    obs[StateSpec.index("rel_dy")] = 0.0
    obs[StateSpec.index("in_strike_range")] = 1.0
    obs[StateSpec.index("frame_advantage_estimate")] = -1.0

    feats = extract_curriculum_goal_features(obs)

    assert feats.shape == (GOAL_DIM,)
    assert len(CURRICULUM_GOAL_FEATURES) == GOAL_DIM
    assert np.all(feats >= 0.0) and np.all(feats <= 1.0)
    assert feats[GOAL_INDEX["signed_dx_to_ledge"]] == 0.0
    assert feats[GOAL_INDEX["dy_to_ledge"]] == 1.0
    assert feats[GOAL_INDEX["player_x"]] == 0.25
    assert feats[GOAL_INDEX["player_has_weapon"]] == 1.0
    assert feats[GOAL_INDEX["weapon_dy"]] == 0.0
    assert feats[GOAL_INDEX["frame_advantage_estimate"]] == 0.0

