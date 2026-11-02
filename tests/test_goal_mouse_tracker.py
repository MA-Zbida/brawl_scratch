from __future__ import annotations

import numpy as np

from brawl.state.canonicalize import mirror_goal_target
from brawl.pipelines.llc.train_curriculum import GoalMouseTrackerWrapper


def test_mouse_tracker_uses_world_coordinates_for_mirrored_goal() -> None:
    # Movement x/y are not the first two entries in the curriculum goal. The
    # leading recovery fields caught the original reset-time cursor bug.
    names = ["signed_dx_to_ledge", "dy_to_ledge", "player_x", "player_y"]
    world_target = np.asarray([0.5, 0.5, 0.37, 0.61], dtype=np.float32)
    policy_target = mirror_goal_target(world_target, names)
    wrapper = object.__new__(GoalMouseTrackerWrapper)

    goal_xy = wrapper._goal_xy_from_info(
        {
            "goal_target": policy_target,
            "goal_target_world": world_target,
            "goal_mask": np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
            "stage_feature_names": names,
            "canon_mirrored": 1.0,
        }
    )

    np.testing.assert_allclose(goal_xy, world_target[2:4])
