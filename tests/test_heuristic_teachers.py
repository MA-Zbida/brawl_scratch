from __future__ import annotations

import numpy as np

from feature_extractor.memory.state_spec import StateSpec
from train.curriculum_goals import GOAL_DIM, GOAL_INDEX, default_goal_target
from action_space import Action, describe
from train.heuristic_teachers import heuristic_action


def _obs() -> np.ndarray:
    obs = np.zeros((StateSpec.dim(),), dtype=np.float32)
    obs[StateSpec.index("player_x")] = 0.5
    obs[StateSpec.index("player_y")] = 0.5
    obs[StateSpec.index("player_jumps_norm")] = 1.0
    obs[StateSpec.index("opponent_exists")] = 1.0
    return obs


def _aug(obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return np.concatenate([obs, goal, np.ones((GOAL_DIM,), dtype=np.float32)]).astype(np.float32)


def test_movement_moves_right_when_player_x_is_left_of_target() -> None:
    obs = _obs()
    goal = default_goal_target()
    goal[GOAL_INDEX["player_x"]] = 0.75
    goal[GOAL_INDEX["player_y"]] = 0.5

    action = heuristic_action("movement_fluency", _aug(obs, goal))

    assert action == int(Action.MOVE_TOWARD), describe(action)


def test_movement_jumps_when_target_y_is_above_player() -> None:
    obs = _obs()
    goal = default_goal_target()
    goal[GOAL_INDEX["player_x"]] = 0.5
    goal[GOAL_INDEX["player_y"]] = 0.25

    action = heuristic_action("movement_fluency", _aug(obs, goal))

    assert action in (int(Action.JUMP), int(Action.JUMP_TOWARD), int(Action.JUMP_AWAY)), describe(action)


def test_weapon_teacher_presses_num5_when_close_and_unarmed() -> None:
    obs = _obs()
    obs[StateSpec.index("player_has_weapon")] = 0.0
    obs[StateSpec.index("weapon_dx")] = 0.03
    obs[StateSpec.index("weapon_dy")] = 0.01

    action = heuristic_action("weapon_acquisition", obs)

    assert action == int(Action.PICKUP), describe(action)


def test_weapon_teacher_idles_when_already_armed() -> None:
    obs = _obs()
    obs[StateSpec.index("player_has_weapon")] = 1.0
    obs[StateSpec.index("weapon_dx")] = 0.0
    obs[StateSpec.index("weapon_dy")] = 0.0

    action = heuristic_action("weapon_acquisition", obs)

    assert action == int(Action.NOOP), describe(action)


def test_recovery_moves_toward_ledge_and_jumps_when_below() -> None:
    obs = _obs()
    obs[StateSpec.index("player_is_offstage")] = 1.0
    obs[StateSpec.index("signed_dx_to_ledge")] = 0.08
    obs[StateSpec.index("dy_to_ledge")] = 0.12

    action = heuristic_action("recovery_mastery", obs)

    assert action in (int(Action.JUMP), int(Action.JUMP_TOWARD), int(Action.JUMP_AWAY)), describe(action)


def test_spacing_approaches_when_too_far_from_target_distance() -> None:
    obs = _obs()
    obs[StateSpec.index("rel_dx")] = 0.2
    obs[StateSpec.index("rel_distance")] = 0.5
    goal = default_goal_target()
    goal[GOAL_INDEX["rel_distance"]] = 0.1

    action = heuristic_action("spacing_neutral", _aug(obs, goal))

    assert action == int(Action.MOVE_TOWARD), describe(action)


def test_combat_approaches_when_out_of_range() -> None:
    """Out of range, close the gap rather than swinging at air.

    No facing feature is consulted: the observation is canonicalised so the
    opponent is always at positive rel_dx, which makes "toward the opponent" a
    single fixed input.
    """
    obs = _obs()
    obs[StateSpec.index("rel_dx")] = 0.30
    obs[StateSpec.index("rel_dy")] = 0.0

    action = heuristic_action("combat_execution", obs)

    assert action == int(Action.MOVE_TOWARD), describe(action)


def test_combat_uses_a_directional_light_inside_light_range() -> None:
    """The direction is part of the move -- the old space could not express this."""
    obs = _obs()
    obs[StateSpec.index("rel_dx")] = 0.04
    obs[StateSpec.index("rel_dy")] = 0.0

    action = heuristic_action("combat_execution", obs)

    assert action == int(Action.LIGHT_TOWARD), describe(action)


def test_combat_uses_heavy_attack_inside_heavy_range() -> None:
    obs = _obs()
    obs[StateSpec.index("rel_dx")] = 0.065
    obs[StateSpec.index("rel_dy")] = 0.0

    action = heuristic_action("combat_execution", obs)

    assert action == int(Action.HEAVY_TOWARD), describe(action)


def test_combat_aims_downward_at_an_opponent_below() -> None:
    """Vertically misaligned no longer means "do nothing".

    With a down attack available, an opponent below is answered with LIGHT_DOWN
    rather than a neutral swing that would miss.
    """
    obs = _obs()
    obs[StateSpec.index("rel_dx")] = 0.04
    obs[StateSpec.index("rel_dy")] = 0.09     # opponent below (y grows downward)

    action = heuristic_action("combat_execution", obs)

    assert action == int(Action.LIGHT_DOWN), describe(action)


def test_all_skills_dispatches_from_goal_type_info() -> None:
    obs = _obs()
    goal = default_goal_target()
    goal[GOAL_INDEX["player_x"]] = 0.25
    goal[GOAL_INDEX["player_y"]] = 0.5

    action = heuristic_action("all_skills_llc", _aug(obs, goal), {"goal_type": "movement"})

    assert action == int(Action.MOVE_AWAY), describe(action)
