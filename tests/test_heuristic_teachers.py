from __future__ import annotations

import numpy as np

from feature_extractor.memory.state_spec import StateSpec
from train.curriculum_goals import GOAL_DIM, GOAL_INDEX, default_goal_target
from action_space import Action, components, describe
from train import heuristic_teachers as teachers
from train.heuristic_teachers import HeuristicConfig, HeuristicTeacher, heuristic_action


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


def test_weapon_teacher_keeps_moving_when_already_armed() -> None:
    obs = _obs()
    obs[StateSpec.index("player_has_weapon")] = 1.0
    obs[StateSpec.index("weapon_dx")] = 0.0
    obs[StateSpec.index("weapon_dy")] = 0.0

    action = heuristic_action("weapon_acquisition", obs)

    assert action in (int(Action.MOVE_TOWARD), int(Action.MOVE_AWAY)), describe(action)


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


def test_recovery_dodges_toward_high_ledge_when_out_of_jumps() -> None:
    obs = _obs()
    obs[StateSpec.index("player_is_offstage")] = 1.0
    obs[StateSpec.index("player_grounded")] = 0.0
    obs[StateSpec.index("player_jumps_norm")] = 0.0
    obs[StateSpec.index("signed_dx_to_ledge")] = 0.12
    obs[StateSpec.index("dy_to_ledge")] = 0.10

    action = HeuristicTeacher().action("recovery_mastery", obs)

    assert action == int(Action.DODGE_UP_TOWARD), describe(action)


def test_recovery_uses_directional_heavy_when_air_dodge_is_unavailable() -> None:
    obs = _obs()
    obs[StateSpec.index("player_is_offstage")] = 1.0
    obs[StateSpec.index("player_grounded")] = 0.0
    obs[StateSpec.index("player_jumps_norm")] = 0.0
    obs[StateSpec.index("dodge_cooldown_norm")] = 1.0
    obs[StateSpec.index("signed_dx_to_ledge")] = 0.12
    obs[StateSpec.index("dy_to_ledge")] = 0.10

    action = HeuristicTeacher().action("recovery_mastery", obs)

    assert action == int(Action.HEAVY_TOWARD), describe(action)


def test_recovery_uses_heavy_for_shallow_edge_return_when_dodge_is_unavailable() -> None:
    """Horizontal offstage recoveries still need the dedicated recovery move.

    The live setup pilot stayed slightly above the ledge, but every episode spent
    many frames out of jumps with dodge unavailable. Requiring the ledge to be
    vertically above suppressed aerial heavy from the entire archive.
    """
    obs = _obs()
    obs[StateSpec.index("player_is_offstage")] = 1.0
    obs[StateSpec.index("player_grounded")] = 0.0
    obs[StateSpec.index("player_jumps_norm")] = 0.0
    obs[StateSpec.index("dodge_cooldown_norm")] = 1.0
    obs[StateSpec.index("signed_dx_to_ledge")] = 0.12
    obs[StateSpec.index("dy_to_ledge")] = -0.04

    action = HeuristicTeacher().action("recovery_mastery", obs)

    assert action == int(Action.HEAVY_TOWARD), describe(action)


def test_recovery_uses_neutral_heavy_without_a_toward_component() -> None:
    obs = _obs()
    obs[StateSpec.index("player_is_offstage")] = 1.0
    obs[StateSpec.index("player_grounded")] = 0.0
    obs[StateSpec.index("player_jumps_norm")] = 0.0
    obs[StateSpec.index("dodge_cooldown_norm")] = 1.0
    obs[StateSpec.index("signed_dx_to_ledge")] = 0.0
    obs[StateSpec.index("dy_to_ledge")] = 0.10

    action = HeuristicTeacher().action("recovery_mastery", obs)

    assert action == int(Action.HEAVY_NEUTRAL), describe(action)


def test_recovery_heavy_is_not_emitted_on_consecutive_steps() -> None:
    obs = _obs()
    obs[StateSpec.index("player_is_offstage")] = 1.0
    obs[StateSpec.index("player_grounded")] = 0.0
    obs[StateSpec.index("player_jumps_norm")] = 0.0
    obs[StateSpec.index("dodge_cooldown_norm")] = 1.0
    obs[StateSpec.index("signed_dx_to_ledge")] = 0.12
    obs[StateSpec.index("dy_to_ledge")] = 0.10
    teacher = HeuristicTeacher(HeuristicConfig(recovery_heavy_cooldown_steps=4))

    first = teacher.action("recovery_mastery", obs)
    second = teacher.action("recovery_mastery", obs)

    assert first == int(Action.HEAVY_TOWARD), describe(first)
    assert second == int(Action.MOVE_TOWARD), describe(second)


def test_recovery_setup_approaches_then_jumps_outward_through_ledge_deadband() -> None:
    setup = teachers.RecoverySetupController()
    obs = _obs()
    obs[StateSpec.index("player_grounded")] = 1.0
    obs[StateSpec.index("signed_dx_to_ledge")] = 0.12

    approach = setup.action(obs)
    obs[StateSpec.index("signed_dx_to_ledge")] = 0.0
    cross = setup.action(obs)

    assert approach == int(Action.MOVE_TOWARD), describe(approach)
    assert cross == int(Action.JUMP_TOWARD), describe(cross)


def test_recovery_setup_holds_outward_direction_after_crossing_the_ledge() -> None:
    setup = teachers.RecoverySetupController()
    obs = _obs()
    obs[StateSpec.index("player_grounded")] = 1.0
    obs[StateSpec.index("signed_dx_to_ledge")] = 0.03
    setup.action(obs)

    obs[StateSpec.index("player_grounded")] = 0.0
    obs[StateSpec.index("player_vy")] = -0.10
    # Once the character crosses the ledge, ledge_x - player_x changes sign.
    # Setup must not interpret that geometric sign flip as a command to turn back.
    obs[StateSpec.index("signed_dx_to_ledge")] = -0.03

    action = setup.action(obs)

    assert action == int(Action.MOVE_TOWARD), describe(action)


def test_recovery_setup_emits_only_one_jump_for_a_committed_crossing() -> None:
    setup = teachers.RecoverySetupController()
    obs = _obs()
    obs[StateSpec.index("player_grounded")] = 1.0
    obs[StateSpec.index("signed_dx_to_ledge")] = 0.02

    first = setup.action(obs)
    second = setup.action(obs)

    assert first == int(Action.JUMP_TOWARD), describe(first)
    assert second == int(Action.MOVE_TOWARD), describe(second)


def test_recovery_setup_tracks_canonical_stage_center_direction() -> None:
    setup = teachers.RecoverySetupController()
    obs = _obs()
    obs[StateSpec.index("player_grounded")] = 1.0
    obs[StateSpec.index("signed_dx_to_ledge")] = 0.02
    obs[StateSpec.index("signed_dx_to_stage_center")] = -0.20
    setup.action(obs)

    # A canonical mirror flips every signed horizontal feature. The canonical
    # action must flip too so the physical key continues pointing outward.
    obs[StateSpec.index("player_grounded")] = 0.0
    obs[StateSpec.index("signed_dx_to_ledge")] = -0.02
    obs[StateSpec.index("signed_dx_to_stage_center")] = 0.20

    action = setup.action(obs)

    assert action == int(Action.MOVE_AWAY), describe(action)


def test_movement_dashes_toward_a_far_ground_target() -> None:
    obs = _obs()
    obs[StateSpec.index("player_grounded")] = 1.0
    goal = default_goal_target()
    goal[GOAL_INDEX["player_x"]] = 0.90
    goal[GOAL_INDEX["player_y"]] = 0.50

    action = HeuristicTeacher().action("movement_fluency", _aug(obs, goal))

    assert action == int(Action.DODGE_TOWARD), describe(action)


def test_spacing_dodges_away_when_grounded_and_well_inside_target_distance() -> None:
    obs = _obs()
    obs[StateSpec.index("player_grounded")] = 1.0
    obs[StateSpec.index("rel_distance")] = 0.04
    goal = default_goal_target()
    goal[GOAL_INDEX["rel_distance"]] = 0.15

    action = HeuristicTeacher().action("spacing_neutral", _aug(obs, goal))

    assert action == int(Action.DODGE_AWAY), describe(action)


def test_combat_sweep_includes_defensive_dodge_and_non_attack_action() -> None:
    obs = _obs()
    obs[StateSpec.index("player_grounded")] = 1.0
    obs[StateSpec.index("in_strike_range")] = 1.0
    obs[StateSpec.index("rel_dx")] = 0.04
    obs[StateSpec.index("rel_dy")] = 0.0
    obs[StateSpec.index("opponent_dw")] = 0.10
    teacher = HeuristicTeacher(HeuristicConfig(combat_defense_interval=3))

    actions = [teacher.action("combat_execution", obs) for _ in range(6)]

    assert any(components(action).dodge for action in actions)
    assert any(action in (int(Action.MOVE_AWAY), int(Action.NOOP)) for action in actions)
    assert any(components(action).light or components(action).heavy for action in actions)


def test_weapon_pickup_is_not_repeated_on_consecutive_in_range_steps() -> None:
    obs = _obs()
    obs[StateSpec.index("player_has_weapon")] = 0.0
    obs[StateSpec.index("weapon_dx")] = 0.01
    obs[StateSpec.index("weapon_dy")] = 0.01
    teacher = HeuristicTeacher(HeuristicConfig(pickup_retry_cooldown_steps=4))

    first = teacher.action("weapon_acquisition", obs)
    second = teacher.action("weapon_acquisition", obs)

    assert first == int(Action.PICKUP), describe(first)
    assert second != int(Action.PICKUP), describe(second)
