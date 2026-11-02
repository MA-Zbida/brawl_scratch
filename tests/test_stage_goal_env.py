from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
pytest.importorskip("ultralytics")

import gymnasium as gym

from action_space import ACTION_DIM, Action, components as action_components


def _is_attack(action) -> bool:
    comp = action_components(int(np.asarray(action).reshape(-1)[0]))
    return bool(comp.light or comp.heavy)

from feature_extractor.memory.state_spec import StateSpec
from feature_extractor.memory.canonicalize import mirror_goal_target
from train.curriculum_config import build_phase_spec
from train.curriculum_goals import GOAL_INDEX, default_goal_target
from train.llc_stage_common import StageGoalEnv


class _DummyPlayer:
    def __init__(self) -> None:
        self.weapon_state = 1.0


class _DummyMemory:
    def __init__(self) -> None:
        self.player = _DummyPlayer()


class DummyStageEnv(gym.Env):
    metadata = {}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(StateSpec.dim(),),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(ACTION_DIM)
        self.steps = 0
        self.actions: list[int] = []
        self.memory = _DummyMemory()

    def _obs(self) -> np.ndarray:
        obs = np.zeros((StateSpec.dim(),), dtype=np.float32)
        obs[StateSpec.index("player_x")] = 0.45
        obs[StateSpec.index("player_y")] = 0.58
        obs[StateSpec.index("signed_dx_to_ledge")] = 0.0
        obs[StateSpec.index("dy_to_ledge")] = 0.0
        obs[StateSpec.index("player_has_weapon")] = 0.0
        obs[StateSpec.index("weapon_dx")] = 0.1
        obs[StateSpec.index("weapon_dy")] = 0.0
        obs[StateSpec.index("rel_distance")] = 0.2
        obs[StateSpec.index("rel_dy")] = 0.0
        obs[StateSpec.index("in_strike_range")] = 1.0
        obs[StateSpec.index("opponent_damage_pct")] = 0.2
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self._obs(), {"player_exists": 1.0}

    def step(self, action):
        self.steps += 1
        self.actions.append(int(np.asarray(action).reshape(-1)[0]))
        info = {
            # An attack landed iff the executed action carries a light/heavy component.
            "op_delta_damage": 1.0 if _is_attack(action) else 0.0,
            "self_delta_damage": 0.25,
            "player_exists": 1.0,
            "player_respawn_timer": 0.0,
        }
        return self._obs(), 0.0, False, self.steps >= 3, info


class MirroredDummyStageEnv(DummyStageEnv):
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info["canon_mirrored"] = 1.0
        return obs, info


def test_stage_goal_env_emits_dynamic_goal_diagnostics() -> None:
    spec = build_phase_spec("all_skills_llc", terminate_on_death=False)
    env = StageGoalEnv(DummyStageEnv(), spec)

    obs, info = env.reset(seed=7)
    assert obs.shape[0] == StateSpec.dim() + (2 * len(spec.feature_names or []))
    assert "goal_mask" in info

    # Action 0 is NOOP; use a real attack so the attack diagnostic is exercised.
    _, _, _, _, info = env.step(int(Action.LIGHT_TOWARD))
    assert "active_goal_feature_errors" in info
    assert "raw_goal_feats" in info
    assert "stage_action" in info
    assert info["stage_action"] == int(Action.LIGHT_TOWARD)
    assert info["stage_action_attack"] == 1
    assert info["stage_action_hdir"] == 1
    assert np.asarray(info["goal_mask"]).shape == np.asarray(info["goal_target"]).shape


def test_stage_goal_env_exposes_world_goal_separately_from_policy_goal() -> None:
    spec = build_phase_spec("movement_fluency", terminate_on_death=False)
    world_target = np.full((len(spec.feature_names or []),), 0.5, dtype=np.float32)
    player_x_idx = list(spec.feature_names or []).index("player_x")
    player_y_idx = list(spec.feature_names or []).index("player_y")
    world_target[player_x_idx] = 0.37
    world_target[player_y_idx] = 0.61
    spec = replace(spec, target_sampler=lambda _obs: world_target.copy())
    env = StageGoalEnv(MirroredDummyStageEnv(), spec)

    _, info = env.reset(seed=7)

    assert info["stage_feature_names"] == list(spec.feature_names or [])
    np.testing.assert_allclose(info["goal_target_world"], world_target)
    np.testing.assert_allclose(
        info["goal_target"],
        mirror_goal_target(world_target, spec.feature_names or []),
    )


class _WeaponResetEnv(DummyStageEnv):
    def __init__(self, *, drop_succeeds: bool = True) -> None:
        super().__init__()
        self.armed = True
        self.drop_succeeds = bool(drop_succeeds)

    def _obs(self) -> np.ndarray:
        obs = super()._obs()
        obs[StateSpec.index("player_has_weapon")] = float(self.armed)
        return obs

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self.steps = 0
        return self._obs(), {"player_exists": 1.0, "player_respawn_timer": 0.0}

    def step(self, action):
        action_id = int(np.asarray(action).reshape(-1)[0])
        self.actions.append(action_id)
        self.steps += 1
        if action_id == int(Action.PICKUP) and self.drop_succeeds:
            self.armed = False
        return self._obs(), 0.0, False, False, {
            "player_exists": 1.0,
            "player_respawn_timer": 0.0,
            "op_delta_damage": 0.0,
            "self_delta_damage": 0.0,
        }


def test_weapon_reset_drops_held_weapon_before_activating_goal() -> None:
    spec = build_phase_spec("weapon_acquisition", terminate_on_death=False)
    base = _WeaponResetEnv()
    env = StageGoalEnv(base, spec)

    obs, info = env.reset(seed=7)

    assert base.actions == [int(Action.PICKUP)]
    assert StateSpec.get(obs, "player_has_weapon") == 0.0
    assert info["weapon_reset_steps"] == 1
    assert info["weapon_reset_drop_attempts"] == 1
    assert info["weapon_reset_succeeded"] == 1.0
    assert info["goal_active"] == 1.0


def test_weapon_reset_fails_closed_when_drop_cannot_be_confirmed() -> None:
    spec = replace(
        build_phase_spec("weapon_acquisition", terminate_on_death=False),
        weapon_reset_max_steps=3,
        weapon_reset_retry_interval=2,
    )
    base = _WeaponResetEnv(drop_succeeds=False)
    env = StageGoalEnv(base, spec)

    with pytest.raises(RuntimeError, match="could not confirm an unarmed player"):
        env.reset(seed=7)

    assert base.actions == [int(Action.PICKUP), int(Action.NOOP), int(Action.PICKUP)]


class _VisualWeaponEnv(DummyStageEnv):
    def __init__(self) -> None:
        super().__init__()
        self.armed = False

    def _obs(self) -> np.ndarray:
        obs = super()._obs()
        obs[StateSpec.index("player_has_weapon")] = float(self.armed)
        obs[StateSpec.index("weapon_on_ground")] = float(not self.armed)
        obs[StateSpec.index("weapon_dx")] = 0.01 if not self.armed else 0.0
        obs[StateSpec.index("weapon_dy")] = 0.0
        return obs

    def step(self, action):
        self.steps += 1
        action_id = int(np.asarray(action).reshape(-1)[0])
        self.actions.append(action_id)
        if action_id == int(Action.PICKUP):
            # Action inference changes the state immediately. The independent
            # visual source confirms it on the following classifier refresh.
            self.armed = True
        visually_confirmed = self.steps >= 2
        return self._obs(), 0.0, False, False, {
            "player_exists": 1.0,
            "player_respawn_timer": 0.0,
            "op_delta_damage": 0.0,
            "self_delta_damage": 0.0,
            "player_weapon_visual_valid": float(visually_confirmed),
            "player_weapon_visual_armed": float(visually_confirmed),
            "player_weapon_visual_confidence": 0.99 if visually_confirmed else 0.0,
            "player_weapon_visual_fresh": float(visually_confirmed),
            "weapon_pickup_action": float(action_id == int(Action.PICKUP)),
        }


def test_weapon_success_and_pickup_bonus_wait_for_visual_confirmation() -> None:
    spec = replace(
        build_phase_spec("weapon_acquisition", terminate_on_death=False),
        reset_drop_weapon_when_armed=False,
        step_penalty=0.0,
        progress_scale=0.0,
    )
    env = StageGoalEnv(_VisualWeaponEnv(), spec)
    env.reset(seed=7)

    _, first_reward, _, first_truncated, first = env.step(int(Action.PICKUP))
    _, confirmed_reward, _, confirmed_truncated, confirmed = env.step(int(Action.NOOP))

    assert first["goal_error"] < spec.success_threshold
    assert first["goal_success"] == 0.0
    assert first["weapon_success_blocked_visual"] == 1.0
    assert first["weapon_pickup_bonus_applied"] == 0.0
    assert not first_truncated
    assert confirmed["weapon_visual_confirmed"] == 1.0
    assert confirmed["goal_success"] == 1.0
    assert confirmed["weapon_pickup_bonus_applied"] == pytest.approx(spec.weapon_pickup_bonus)
    assert confirmed_reward >= spec.weapon_pickup_bonus + spec.success_bonus
    assert confirmed_truncated


def test_low_confidence_unarmed_does_not_clear_visual_confirmation(monkeypatch):
    base = _VisualWeaponEnv()
    spec = replace(build_phase_spec("weapon_acquisition"), reset_drop_weapon_when_armed=False)
    env = StageGoalEnv(base, spec)
    env.reset(seed=7)
    env.step(int(Action.PICKUP))
    env.step(int(Action.NOOP))
    original = base.step

    def uncertain(action):
        obs, reward, terminated, truncated, info = original(action)
        info.update(player_weapon_visual_armed=0., player_weapon_visual_confidence=.6)
        return obs, reward, terminated, truncated, info

    monkeypatch.setattr(base, "step", uncertain)
    assert env.step(int(Action.NOOP))[-1]["weapon_visual_confirmed"] == 1.


class _SpacingSafetyEnv(DummyStageEnv):
    def __init__(self, *, offstage: bool) -> None:
        super().__init__()
        self.offstage = bool(offstage)

    def _obs(self) -> np.ndarray:
        obs = super()._obs()
        obs[StateSpec.index("rel_distance")] = 1.0 if self.steps == 0 else 0.2
        obs[StateSpec.index("rel_dy")] = 0.0
        obs[StateSpec.index("player_is_offstage")] = float(self.offstage and self.steps > 0)
        return obs

    def step(self, action):
        self.steps += 1
        self.actions.append(int(np.asarray(action).reshape(-1)[0]))
        return self._obs(), 0.0, False, False, {
            "player_exists": 1.0,
            "player_respawn_timer": 0.0,
            "op_delta_damage": 0.0,
            "self_delta_damage": 0.0,
        }


def _spacing_target() -> np.ndarray:
    target = default_goal_target()
    target[GOAL_INDEX["rel_distance"]] = 0.1
    target[GOAL_INDEX["rel_dy"]] = 0.5
    return target


def test_spacing_goal_cannot_succeed_offstage() -> None:
    spec = replace(
        build_phase_spec("spacing_neutral", terminate_on_death=False),
        target_sampler=lambda _obs: _spacing_target(),
        progress_scale=0.0,
        success_bonus=0.0,
        step_penalty=0.0,
    )
    env = StageGoalEnv(_SpacingSafetyEnv(offstage=True), spec)
    env.reset(seed=7)

    _, reward, terminated, truncated, info = env.step(int(Action.NOOP))

    assert not terminated
    assert not truncated
    assert info["goal_error"] < spec.success_threshold
    assert info["goal_success"] == 0.0
    assert info["goal_success_blocked_offstage"] == 1.0
    assert info["combat_penalty_offstage"] > 0.0
    assert reward < 0.0


def test_spacing_goal_still_succeeds_onstage() -> None:
    spec = replace(
        build_phase_spec("spacing_neutral", terminate_on_death=False),
        target_sampler=lambda _obs: _spacing_target(),
    )
    env = StageGoalEnv(_SpacingSafetyEnv(offstage=False), spec)
    env.reset(seed=7)

    _, _, _, truncated, info = env.step(int(Action.NOOP))

    assert truncated
    assert info["goal_success"] == 1.0
    assert info["goal_success_blocked_offstage"] == 0.0


class _CombatTimingEnv(DummyStageEnv):
    def __init__(self, *, damage_steps: set[int]) -> None:
        super().__init__()
        self.damage_steps = set(damage_steps)

    def _obs(self) -> np.ndarray:
        obs = super()._obs()
        obs[StateSpec.index("opponent_damage_pct")] = 0.1
        return obs

    def step(self, action):
        self.steps += 1
        self.actions.append(int(np.asarray(action).reshape(-1)[0]))
        damage = 0.05 if self.steps in self.damage_steps else 0.0
        return self._obs(), 0.0, False, False, {
            "player_exists": 1.0,
            "player_respawn_timer": 0.0,
            "op_delta_damage": damage,
            "self_delta_damage": 0.0,
        }


def _combat_timing_spec():
    target = default_goal_target()
    target[GOAL_INDEX["opponent_damage_pct"]] = 0.9
    return replace(
        build_phase_spec("combat_execution", terminate_on_death=False),
        target_sampler=lambda _obs: target.copy(),
        attack_outcome_window_steps=3,
        attack_whiff_penalty_scale=0.2,
        progress_scale=0.0,
        hit_event_bonus=0.0,
        damage_dealt_scale=0.0,
        step_penalty=0.0,
    )


def test_delayed_hit_credits_pending_attack_without_whiff_penalty() -> None:
    env = StageGoalEnv(_CombatTimingEnv(damage_steps={3}), _combat_timing_spec())
    env.reset(seed=7)

    _, first_reward, _, _, first = env.step(int(Action.LIGHT_TOWARD))
    _, second_reward, _, _, second = env.step(int(Action.NOOP))
    _, hit_reward, _, _, hit = env.step(int(Action.NOOP))

    assert first["attack_whiff_events"] == 0
    assert second["attack_whiff_events"] == 0
    assert hit["attack_hit_credited"] == 1.0
    assert hit["attack_whiff_events"] == 0
    assert first_reward == second_reward == hit_reward == 0.0


def test_whiff_penalty_is_delayed_until_attack_window_expires() -> None:
    env = StageGoalEnv(_CombatTimingEnv(damage_steps=set()), _combat_timing_spec())
    env.reset(seed=7)

    _, attack_reward, _, _, attack = env.step(int(Action.LIGHT_TOWARD))
    _, _, _, _, waiting = env.step(int(Action.NOOP))
    _, _, _, _, still_waiting = env.step(int(Action.NOOP))
    _, expired_reward, _, _, expired = env.step(int(Action.NOOP))

    assert attack["attack_attempt_event"] == 1.0
    assert attack["attack_whiff_events"] == 0
    assert waiting["attack_whiff_events"] == 0
    assert still_waiting["attack_whiff_events"] == 0
    assert attack_reward == 0.0
    assert expired["attack_whiff_events"] == 1
    assert expired["combat_penalty_attack_whiff"] == pytest.approx(0.2)
    assert expired_reward == pytest.approx(-0.2)


def test_recovery_executes_allowed_heavy_and_rejects_other_attacks() -> None:
    spec = build_phase_spec("recovery_mastery", terminate_on_death=False)

    allowed_env = StageGoalEnv(DummyStageEnv(), spec)
    allowed_env.reset(seed=7)
    _, _, _, _, allowed_info = allowed_env.step(int(Action.HEAVY_TOWARD))

    blocked_env = StageGoalEnv(DummyStageEnv(), spec)
    blocked_env.reset(seed=7)
    _, _, _, _, blocked_info = blocked_env.step(int(Action.LIGHT_TOWARD))

    assert allowed_info["stage_action"] == int(Action.HEAVY_TOWARD)
    assert allowed_info["stage_action_requested"] == int(Action.HEAVY_TOWARD)
    assert blocked_info["stage_action"] == int(Action.NOOP)
    assert blocked_info["stage_action_requested"] == int(Action.LIGHT_TOWARD)


def test_reset_perturbation_emits_discrete_actions_and_preserves_jump(monkeypatch) -> None:
    spec = replace(
        build_phase_spec("movement_fluency", terminate_on_death=False),
        reset_perturb_steps=2,
    )
    base = DummyStageEnv()
    env = StageGoalEnv(base, spec)
    draws = iter((0.8, 0.1, 0.8, 0.8))  # toward, jump, no jump, goal family
    monkeypatch.setattr(np.random, "rand", lambda: next(draws))

    env.reset()

    assert len(base.actions) == 2
    assert all(0 <= action < ACTION_DIM for action in base.actions)
    assert any(action_components(action).jump for action in base.actions)


def test_missing_classifier_downgrades_the_requirement_instead_of_raising() -> None:
    """Without a classifier the requirement is unsatisfiable, so clear it and warn.

    The requirement lives on the phase spec; the classifier lives on the env. An
    exception here forced every entry point that builds a StageGoalEnv -- collect,
    BC, PPO, retention eval -- to opt out by hand, and each missing opt-out was
    found only by crashing mid-run, twice overnight. Downgrading makes possession
    fall back to action inference, which is what produced every working archive.
    """
    import gymnasium as gym

    from train.curriculum_config import build_phase_spec
    from train.llc_stage_common import StageGoalEnv

    class _NoClassifierEnv(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, (168,))
        action_space = gym.spaces.Discrete(27)
        weapon_classifier = None

        def reset(self, **_kwargs):
            return None, {}

        def step(self, _action):
            return None, 0.0, False, False, {}

    spec = build_phase_spec("weapon_acquisition", death_penalty=1.0, terminate_on_death=True)
    assert spec.require_visual_weapon_confirmation, "phase no longer sets the requirement"

    env = StageGoalEnv(_NoClassifierEnv(), spec)

    assert env.stage_spec.require_visual_weapon_confirmation is False
