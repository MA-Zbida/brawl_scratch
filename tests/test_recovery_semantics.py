from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pytest

from action_space import ACTION_DIM, Action
from feature_extractor.memory.state_spec import StateSpec
from feature_extractor.memory.structured_memory import FighterState, Memory
from train.curriculum_config import build_phase_spec
from train.llc_stage_common import StageGoalEnv


def _fighter(
    memory: Memory,
    *,
    x: float,
    y: float,
    grounded: bool = False,
    exists: bool = True,
    was_offstage: bool = False,
) -> FighterState:
    return FighterState(
        exists=exists,
        x=x,
        y=y,
        grounded=grounded,
        off_stage=was_offstage,
    )


@pytest.mark.parametrize("x", [0.20, 0.80])
def test_high_lateral_player_is_offstage(x: float) -> None:
    memory = Memory()
    player = _fighter(memory, x=x, y=memory.platform.y_min - 0.10)

    assert memory.update_off_stage(player) is True


def test_player_under_platform_is_offstage_even_between_ledges() -> None:
    memory = Memory()
    player = _fighter(
        memory,
        x=0.50,
        y=memory.platform.y_min + memory.physics.ground_y_tolerance + 0.02,
    )

    assert memory.update_off_stage(player) is True


def test_airborne_player_above_platform_is_not_offstage() -> None:
    memory = Memory()
    player = _fighter(memory, x=0.50, y=memory.platform.y_min - 0.10)

    assert memory.update_off_stage(player) is False


@pytest.mark.parametrize(
    ("grounded", "exists"),
    [(True, True), (False, False)],
)
def test_grounded_or_missing_player_is_never_offstage(
    grounded: bool,
    exists: bool,
) -> None:
    memory = Memory()
    player = _fighter(
        memory,
        x=0.20,
        y=memory.platform.y_min + 0.10,
        grounded=grounded,
        exists=exists,
    )

    assert memory.update_off_stage(player) is False


def test_offstage_hysteresis_ignores_small_inward_ledge_jitter() -> None:
    memory = Memory()
    player = _fighter(
        memory,
        x=memory.platform.x_min + 0.005,
        y=memory.platform.y_min - 0.10,
        was_offstage=True,
    )

    assert memory.update_off_stage(player) is True

    player.x = 0.50
    assert memory.update_off_stage(player) is False


def _recovery_obs(
    *,
    signed_dx_to_ledge: float,
    offstage: bool,
    grounded: bool,
) -> np.ndarray:
    obs = np.zeros((StateSpec.dim(),), dtype=np.float32)
    obs[StateSpec.index("player_x")] = 0.50
    obs[StateSpec.index("player_y")] = 0.50
    obs[StateSpec.index("signed_dx_to_ledge")] = signed_dx_to_ledge
    obs[StateSpec.index("dy_to_ledge")] = 0.0
    obs[StateSpec.index("player_is_offstage")] = float(offstage)
    obs[StateSpec.index("player_grounded")] = float(grounded)
    obs[StateSpec.index("player_jumps_norm")] = 1.0
    return obs


@dataclass
class _DummyPlayer:
    weapon_state: float = 0.0


@dataclass
class _DummyMemory:
    player: _DummyPlayer


class _ScriptedRecoveryEnv(gym.Env):
    metadata = {}

    def __init__(self, observations: list[np.ndarray]) -> None:
        super().__init__()
        self.observations = [np.asarray(obs, dtype=np.float32) for obs in observations]
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(StateSpec.dim(),),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(ACTION_DIM)
        self.memory = _DummyMemory(player=_DummyPlayer())
        self._index = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._index = 0
        return self.observations[0].copy(), self._info()

    def step(self, action):
        del action
        self._index = min(self._index + 1, len(self.observations) - 1)
        return self.observations[self._index].copy(), 0.0, False, False, self._info()

    @staticmethod
    def _info() -> dict[str, float]:
        return {
            "player_exists": 1.0,
            "player_respawn_timer": 0.0,
            "self_stock_lost_step": 0.0,
            "op_stock_lost_step": 0.0,
            "op_delta_damage": 0.0,
            "self_delta_damage": 0.0,
            "canon_mirrored": 0.0,
        }


def _scripted_recovery_env() -> StageGoalEnv:
    observations = [
        # Reset and another onstage frame are already inside the geometric radius.
        _recovery_obs(signed_dx_to_ledge=0.10, offstage=False, grounded=True),
        _recovery_obs(signed_dx_to_ledge=0.10, offstage=False, grounded=True),
        # Arming must depend on the offstage event, not proximity to the ledge.
        _recovery_obs(signed_dx_to_ledge=0.25, offstage=True, grounded=False),
        _recovery_obs(signed_dx_to_ledge=0.02, offstage=True, grounded=False),
        # Airborne over the stage is not yet a completed recovery.
        _recovery_obs(signed_dx_to_ledge=0.02, offstage=False, grounded=False),
        _recovery_obs(signed_dx_to_ledge=0.02, offstage=False, grounded=True),
    ]
    spec = build_phase_spec("recovery_mastery", terminate_on_death=False)
    return StageGoalEnv(_ScriptedRecoveryEnv(observations), spec)


def test_recovery_approach_from_onstage_cannot_succeed() -> None:
    env = _scripted_recovery_env()
    env.reset(seed=7)

    _, _, terminated, truncated, info = env.step(int(Action.MOVE_TOWARD))

    assert terminated is False
    assert truncated is False
    assert info["goal_error"] < env.stage_spec.success_threshold
    assert info["goal_success"] == 0.0
    assert info["sequential_phase"] == 1
    assert info["sequential_step1_completed"] == 0.0


def test_recovery_arms_offstage_then_requires_grounded_onstage_return() -> None:
    env = _scripted_recovery_env()
    env.reset(seed=7)
    env.step(int(Action.MOVE_TOWARD))

    _, _, _, truncated, armed_info = env.step(int(Action.MOVE_AWAY))
    assert truncated is False
    assert armed_info["goal_error"] > env.stage_spec.success_threshold
    assert armed_info["goal_success"] == 0.0
    assert armed_info["sequential_phase"] == 2
    assert armed_info["sequential_step1_transition"] == 1.0
    assert armed_info["sequential_step1_completed"] == 1.0

    _, _, _, truncated, offstage_info = env.step(int(Action.MOVE_TOWARD))
    assert truncated is False
    assert offstage_info["goal_success"] == 0.0

    _, _, _, truncated, airborne_info = env.step(int(Action.MOVE_TOWARD))
    assert truncated is False
    assert airborne_info["goal_success"] == 0.0

    _, _, _, truncated, landed_info = env.step(int(Action.MOVE_TOWARD))
    assert truncated is True
    assert landed_info["goal_success"] == 1.0
    assert landed_info["terminal_success"] == 1.0


def test_recovery_reset_clears_offstage_arming() -> None:
    env = _scripted_recovery_env()
    env.reset(seed=7)
    env.step(int(Action.MOVE_TOWARD))
    env.step(int(Action.MOVE_AWAY))

    env.reset(seed=8)
    _, _, _, truncated, info = env.step(int(Action.MOVE_TOWARD))

    assert truncated is False
    assert info["goal_success"] == 0.0
    assert info["sequential_phase"] == 1
    assert info["sequential_step1_completed"] == 0.0


def test_recovery_never_emits_noop_while_offstage() -> None:
    """A fighter falling past the ledge must keep steering, whatever is on cooldown.

    With jumps spent, dodge cooling and the recovery heavy cooling, the fallback was
    `_MOVE[hdir]` -- and inside the horizontal deadband that is NOOP. A sweep found
    4.8% of that state space emitting NOOP while offstage: the agent stops steering
    and falls straight past the stage. The deadband suppresses direction chatter on
    the ground; in the air it is actively harmful.
    """
    import numpy as np

    from action_space import components
    from feature_extractor.memory.state_spec import StateSpec
    from train.heuristic_teachers import HeuristicConfig, HeuristicState, _recovery_action

    cfg = HeuristicConfig()
    offenders = 0
    total = 0
    for dx in np.linspace(-0.25, 0.25, 21):
        for dy in np.linspace(-0.20, 0.20, 9):
            obs = np.zeros(StateSpec.dim(), dtype=np.float32)
            for key, value in (
                ("signed_dx_to_ledge", float(dx)),
                ("dy_to_ledge", float(dy)),
                ("player_is_offstage", 1.0),
                ("player_jumps_norm", 0.0),      # no jumps
                ("dodge_cooldown_norm", 1.0),    # dodge cooling
                ("player_vy", 0.3),
            ):
                obs[StateSpec.index(key)] = value

            state = HeuristicState()
            state.step = 100
            state.last_recovery_heavy_step = 100   # heavy just fired, still cooling

            comp = components(_recovery_action(obs, cfg, state))
            total += 1
            if not any((comp.hdir, comp.vdir, comp.jump, comp.dodge, comp.light, comp.heavy)):
                offenders += 1

    assert total > 0
    assert offenders == 0, f"{offenders}/{total} offstage states still emit a no-op"
