"""The weapon phase must not pay for repeated pickup/drop.

`weapon_pickup_bonus` used to fire on every 0->1 transition while
`agent_weapon_drop_penalty` was smaller, so a pickup/drop cycle was worth a net
positive. Standing next to a weapon and alternating the grab input paid better per
second than learning to fetch one, and PPO finds that long before it finds walking.

The reward never lies about the game — the game is fine. It is the *shaping* that
was exploitable, which is harder to notice: training looks like it is working, the
return climbs, and the agent has learned to farm a bonus.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.curriculum_config import build_phase_spec


def _weapon_spec():
    return build_phase_spec("weapon_acquisition", terminate_on_death=False)


def test_dropping_costs_more_than_picking_up():
    """A full pickup->drop cycle must never be net-positive."""
    spec = _weapon_spec()

    cycle_value = spec.weapon_pickup_bonus - spec.agent_weapon_drop_penalty
    assert cycle_value < 0.0, (
        f"pickup {spec.weapon_pickup_bonus} minus drop {spec.agent_weapon_drop_penalty} "
        f"= {cycle_value:+.2f}; a non-negative cycle is farmable without moving"
    )


def test_pickup_bonus_is_capped_per_episode():
    spec = _weapon_spec()
    assert spec.weapon_pickup_bonus_once_per_episode, (
        "without this, repeated pickups pay repeatedly even when the drop penalty "
        "makes a full cycle unprofitable"
    )


def test_holding_a_weapon_cannot_outpay_the_goal():
    """The per-step hold bonus must stay small relative to completing the goal.

    A large per-step bonus for merely holding turns 'stand still while armed' into a
    competitive policy against actually satisfying the goal.
    """
    spec = _weapon_spec()
    assert spec.player_has_weapon_bonus < spec.success_bonus


def test_step_penalty_makes_standing_still_lose():
    """With no reward source available, idling must be strictly negative."""
    spec = _weapon_spec()
    assert spec.step_penalty > 0.0


@pytest.mark.parametrize("phase", ["weapon_acquisition"])
def test_pickup_is_reachable_within_the_action_space(phase):
    """The phase restricts actions; PICKUP must survive that restriction."""
    from action_space import Action

    spec = build_phase_spec(phase, terminate_on_death=False)
    assert spec.allowed_actions is not None
    assert int(Action.PICKUP) in spec.allowed_actions


def test_weapon_phase_cannot_attack():
    """Swinging is not part of acquiring a weapon; the phase should exclude it."""
    from action_space import Action, components

    spec = build_phase_spec("weapon_acquisition", terminate_on_death=False)
    for action in spec.allowed_actions:
        comp = components(action)
        assert not comp.light and not comp.heavy, (
            f"{Action(action).name} is an attack and should not be allowed in the weapon phase"
        )


def test_idling_while_armed_is_never_net_positive():
    """The second farm: hold bonus above the step penalty pays for standing still.

    At +0.1 hold against a -0.05 step penalty, an armed agent earned +0.05 every
    step forever. Over a 140-step episode that is +6.0 against a +1.0 success bonus,
    so "pick up one weapon and stop" strictly dominates completing the goal.
    """
    spec = _weapon_spec()

    net_per_step = spec.player_has_weapon_bonus - spec.step_penalty
    assert net_per_step < 0.0, (
        f"armed idling pays {net_per_step:+.3f}/step; any non-negative value makes "
        "standing still a viable policy"
    )


def test_completing_the_goal_beats_idling_for_a_full_episode():
    """Success must dominate the best available do-nothing return."""
    spec = _weapon_spec()

    max_episode_steps = 140          # train/collect_heuristic_curriculum_demos.py
    best_idle_return = max(0.0, spec.player_has_weapon_bonus - spec.step_penalty) * max_episode_steps

    assert spec.success_bonus > best_idle_return, (
        f"idling for a whole episode returns {best_idle_return:+.2f} against a "
        f"success bonus of {spec.success_bonus:+.2f}"
    )


def test_holding_still_beats_not_holding():
    """Lowering the bonus must not remove the incentive to keep the weapon."""
    spec = _weapon_spec()
    assert spec.player_has_weapon_bonus > 0.0
