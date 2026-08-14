"""Categorical policy support for phase-level action restrictions.

PPO must sample and score the same action that reaches the environment.  Replacing
an illegal sampled action with ``NOOP`` inside a wrapper breaks that contract: the
rollout stores the illegal action's log-probability beside the transition produced
by ``NOOP``.  This module removes illegal logits before the categorical sample, so
sampling, PPO updates, checkpoint inference, and execution all agree.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import CategoricalDistribution, Distribution
from stable_baselines3.common.policies import ActorCriticPolicy

from action_space import ACTION_DIM, Action, components


def phase_allowed_action_ids(stage_spec: Any) -> tuple[int, ...]:
    """Resolve one unambiguous action allowlist from a stage specification."""
    explicit = getattr(stage_spec, "allowed_actions", None)
    if explicit is not None:
        allowed = {int(action) for action in explicit}
        allowed = {action for action in allowed if 0 <= action < ACTION_DIM}
        allowed.add(int(Action.NOOP))
        return tuple(sorted(allowed))

    allowed = set(range(ACTION_DIM))

    for action in tuple(allowed):
        comp = components(action)
        if bool(getattr(stage_spec, "disable_attack", False)) and (comp.light or comp.heavy):
            allowed.discard(action)
        if bool(getattr(stage_spec, "disable_dodge", False)) and comp.dodge:
            allowed.discard(action)
        if bool(getattr(stage_spec, "disable_jump", False)) and comp.jump:
            allowed.discard(action)

    allowed = {action for action in allowed if 0 <= action < ACTION_DIM}
    allowed.add(int(Action.NOOP))
    return tuple(sorted(allowed))


class PhaseMaskedActorCriticPolicy(ActorCriticPolicy):
    """Actor-critic policy whose categorical distribution has a static phase mask.

    The mask is static for one skill-training run.  It remains a full 27-logit head,
    so weights transfer between phases; changing phase only changes which logits may
    participate in sampling and PPO log-probability calculations.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args: Any,
        allowed_action_ids: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> None:
        allowed = tuple(
            sorted(
                {
                    int(action)
                    for action in (
                        range(ACTION_DIM) if allowed_action_ids is None else allowed_action_ids
                    )
                }
            )
        )
        if not isinstance(action_space, spaces.Discrete) or int(action_space.n) != ACTION_DIM:
            raise TypeError(
                "PhaseMaskedActorCriticPolicy requires the canonical Discrete(27) action space"
            )
        if not allowed or any(action < 0 or action >= ACTION_DIM for action in allowed):
            raise ValueError(f"allowed_action_ids must be non-empty ids in [0, {ACTION_DIM - 1}]")
        if int(Action.NOOP) not in allowed:
            raise ValueError("allowed_action_ids must include NOOP")

        # ActorCriticPolicy builds the network during super().__init__.  The
        # distribution is not evaluated there, but setting these first keeps the
        # override safe if SB3 changes that construction detail later.
        object.__setattr__(self, "allowed_action_ids", allowed)
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        mask = th.zeros((ACTION_DIM,), dtype=th.bool)
        mask[list(allowed)] = True
        # The mask is reconstructible from policy_kwargs.  Keeping it out of the
        # state_dict lets legacy ActorCriticPolicy checkpoints load exactly while
        # preserving every learned parameter and optimizer tensor.
        self.register_buffer("_legal_action_mask", mask, persistent=False)

    def set_allowed_action_ids(self, allowed_action_ids: Sequence[int]) -> None:
        """Retarget the transient mask without changing any learned parameter."""
        allowed = tuple(sorted({int(action) for action in allowed_action_ids}))
        if not allowed or any(action < 0 or action >= ACTION_DIM for action in allowed):
            raise ValueError(f"allowed_action_ids must be non-empty ids in [0, {ACTION_DIM - 1}]")
        if int(Action.NOOP) not in allowed:
            raise ValueError("allowed_action_ids must include NOOP")

        mask = th.zeros_like(self._legal_action_mask)
        mask[list(allowed)] = True
        self._legal_action_mask.copy_(mask)
        self.allowed_action_ids = allowed

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        if not isinstance(self.action_dist, CategoricalDistribution):
            return super()._get_action_dist_from_latent(latent_pi)

        logits = self.action_net(latent_pi)
        mask = getattr(self, "_legal_action_mask", None)
        if mask is None:
            return self.action_dist.proba_distribution(action_logits=logits)
        masked_logits = logits.masked_fill(~mask.to(device=logits.device), float("-inf"))
        return self.action_dist.proba_distribution(action_logits=masked_logits)


def masked_policy_custom_objects(policy_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """SB3 load overrides needed to convert or retarget a skill checkpoint.

    ``policy_kwargs`` is replaced as one object because SB3 rejects a load when
    caller and checkpoint kwargs differ.  The network topology stays identical;
    only ``allowed_action_ids`` changes for the new phase.
    """
    return {
        "policy_class": PhaseMaskedActorCriticPolicy,
        "policy_kwargs": dict(policy_kwargs),
    }


def set_model_phase_action_mask(model: Any, phase: str) -> tuple[int, ...]:
    """Apply the evaluated phase's action contract to a loaded skill model."""
    policy = getattr(model, "policy", None)
    if not isinstance(policy, PhaseMaskedActorCriticPolicy):
        raise TypeError(
            "Cross-phase evaluation requires PhaseMaskedActorCriticPolicy; "
            "resume and save the legacy checkpoint through train_curriculum first"
        )

    # Local import avoids a module cycle: curriculum_config imports StageSpec,
    # whose training module imports this policy implementation.
    from train.curriculum_config import build_phase_spec

    allowed = phase_allowed_action_ids(build_phase_spec(str(phase)))
    policy.set_allowed_action_ids(allowed)
    return allowed
