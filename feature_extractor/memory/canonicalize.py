"""Horizontal mirror canonicalisation.

Brawlhalla is left-right symmetric, but a policy trained on raw observations has
to learn "opponent on my right" and "opponent on my left" as two unrelated
problems from the same budget of real-time samples.

Canonicalising removes that duplication: the observation is flipped whenever
needed so the opponent is always on the same side, and the policy's action is
flipped back before it reaches the game. The policy only ever sees one
orientation, which roughly halves the state space it must cover -- the single
cheapest sample-efficiency win available on a wall-clock-bound setup.

Stability matters as much as correctness. A naive ``mirror when rel_dx < 0`` flips
every time the fighters cross, and near ``rel_dx == 0`` it would chatter between
frames, making the observation jump discontinuously. ``should_mirror`` therefore
applies a deadband and holds the previous decision inside it.

The mirror is an involution: applying it twice restores the original, so the same
routine both applies and undoes it.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from config import PLATFORM_BOUNDS
from feature_extractor.memory.state_spec import StateSpec

#: Reflection axis for absolute positions.
#:
#: This is the **stage** centre, not the screen centre. The calibrated platform
#: spans an interval that is not centred on 0.5, so reflecting about 0.5 would
#: map the stage slightly off itself and leave a residual asymmetry in every
#: ledge-relative feature -- small, but enough that mirrored situations would no
#: longer produce identical observations, which is the entire point.
STAGE_CENTER_X: float = (float(PLATFORM_BOUNDS["x_min"]) + float(PLATFORM_BOUNDS["x_max"])) / 2.0

#: Absolute positions in [0, 1]: reflected about the stage centre.
POSITION_X_FEATURES: tuple[str, ...] = (
    "player_x",
    "opponent_x",
)

#: Signed horizontal quantities: negated.
SIGNED_X_FEATURES: tuple[str, ...] = (
    "player_vx",
    "opponent_vx",
    "rel_dx",
    "rel_vx",
    "weapon_dx",
    "signed_dx_to_ledge",
    "signed_dx_to_stage_center",
    # The executed action's horizontal component lives in the dynamic block, so a
    # mirrored history slice must flip it too -- otherwise the recorded action
    # contradicts the motion it produced.
    "prev_hdir",
)

#: Goal-space dimensions holding an absolute x: reflected about the stage centre.
GOAL_POSITION_FEATURES: frozenset[str] = frozenset({"player_x"})

#: Goal-space dimensions holding a signed x offset. These live in [0, 1] with 0.5
#: as the zero point, so negating the underlying value is ``1 - v``.
GOAL_SIGNED_FEATURES: frozenset[str] = frozenset({
    "weapon_dx",
    "signed_dx_to_ledge",
    "rel_dx",
    "signed_dx_to_stage_center",
})

#: Every goal dimension affected by a mirror.
GOAL_MIRROR_FEATURES: frozenset[str] = GOAL_POSITION_FEATURES | GOAL_SIGNED_FEATURES

DEFAULT_DEADBAND: float = 0.03

_POSITION_IDX = tuple(StateSpec.index(n) for n in POSITION_X_FEATURES)
_SIGNED_IDX = tuple(StateSpec.index(n) for n in SIGNED_X_FEATURES)


def _reflect_x(value: float) -> float:
    """Reflect an absolute x about the stage centre, kept inside [0, 1]."""
    return float(np.clip((2.0 * STAGE_CENTER_X) - float(value), 0.0, 1.0))


def should_mirror(
    *,
    rel_dx: float,
    opponent_exists: bool,
    signed_dx_to_stage_center: float,
    previous: bool = False,
    deadband: float = DEFAULT_DEADBAND,
) -> bool:
    """Decide whether this frame should be flipped, with hysteresis.

    Canonical orientation puts the opponent on the **right**. When no opponent is
    visible the stage-centre offset is used instead, so recovery and movement
    still benefit from symmetry (both ledges become "the same ledge").

    Inside the deadband the previous decision is held, which prevents the
    observation from chattering when the key quantity hovers near zero.
    """
    deadband = abs(float(deadband))

    key = float(rel_dx) if opponent_exists else float(signed_dx_to_stage_center)
    if key < -deadband:
        return True
    if key > deadband:
        return False
    return bool(previous)


def mirror_state_vector(vec: np.ndarray, *, in_place: bool = False) -> np.ndarray:
    """Flip a single-frame state vector horizontally.

    Involutive: ``mirror_state_vector(mirror_state_vector(v)) == v``.
    """
    out = np.asarray(vec, dtype=np.float32)
    if not in_place:
        out = out.copy()

    for idx in _POSITION_IDX:
        out[idx] = _reflect_x(out[idx])
    for idx in _SIGNED_IDX:
        out[idx] = -out[idx]

    return out


def mirror_dynamic_block(block: np.ndarray, *, in_place: bool = False) -> np.ndarray:
    """Flip a stacked history slice (the leading dynamic features only)."""
    out = np.asarray(block, dtype=np.float32)
    if not in_place:
        out = out.copy()

    dynamic_dim = StateSpec.dynamic_dim()
    for idx in _POSITION_IDX:
        if idx < dynamic_dim:
            out[idx] = _reflect_x(out[idx])
    for idx in _SIGNED_IDX:
        if idx < dynamic_dim:
            out[idx] = -out[idx]
    return out


def mirror_goal_target(
    target: np.ndarray,
    feature_names: Sequence[str],
    *,
    in_place: bool = False,
) -> np.ndarray:
    """Flip a goal target expressed in normalised [0, 1] goal space."""
    out = np.asarray(target, dtype=np.float32)
    if not in_place:
        out = out.copy()

    for idx, name in enumerate(feature_names):
        if idx >= out.shape[0]:
            continue
        if name in GOAL_POSITION_FEATURES:
            out[idx] = _reflect_x(out[idx])
        elif name in GOAL_SIGNED_FEATURES:
            out[idx] = 1.0 - out[idx]
    return out
