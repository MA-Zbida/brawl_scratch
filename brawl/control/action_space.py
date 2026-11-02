"""Single categorical action space for the low-level controller.

Why one head instead of four
----------------------------
A factorised ``MultiDiscrete([movement, jump, dodge, attack])`` models

    pi(a|s) = pi(m|s) pi(j|s) pi(d|s) pi(k|s)

which asserts that direction and attack are chosen independently. In a platform
fighter they are not: a side-light and a down-light are different moves, and the
direction *is* part of the move. The factorised form also spends probability mass on
incoherent chords like jump+dodge+heavy+down. A single ``Discrete(27)`` puts one
logit on each executable control, and 27 logits cost nothing.

It also makes the moveset reachable at all. The old space had no direction modifier
on attacks, so side, down and aerial variants -- most of Brawlhalla's kit -- could
not be expressed.

Canonical directions
--------------------
Actions name **TOWARD** and **AWAY**, never left and right. The observation is
horizontally canonicalised so the opponent is always on the same side, and naming
actions the same way completes that: the policy never learns a left variant and a
right variant of the same skill, and the mirror becomes a single conversion in
``to_keys`` rather than policy-facing logic.

Grounded/aerial context is deliberately *not* encoded. The same input means
different things depending on state -- a directional dodge is a dash on the ground
and an air-dodge in the air; neutral heavy is a signature on the ground and a
recovery in the air. Duplicating actions per context would double the space to
express something the state already determines.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Iterable, NamedTuple, Optional, Sequence

import numpy as np


class Action(IntEnum):
    # ── locomotion (9) ──────────────────────────────────────────────────────
    NOOP = 0
    MOVE_TOWARD = 1
    MOVE_AWAY = 2
    FAST_FALL = 3
    FAST_FALL_TOWARD = 4
    FAST_FALL_AWAY = 5
    JUMP = 6
    JUMP_TOWARD = 7
    JUMP_AWAY = 8

    # ── dodge / dash (9) — ground gives dashes, air gives directional dodges ─
    DODGE_SPOT = 9
    DODGE_TOWARD = 10
    DODGE_AWAY = 11
    DODGE_UP = 12
    DODGE_DOWN = 13
    DODGE_UP_TOWARD = 14
    DODGE_UP_AWAY = 15
    DODGE_DOWN_TOWARD = 16
    DODGE_DOWN_AWAY = 17

    # ── light attacks (4) — ground: n/s/d light; air: n/s/d air ─────────────
    LIGHT_NEUTRAL = 18
    LIGHT_TOWARD = 19
    LIGHT_AWAY = 20
    LIGHT_DOWN = 21

    # ── heavy attacks (4) — ground: signatures; air: recovery / ground pound ─
    HEAVY_NEUTRAL = 22
    HEAVY_TOWARD = 23
    HEAVY_AWAY = 24
    HEAVY_DOWN = 25

    # ── interaction (1) ─────────────────────────────────────────────────────
    PICKUP = 26


ACTION_DIM: int = len(Action)


class Components(NamedTuple):
    """Decomposed action, in canonical (toward/away) space.

    ``hdir``: +1 toward the opponent, -1 away, 0 neutral.
    ``vdir``: +1 down, -1 up, 0 neutral (screen coordinates, y grows downward).
    """

    hdir: int
    vdir: int
    jump: int
    dodge: int
    light: int
    heavy: int
    interact: int


#: Number of scalars a decomposed action contributes to the observation.
COMPONENT_DIM: int = len(Components._fields)

COMPONENT_NAMES: tuple[str, ...] = (
    "prev_hdir",
    "prev_vdir",
    "prev_jump",
    "prev_dodge",
    "prev_light",
    "prev_heavy",
    "prev_interact",
)


def _c(hdir=0, vdir=0, jump=0, dodge=0, light=0, heavy=0, interact=0) -> Components:
    return Components(hdir, vdir, jump, dodge, light, heavy, interact)


COMPONENTS: dict[Action, Components] = {
    Action.NOOP: _c(),
    Action.MOVE_TOWARD: _c(hdir=+1),
    Action.MOVE_AWAY: _c(hdir=-1),
    Action.FAST_FALL: _c(vdir=+1),
    Action.FAST_FALL_TOWARD: _c(hdir=+1, vdir=+1),
    Action.FAST_FALL_AWAY: _c(hdir=-1, vdir=+1),
    Action.JUMP: _c(jump=1),
    Action.JUMP_TOWARD: _c(hdir=+1, jump=1),
    Action.JUMP_AWAY: _c(hdir=-1, jump=1),

    Action.DODGE_SPOT: _c(dodge=1),
    Action.DODGE_TOWARD: _c(hdir=+1, dodge=1),
    Action.DODGE_AWAY: _c(hdir=-1, dodge=1),
    Action.DODGE_UP: _c(vdir=-1, dodge=1),
    Action.DODGE_DOWN: _c(vdir=+1, dodge=1),
    Action.DODGE_UP_TOWARD: _c(hdir=+1, vdir=-1, dodge=1),
    Action.DODGE_UP_AWAY: _c(hdir=-1, vdir=-1, dodge=1),
    Action.DODGE_DOWN_TOWARD: _c(hdir=+1, vdir=+1, dodge=1),
    Action.DODGE_DOWN_AWAY: _c(hdir=-1, vdir=+1, dodge=1),

    Action.LIGHT_NEUTRAL: _c(light=1),
    Action.LIGHT_TOWARD: _c(hdir=+1, light=1),
    Action.LIGHT_AWAY: _c(hdir=-1, light=1),
    Action.LIGHT_DOWN: _c(vdir=+1, light=1),

    Action.HEAVY_NEUTRAL: _c(heavy=1),
    Action.HEAVY_TOWARD: _c(hdir=+1, heavy=1),
    Action.HEAVY_AWAY: _c(hdir=-1, heavy=1),
    Action.HEAVY_DOWN: _c(vdir=+1, heavy=1),

    Action.PICKUP: _c(interact=1),
}

#: Component table as an array, for cheap lookup into the observation.
COMPONENT_TABLE: np.ndarray = np.array(
    [tuple(COMPONENTS[Action(i)]) for i in range(ACTION_DIM)], dtype=np.float32
)

# ── physical keys ───────────────────────────────────────────────────────────
# Brawlhalla's defaults for this project's binding: WASD to move, Space to jump,
# E to dodge, numpad 4/6/5 for light / heavy / grab-throw.
KEY_LEFT = "a"
KEY_RIGHT = "d"
KEY_DOWN = "s"
KEY_UP = "w"
KEY_JUMP = "space"
KEY_DODGE = "e"
KEY_LIGHT = "num4"
KEY_HEAVY = "num6"
KEY_INTERACT = "num5"

#: Keys the controller holds for as long as the action selects them.
HELD_KEYS: frozenset[str] = frozenset({KEY_LEFT, KEY_RIGHT, KEY_DOWN, KEY_UP})
#: Keys emitted as a latched tap.
TAP_KEYS: frozenset[str] = frozenset({KEY_JUMP, KEY_DODGE, KEY_LIGHT, KEY_HEAVY, KEY_INTERACT})


def components(action: int) -> Components:
    return COMPONENTS[Action(int(action))]


def to_keys(action: int, mirrored: bool) -> tuple[set[str], set[str]]:
    """Translate a canonical action into physical (held, tapped) key sets.

    ``mirrored`` says the observation was horizontally flipped, so canonical
    "toward" corresponds to physical left. This is the only place the mirror
    touches the action path.
    """
    comp = components(action)

    held: set[str] = set()
    hdir = -comp.hdir if mirrored else comp.hdir
    if hdir > 0:
        held.add(KEY_RIGHT)
    elif hdir < 0:
        held.add(KEY_LEFT)
    if comp.vdir > 0:
        held.add(KEY_DOWN)
    elif comp.vdir < 0:
        held.add(KEY_UP)

    tapped: set[str] = set()
    if comp.jump:
        tapped.add(KEY_JUMP)
    if comp.dodge:
        tapped.add(KEY_DODGE)
    if comp.light:
        tapped.add(KEY_LIGHT)
    if comp.heavy:
        tapped.add(KEY_HEAVY)
    if comp.interact:
        tapped.add(KEY_INTERACT)

    return held, tapped


def component_vector(action: Optional[int]) -> np.ndarray:
    """Decomposed action for the observation.

    A raw action index would be worse than useless here: feeding ``a / 26`` invents
    an ordinal geometry in which LIGHT_TOWARD is "near" LIGHT_AWAY and "far" from
    NOOP, which is meaningless. The components carry the structure that actually
    predicts the next state.
    """
    if action is None:
        return np.zeros((COMPONENT_DIM,), dtype=np.float32)
    return COMPONENT_TABLE[int(action)].copy()


def mirror_component_vector(vec: np.ndarray) -> np.ndarray:
    """Flip the horizontal component of a decomposed action.

    Needed when history is replayed under a different mirror decision than the one
    in force when it was recorded.
    """
    out = np.asarray(vec, dtype=np.float32).copy()
    if out.shape[0] > 0:
        out[0] = -out[0]
    return out


# ── legality ────────────────────────────────────────────────────────────────

def legal_action_mask(
    *,
    dodge_available: bool = True,
    jumps_left: float = 3.0,
    weapon_in_range: bool = False,
    has_weapon: bool = False,
    grounded: bool = True,
) -> np.ndarray:
    """Boolean mask over the action set: True where the action can do something.

    The game already ignores impossible inputs, so this is not required for
    correctness -- pressing dodge on cooldown simply does nothing. Its value is
    sample efficiency: with a single categorical head, illegal logits can be zeroed
    before the softmax so exploration is never spent on inputs that cannot fire.
    (That requires a masking-aware algorithm; until then this is exposed through
    ``info`` and is informational.)
    """
    mask = np.ones((ACTION_DIM,), dtype=bool)

    if not dodge_available:
        for action, comp in COMPONENTS.items():
            if comp.dodge:
                mask[int(action)] = False

    if jumps_left <= 0.0:
        for action, comp in COMPONENTS.items():
            if comp.jump:
                mask[int(action)] = False

    # Grab/throw only means something next to a weapon, or while holding one.
    if not (weapon_in_range or has_weapon):
        mask[int(Action.PICKUP)] = False

    # Fast-fall is a no-op on the ground; down is still needed for down attacks,
    # so only the pure-movement variants are masked.
    if grounded:
        for action in (Action.FAST_FALL, Action.FAST_FALL_TOWARD, Action.FAST_FALL_AWAY):
            mask[int(action)] = False

    mask[int(Action.NOOP)] = True   # never mask every option
    return mask


def describe(action: int) -> str:
    return Action(int(action)).name


def sanitize(action) -> int:
    """Coerce whatever the policy emitted into a valid action index."""
    arr = np.asarray(action).reshape(-1)
    value = int(arr[0]) if arr.size else 0
    return int(np.clip(value, 0, ACTION_DIM - 1))


def actions_with(**predicate: int) -> tuple[Action, ...]:
    """All actions whose components match, e.g. ``actions_with(dodge=1)``."""
    out: list[Action] = []
    for action, comp in COMPONENTS.items():
        if all(getattr(comp, key) == value for key, value in predicate.items()):
            out.append(action)
    return tuple(sorted(out))


def action_names() -> list[str]:
    return [Action(i).name for i in range(ACTION_DIM)]
