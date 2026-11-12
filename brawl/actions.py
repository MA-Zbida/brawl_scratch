"""The action space: MultiDiscrete([9, 6]) <-> keys (research/DESIGN.md section 6).

An action is (direction, button), both held for the whole decision period. The direction moves,
aims attacks and steers recovery. Choosing the same button again at the next decision keeps it
held, which is how a signature is charged (29% of the human's heavy presses last 10 frames or
more); pressing it a second time needs one decision without it in between (50 ms at k = 3).

The same table reads a human's recorded keys back into actions (``from_demo``), so demonstrations
and the agent share one action space.
"""
from __future__ import annotations

from collections import Counter

from brawl.game.keys import DODGE, DOWN, HEAVY, JUMP, LEFT, LIGHT, RIGHT, THROW, UP, from_mask

DIRECTIONS: tuple[frozenset[str], ...] = tuple(frozenset(d) for d in (
    (), (LEFT,), (RIGHT,), (UP,), (DOWN,), (UP, LEFT), (UP, RIGHT), (DOWN, LEFT), (DOWN, RIGHT)))
BUTTONS: tuple[str | None, ...] = (None, JUMP, LIGHT, HEAVY, DODGE, THROW)
DIRECTION_NAMES = ("none", "left", "right", "up", "down", "up-left", "up-right", "down-left", "down-right")
BUTTON_NAMES = ("none", "jump", "light", "heavy", "dodge", "throw")
SHAPE = (len(DIRECTIONS), len(BUTTONS))
_MIRROR = {LEFT: RIGHT, RIGHT: LEFT}


def name(action: tuple[int, int]) -> str:
    return f"{DIRECTION_NAMES[action[0]]}+{BUTTON_NAMES[action[1]]}"


def to_plan(action: tuple[int, int], frames: int = 3) -> list[frozenset[str]]:
    """The keys to hold on each game frame of one decision (a plan for ``Clock.advance``)."""
    direction, button = DIRECTIONS[action[0]], BUTTONS[action[1]]
    return [direction | {button} if button else direction] * frames


def mirror(action: tuple[int, int]) -> tuple[int, int]:
    """The same action with left and right exchanged."""
    return DIRECTIONS.index(frozenset(_MIRROR.get(k, k) for k in DIRECTIONS[action[0]])), action[1]


def direction_of(held: frozenset[str]) -> int:
    """The direction a set of held keys means. Opposite keys held together cancel."""
    keys = set(held & {LEFT, RIGHT, UP, DOWN})
    for a, b in ((LEFT, RIGHT), (UP, DOWN)):
        if a in keys and b in keys:
            keys -= {a, b}
    return DIRECTIONS.index(frozenset(keys))


def from_demo(held: list[int], pressed: list[int], previous: int = 0) -> tuple[tuple[int, int], int]:
    """One decision's action from the recorded key masks of its frames (brawl.game.keys.to_mask):
    the direction held on most of the frames (the latest on a tie); the first button pressed, or the
    ``previous`` decision's button while it is still held on most of the frames. Also returns how
    many presses the action cannot express: a second button in the period, or the same button
    pressed again right after itself (the agent's repeat is one long hold)."""
    directions = [direction_of(from_mask(m)) for m in held]
    count = Counter(directions)
    direction = max(reversed(directions), key=lambda d: count[d])
    buttons = [BUTTONS.index(b) for m in pressed for b in BUTTONS[1:] if b in from_mask(m)]
    if buttons:
        return (direction, buttons[0]), len(buttons) - 1 + (buttons[0] == previous)
    still = previous and 2 * sum(BUTTONS[previous] in from_mask(m) for m in held) > len(held)
    return (direction, previous if still else 0), 0
