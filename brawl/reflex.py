"""The recovery reflex: a hand-written rule that brings Wu Shang back to the stage when he is falling off it.

Why (research/LAB_NOTEBOOK.md, 2026-11-19): on night 4, 35% of the stocks lost were falls nobody was
credited for, at low damage (median 30%). Before them the agent steered toward the stage but used up + heavy
(the recovery) on only 7.5% of its decisions off stage. It comes back from 97% of off-stage spells, so a
failure is rare and its cost arrives 80 decisions late: the value learning does not find it.

The rule reads only the observation vector (brawl/observation.py), so it can be replayed on a run's trace.
It takes over while the agent is beyond a ledge and falling, or anywhere below the stage's top (also under
the stage), and otherwise returns None: the agent decides. Beyond the screen's edge he is not detected, and
the observation keeps his last place with a speed of zero: the rule goes on from there (on night 4, 483 times
off the screen from off the stage, 1.8 s each, 27 of them ended in a lost stock).

Where to go: always toward the stage. Beside it below the top, that presses him against its side, where he
clings and gets a jump back. (Night 5's first version rose straight up beside the ledge and steered away
from it under the stage: it doubled the self-falls, 0.35 against 0.14 a minute, the falls ending straight
down beside the wall with nothing left to press.)

What to press, while falling (rising, a jump is still carrying him: hold the direction only):
1. Above the stage's top and near the ledge: nothing; drift and keep the jumps (he may land).
2. Jumps left (3 per airtime, the ground jump included): a new jump press.
3. Jumps spent, recovery not used since he left the ground or was last hit: up + heavy.
4. Recovery spent: a dodge, if not used since the ground or the last hit.
Never down (down + heavy is the ground pound, and down falls faster). A press needs one decision without
that button first, so a button the last action held is not chosen again.
"""
from __future__ import annotations

from brawl import actions, observation

F = {n: i for i, n in enumerate(observation.FEATURES)}
NONE, JUMP, LIGHT, HEAVY, DODGE, THROW = range(6)
DIR = {n: actions.DIRECTION_NAMES.index(n) for n in actions.DIRECTION_NAMES}
UP_OF = {DIR["none"]: DIR["up"], DIR["left"]: DIR["up-left"], DIR["right"]: DIR["up-right"]}
JUMPS = 3              # per airtime, the ground jump included
FAR = 250.0            # pixels beyond the ledge from which it jumps even above the stage's top
LAST = observation.HISTORY  # the history's last entry is the action just taken


def _last_button(obs) -> int:
    row = [obs[F[f"action_{LAST}_btn_{b}"]] for b in actions.BUTTON_NAMES]
    return max(range(len(row)), key=row.__getitem__)


def _used(obs, button: str) -> bool:
    """Pressed since he left the ground and since he was last hit (a hit gives the move back)."""
    pressed = obs[F[f"since_press_{button}"]]
    return pressed < obs[F["agent_since_ground"]] and pressed < obs[F["agent_since_hit"]]


def recover(obs) -> tuple[int, int] | None:
    """The action to take now, or None when the agent should decide."""
    offstage, below = obs[F["agent_offstage"]] > 0.5, obs[F["agent_below"]] > 0.5
    falling = obs[F["agent_vy"]] >= 0.0  # screen y grows downward; zero when not seen
    if not (below or (offstage and falling)):
        return None
    left = obs[F["agent_x"]] < 0  # the nearer ledge is the left one
    gap = -obs[F["agent_to_left_ledge" if left else "agent_to_right_ledge"]] * observation.LEDGE  # beyond it; < 0 under
    side = DIR["right"] if left else DIR["left"]
    if not falling or (not below and gap < FAR):
        return side, NONE
    up, last = UP_OF[side], _last_button(obs)
    for button, spare in ((JUMP, round(obs[F["agent_jumps_since_ground"]] * 4) < JUMPS),
                          (HEAVY, not _used(obs, "heavy")), (DODGE, not _used(obs, "dodge"))):
        if spare:
            return (up, button) if last != button else (side, NONE)
    return side, NONE
