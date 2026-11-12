"""Brawlhaven (small): the stage's fixed geometry on the 1920 x 1080 screen (the camera does not move).

Measured on 2026-11-11 from the labelled frames: the grass edge in the median of 40 stills, and
571 fighters standing on the stage (feet at y 613-621, torso 575-590). The blast zones are beyond
the screen, so the screen's own edges are what can be measured against.
"""
from __future__ import annotations

from dataclasses import dataclass

WIDTH, HEIGHT = 1920, 1080
LEFT, RIGHT = 605.0, 1305.0   # the ledges
FLOOR = 617.0                 # y of a standing fighter's feet
TORSO = 35.0                  # a standing fighter's torso point is this far above its feet
CENTRE = ((LEFT + RIGHT) / 2, FLOOR - TORSO)
GROUND_SLACK = 12.0           # the standing torso varies by about 8 px with the pose


@dataclass(frozen=True)
class Place:
    grounded: bool      # standing on the stage
    offstage: bool      # no ground below: beyond a ledge
    below: bool         # lower than the stage's top: must recover upwards
    to_left_ledge: float    # pixels from the left ledge, positive on the stage side
    to_right_ledge: float   # pixels from the right ledge, positive on the stage side
    height: float       # pixels above the standing height (negative below it)


def place(x: float, y: float) -> Place:
    """Where a torso point is, relative to the stage."""
    height = (FLOOR - TORSO) - y
    over = LEFT <= x <= RIGHT
    return Place(grounded=over and abs(height) <= GROUND_SLACK, offstage=not over, below=height < -GROUND_SLACK,
                 to_left_ledge=x - LEFT, to_right_ledge=RIGHT - x, height=height)
