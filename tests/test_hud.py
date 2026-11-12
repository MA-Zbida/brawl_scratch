"""brawl.sense.hud and brawl.ledger: readings from a drawn HUD, and the events between readings."""
from __future__ import annotations

import numpy as np
import pytest

from brawl.ledger import KO_FRAMES, Ledger, reward
from brawl.sense import hud

PORTRAIT = {"mordex": (110, 80, 60), "wu_shang": (55, 125, 150)}  # BGR under the face
BADGE = {"lost": (40, 40, 230), "scored": (240, 220, 60), None: None}


def draw(left, right, scale=1.0):
    """A frame with two slots, each (who, ring RGB, badge)."""
    frame = np.full((1080, 1920, 3), 150, np.uint8)
    for dx, (who, ring, badge) in ((0, left), (hud.SLOT_DX, right)):
        for x, y in hud.RING:
            frame[y - 4:y + 5, x + dx - 4:x + dx + 5] = ring[::-1]
        x0, y0, x1, y1 = hud.IDENTITY
        frame[y0:y1, x0 + dx:x1 + dx] = PORTRAIT[who]
        if badge:
            x0, y0, x1, y1 = hud.BADGE
            frame[y0:y1, x0 + dx:x0 + dx + 20] = BADGE[badge]
    if scale != 1.0:
        import cv2

        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return frame


def test_ring_colours_follow_the_damage_curve():
    assert hud.damage((255, 255, 255)) == 0
    assert hud.damage((255, 255, 190)) == pytest.approx(12.7, abs=0.1)   # overlay showed 13.00
    assert hud.damage((254, 188, 6)) == pytest.approx(82.8, abs=0.5)     # a JPEG's orange
    assert hud.damage((210, 2, 3)) == pytest.approx(185, abs=1)
    assert hud.damage((60, 120, 200)) is None       # not a ring
    assert hud.damage((251, 238, 232)) is None       # a white ring tinted by a badge dropping in


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_fighters_are_read_by_portrait_whichever_side_they_are_on(scale):
    h = hud.read(draw(("mordex", (255, 255, 0), None), ("wu_shang", (255, 255, 255), None), scale))
    assert h["mordex"].damage == pytest.approx(50) and h["wu_shang"].damage == 0
    h = hud.read(draw(("wu_shang", (255, 153, 0), "scored"), ("mordex", (255, 255, 255), "lost"), scale))
    assert h["wu_shang"].damage == pytest.approx(100) and h["wu_shang"].badge == "scored"
    assert h["mordex"].badge == "lost"


def test_a_frame_without_the_hud_is_not_read():
    assert hud.read(np.full((1080, 1920, 3), 30, np.uint8)) is None
    assert hud.read(draw(("mordex", (255, 255, 0), None), ("mordex", (255, 255, 255), None))) is None


def frames(*steps):
    """HUD readings for (mordex damage, wu damage, mordex badge) steps, via ring colours."""
    for m, w, badge in steps:
        ring = lambda d: (255, 255, round(255 - d * 255 / 50))  # noqa: E731  (0-50 damage)
        yield hud.read(draw(("mordex", ring(m), badge), ("wu_shang", ring(w), None)))


def test_damage_and_kos_become_events_once():
    ledger, total = Ledger(), []
    steps = [(0, 0, None), (10, 0, None), (10, 0, None), (10, 20, None), (30, 20, None),
             (0, 20, None), (0, 20, "lost"), (0, 20, "lost"), (5, 20, None)]
    for i, h in enumerate(frames(*steps)):
        total.append(ledger.update(h, i))
    assert sum(e.damage_dealt for e in total) == pytest.approx(35, abs=0.5)  # 10 + 20, then 5 after the KO
    assert sum(e.damage_taken for e in total) == pytest.approx(20, abs=0.5)
    assert [e.ko_scored for e in total] == [0, 0, 0, 0, 0, 1, 0, 0, 0]       # the reset, not again at its badge
    assert sum(e.misread for e in total) == 0
    assert reward(total[5]) == 1.0


def test_a_ko_at_zero_damage_is_seen_by_its_badge_and_unreadable_frames_change_nothing():
    ledger = Ledger()
    events = [ledger.update(h, KO_FRAMES + i) for i, h in enumerate(frames((0, 0, None), (0, 0, "lost")))]
    assert [e.ko_scored for e in events] == [0, 1]
    assert ledger.update(None, KO_FRAMES + 5).ko_scored == 0
    after = ledger.update(next(frames((8, 0, "lost"))), KO_FRAMES + 6)
    assert after.ko_scored == 0 and after.damage_dealt == pytest.approx(8, abs=0.5)
