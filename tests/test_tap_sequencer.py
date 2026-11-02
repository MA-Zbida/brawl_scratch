"""Requests are not new key edges until both down and up windows have elapsed."""
from __future__ import annotations

import pytest


@pytest.mark.parametrize('key', ['space', 'e', 'num4', 'num5', 'num6'])
def test_repeated_requests_have_a_full_release_window(key):
    from brawl.control.taps import TapSequencer
    seq = TapSequencer(hold_steps=1)
    assert seq.advance({key}, now=0) == ({key}, set())
    assert seq.advance({key}, now=.005) == (set(), set())
    assert seq.advance({key}, now=.025) == (set(), {key})
    assert seq.advance({key}, now=.030) == (set(), set())
    assert seq.advance({key}, now=.050) == ({key}, set())


def test_latch_length_is_not_extended_by_repeated_requests():
    from brawl.control.taps import TapSequencer
    seq = TapSequencer(hold_steps=2)
    assert seq.advance({'space'}, now=0)[0] == {'space'}
    assert seq.advance({'space'}, now=.025) == (set(), set())
    assert seq.advance({'space'}, now=.050)[1] == {'space'}


def test_no_queued_tap_fires_after_request_changes():
    from brawl.control.taps import TapSequencer
    seq = TapSequencer(hold_steps=1)
    seq.advance({'space'}, now=0)
    assert seq.advance({'space'}, now=.025)[1] == {'space'}
    assert seq.advance(set(), now=.050) == (set(), set())


def test_reset_releases_holds_and_retains_release_time():
    from brawl.control.taps import TapSequencer
    seq = TapSequencer(hold_steps=8)
    seq.advance({'num6'}, now=0)
    assert seq.release(now=.025) == {'num6'}
    assert seq.release(now=.030) == set()
    assert seq.advance({'num6'}, now=.031) == (set(), set())
    assert seq.advance({'num6'}, now=.050)[0] == {'num6'}
