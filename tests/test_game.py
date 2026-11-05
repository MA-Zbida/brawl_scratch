"""brawl.game: keys, clock (both regimes), the difficulty macro and capture, against FakeGame."""
from __future__ import annotations

import pytest

from brawl.game.capture import Screen
from brawl.game.clock import RealtimeClock, SteppedClock, change_difficulty
from brawl.game.fake import LEVELS, FakeGame
from brawl.game.keys import JUMP, LEFT, LIGHT, RIGHT, Keys

NO_KEYS: frozenset[str] = frozenset()


def keys_for(game: FakeGame) -> Keys:
    return Keys(game, sleep=lambda s: None)


def stepped(game: FakeGame, **kw) -> SteppedClock:
    clock = SteppedClock(keys_for(game), **kw)
    clock.start()
    return clock


# -- keys -----------------------------------------------------------------------------------
def test_hold_sends_only_the_changes():
    game = FakeGame()
    keys = keys_for(game)
    keys.hold({RIGHT, JUMP})
    game.log.clear()
    keys.hold({RIGHT, LIGHT})
    assert game.log == [("up", JUMP), ("down", LIGHT)]
    keys.release_all()
    assert game.held == set() and keys.held == frozenset()


def test_tap_refuses_a_held_key():
    keys = keys_for(FakeGame())
    keys.hold({LIGHT})
    with pytest.raises(ValueError):
        keys.tap(LIGHT)


# -- stepped clock --------------------------------------------------------------------------
def test_start_pauses_and_stop_resumes():
    game = FakeGame()
    clock = stepped(game)
    assert game.paused
    clock.stop()
    assert not game.paused


def test_advance_moves_exactly_one_frame_per_plan_entry():
    game = FakeGame()
    clock = stepped(game)
    timing = clock.advance([frozenset({RIGHT})] * 3)
    assert (game.frame, game.x, clock.frame, timing.frames) == (3, 3, 3, 3)
    clock.advance([NO_KEYS, frozenset({LEFT})])
    assert (game.frame, game.x, clock.frame) == (5, 2, 5)


def test_advance_requires_a_paused_game():
    clock = SteppedClock(keys_for(FakeGame()))
    with pytest.raises(RuntimeError):
        clock.advance([NO_KEYS])


def test_rewind_restores_the_exact_state():
    game = FakeGame()
    clock = stepped(game)
    clock.advance([frozenset({RIGHT})] * 4)
    before = (game.x, game.bot)
    clock.advance([frozenset({RIGHT})] * 3)
    clock.rewind(3)
    assert (game.frame, clock.frame, (game.x, game.bot)) == (4, 4, before)
    with pytest.raises(ValueError):
        clock.rewind(5)


def test_replaying_after_a_rewind_is_deterministic_and_new_keys_change_the_past():
    game = FakeGame()
    screen = Screen(game, now=lambda: 0.0, sleep=lambda s: None)
    clock = stepped(game)
    plan = [frozenset({RIGHT}), NO_KEYS, frozenset({LEFT}), frozenset({RIGHT})]

    def play(p):
        frames = []
        for frame_keys in p:
            clock.advance([frame_keys])
            frames.append(screen.grab().image.tolist())
        return frames

    first = play(plan)
    clock.rewind(len(plan))
    assert play(plan) == first
    clock.rewind(len(plan))
    assert play([frozenset({LEFT})] * len(plan)) != first


@pytest.mark.parametrize("rearm", [False, True])
def test_rearm_mode_is_what_a_game_that_needs_fresh_presses_requires(rearm):
    game = FakeGame(needs_rearm=True)
    clock = stepped(game, rearm_each_frame=rearm)
    clock.advance([frozenset({RIGHT})] * 3)
    # Without re-arming, only the first frame sees the newly pressed key.
    assert game.x == (3 if rearm else 1)


# -- realtime clock -------------------------------------------------------------------------
class FakeTime:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += max(0.0, s)


def test_realtime_holds_each_entry_for_one_frame_period():
    clock_time = FakeTime()
    game = FakeGame()
    clock = RealtimeClock(keys_for(game), now=clock_time.now, sleep=clock_time.sleep)
    clock.start()
    timing = clock.advance([frozenset({RIGHT})] * 3)
    assert timing.frames == 3 and timing.late_frames == 0
    assert timing.wall_s == pytest.approx(3 / 60)
    with pytest.raises(RuntimeError):
        clock.rewind(1)


def test_realtime_counts_late_frames():
    clock_time = FakeTime()
    clock = RealtimeClock(keys_for(FakeGame()), now=clock_time.now, sleep=clock_time.sleep)
    clock.start()
    clock_time.t += 0.1  # a slow decision: 6 frame periods pass before the keys are set
    timing = clock.advance([NO_KEYS, NO_KEYS])
    assert timing.late_frames == 2


# -- operator action ------------------------------------------------------------------------
def test_change_difficulty_releases_fight_keys_and_walks_the_levels():
    game = FakeGame()
    clock = stepped(game)
    clock.keys.hold({RIGHT, LIGHT})
    change_difficulty(clock, 4, sleep=lambda s: None)
    assert LEVELS[game.difficulty] == "chosen"
    assert game.held == set()
    change_difficulty(clock, -2, sleep=lambda s: None)
    assert LEVELS[game.difficulty] == "hard"


# -- capture --------------------------------------------------------------------------------
def test_grab_waits_for_a_new_frame_and_reports_stale_ones():
    game = FakeGame()
    clock_time = FakeTime()
    screen = Screen(game, timeout_s=0.05, now=clock_time.now, sleep=clock_time.sleep)
    assert screen.grab().fresh
    stale = screen.grab()
    assert not stale.fresh and clock_time.t >= 0.05
    clock = stepped(game)
    clock.advance([frozenset({RIGHT})])
    frame = screen.grab()
    assert frame.fresh and frame.image[0, 0, 0] == 1
