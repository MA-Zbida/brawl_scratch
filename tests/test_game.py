"""brawl.game: keys, clock (both regimes), the difficulty macro and capture, against FakeGame."""
from __future__ import annotations

import numpy as np
import pytest

from brawl.game.capture import Screen
from brawl.game.clock import RealtimeClock, StepError, SteppedClock, change_difficulty
from brawl.game.counter import FrameCounter
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


# -- confirmed stepping (the frame counter) -------------------------------------------------
def counted(game: FakeGame, **kw) -> SteppedClock:
    """A stepped clock that confirms every press against the fake game's frame number."""
    clock_time = FakeTime()
    screen = Screen(game, timeout_s=0.01, now=clock_time.now, sleep=clock_time.sleep)
    counter = FrameCounter(screen, value_of=lambda img: int(np.asarray(img)[0, 0, 0]), now=clock_time.now)
    clock = SteppedClock(keys_for(game), counter=counter, sleep=clock_time.sleep, **kw)
    clock.start()
    return clock


def test_without_confirmation_dropped_presses_lose_frames():
    game = FakeGame(drop_presses={1, 4})
    clock = stepped(game)
    clock.advance([frozenset({RIGHT})] * 5)
    assert clock.frame == 5 and game.frame == 3  # the clock believes 5; the game moved 3


def test_confirmed_steps_retry_dropped_presses_and_stay_exact():
    game = FakeGame(drop_presses={1, 4, 5})
    clock = counted(game)
    timing = clock.advance([frozenset({RIGHT})] * 5)
    assert (clock.frame, game.frame, game.x, timing.retries) == (5, 5, 5, 3)
    timing = clock.rewind(5)
    assert (clock.frame, game.frame, game.x) == (0, 0, 0)


def test_a_dropped_key_release_that_blocks_rewind_is_recovered():
    # The game misses the release of RIGHT and, believing it still held, refuses to rewind.
    game = FakeGame(drop_release_of={RIGHT}, rewind_blocked_while_held=True)
    clock = counted(game)
    clock.advance([frozenset({RIGHT})] * 3)
    timing = clock.rewind(3)
    assert (clock.frame, game.frame, game.x) == (0, 0, 0)
    assert timing.retries == 2 and RIGHT not in game.held


def test_a_late_step_that_lands_with_its_retry_is_undone():
    game = FakeGame(late_presses={2})  # the third F6 lands only when the clock presses again
    clock = counted(game)
    timing = clock.advance([frozenset({RIGHT})] * 5)
    assert (clock.frame, game.frame, game.x) == (5, 5, 5)
    assert timing.retries == 1 and timing.repairs == 1


def test_a_late_rewind_is_undone_by_replaying_the_recorded_keys():
    plan = [frozenset({RIGHT}), frozenset({RIGHT}), frozenset({LEFT}), NO_KEYS, frozenset({RIGHT})]
    reference = FakeGame()
    stepped(reference).advance(plan[:2])
    game = FakeGame(late_presses={6})  # presses 0-4 step; the second rewind lands with its retry
    clock = counted(game)
    clock.advance(plan)
    timing = clock.rewind(3)
    assert (clock.frame, game.frame) == (2, 2)
    assert (game.x, game.bot) == (reference.x, reference.bot)
    assert timing.repairs == 1 and game.held == set()


def test_the_clock_adopts_the_games_frame_number():
    game = FakeGame()
    stepped(game).advance([NO_KEYS] * 4)  # the game is at frame 4 before the counted clock starts
    game.paused = False
    clock = counted(game)
    assert clock.frame == 4
    clock.advance([NO_KEYS])
    assert (clock.frame, game.frame) == (5, 5)


def test_a_game_that_never_responds_raises():
    game = FakeGame(drop_presses=set(range(100)))
    clock = counted(game, max_retries=2)
    with pytest.raises(StepError):
        clock.advance([NO_KEYS])


def test_key_gap_only_waits_when_keys_change():
    waits = []
    game = FakeGame()
    clock = SteppedClock(keys_for(game), key_gap_s=0.01, sleep=waits.append)
    clock.start()
    clock.advance([frozenset({RIGHT}), frozenset({RIGHT}), NO_KEYS])
    assert waits.count(0.01) == 2  # RIGHT pressed, then released; the repeat needs no gap


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


# -- reading the counter's number -----------------------------------------------------------
def synthetic_font():
    rng = np.random.default_rng(3)
    font = {}
    for ch in "-0123456789":
        w = 3 if ch == "1" else 5 if ch == "-" else 6
        glyph = (rng.random((10, w)) > 0.4) * 230.0 + 25.0
        glyph[:, 0] = glyph[:, -1] = 255.0  # ink in the edge columns, as the real font has
        if ch == "-":
            glyph[:] = 33.0
            glyph[5, :] = 255.0
        font[ch] = glyph
    return font


def render(text, font, shift=0):
    from brawl.game.counter import CELL_COLUMNS
    crop = np.full((22, 60), 33.0, dtype=np.float32)
    width = sum(font[c].shape[1] for c in text) + len(text) - 1
    x = CELL_COLUMNS[0] + (CELL_COLUMNS[1] - CELL_COLUMNS[0] - width) // 2 + shift
    for c in text:
        g = font[c]
        crop[8:18, x:x + g.shape[1]] = g
        x += g.shape[1] + 1
    return crop


def test_digits_fitted_from_labelled_crops_read_every_number():
    from brawl.game.counter import Digits, held_out_accuracy
    font = synthetic_font()
    labelled = [(render(str(v), font), v) for v in range(-15, 121)]
    digits, skipped = Digits.fit(labelled)
    assert skipped == 0
    assert all(digits.read(crop) == v for crop, v in labelled)
    assert held_out_accuracy(labelled) == (46, 46)
    assert digits.read(render("1544", font, shift=1)) == 1544
    assert digits.read(np.full((22, 60), 33.0, dtype=np.float32)) is None  # no number


def test_digits_round_trip_through_a_file(tmp_path):
    from brawl.game.counter import Digits
    font = synthetic_font()
    digits, _ = Digits.fit([(render(str(v), font), v) for v in range(-12, 30)])
    assert len(digits.exemplars) == 11  # one rendering per character: duplicates dropped
    digits.save(tmp_path / "glyphs.npz")
    loaded = Digits.load(tmp_path / "glyphs.npz")
    assert list(loaded.chars) == list(digits.chars) and loaded.read(render("-7", font)) == -7
