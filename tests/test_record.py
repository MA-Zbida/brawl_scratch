"""brawl.record (the recording format), the human key reader, and the demo and label scripts."""
from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from brawl.game.capture import Frame
from brawl.game.keys import HEAVY, JUMP, KEY_ORDER, LEFT, LIGHT, RIGHT, KeyReader, VIRTUAL_KEYS, from_mask, to_mask
from brawl.paths import REPO_ROOT
from brawl.record import Recording, Writer


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image(value: int) -> np.ndarray:
    return np.full((40, 64, 3), value, dtype=np.uint8)


# -- the recording format ---------------------------------------------------------------------
def test_rows_and_frames_survive_a_round_trip_across_shards(tmp_path):
    with Writer(tmp_path / "rec", {"source": "test"}, shard_rows=3, scale=1.0, quality=95) as w:
        for i in range(7):
            index = w.append(image(30 * i), t=i / 60, keys=i)
        w.still(index, image(200))
    rec = Recording(tmp_path / "rec")
    assert len(rec) == 7 and len(rec.shards) == 3 and rec.meta["source"] == "test"
    cols = rec.columns()
    assert cols["keys"].tolist() == list(range(7)) and cols["t"][-1] == pytest.approx(6 / 60)
    frames = list(rec.frames())
    assert [int(round(f.mean())) for f in frames] == [30 * i for i in range(7)]  # JPEG: near, not exact
    assert [p.name for p in rec.stills()] == ["0000006.png"]


def test_frames_are_downscaled_and_rows_must_share_columns(tmp_path):
    w = Writer(tmp_path / "rec", {}, scale=0.5)
    w.append(image(100), t=0.0)
    with pytest.raises(ValueError):
        w.append(image(100), t=0.1, extra=1)
    w.close()
    assert next(Recording(tmp_path / "rec").frames()).shape == (20, 32, 3)


def test_a_recording_is_never_overwritten(tmp_path):
    Writer(tmp_path / "rec", {}).close()
    with pytest.raises(FileExistsError):
        Writer(tmp_path / "rec", {})


# -- reading the human's keyboard ---------------------------------------------------------------
def test_key_masks_round_trip():
    assert from_mask(to_mask({LEFT, JUMP, HEAVY})) == {LEFT, JUMP, HEAVY}
    assert to_mask(KEY_ORDER) == 2 ** len(KEY_ORDER) - 1


def test_the_reader_keeps_taps_that_end_between_two_takes():
    down: set[int] = set()
    reader = KeyReader(is_down=lambda vk: vk in down)
    down.add(VIRTUAL_KEYS[JUMP][0])
    reader.poll()
    down.clear()               # the tap ends before the next recorded frame
    reader.poll()
    assert reader.take() == (frozenset(), frozenset({JUMP}))
    down.add(VIRTUAL_KEYS[RIGHT][0])
    reader.poll()
    reader.poll()
    assert reader.take() == (frozenset({RIGHT}), frozenset({RIGHT}))
    assert reader.take() == (frozenset({RIGHT}), frozenset())  # held, but not pressed again


def test_the_numpad_is_read_with_num_lock_on_or_off():
    for code in VIRTUAL_KEYS[LIGHT]:
        reader = KeyReader(is_down=lambda vk, code=code: vk == code)
        reader.poll()
        assert reader.take()[0] == {LIGHT}


# -- the demo recorder ----------------------------------------------------------------------------
class FakeTime:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += max(0.0, s)


class FakeScreen:
    def __init__(self, time_):
        self.time, self.grabs = time_, 0

    def grab(self, fresh=True):
        self.grabs += 1
        return Frame(image(self.grabs % 250), True, self.time.now())


def test_the_recorder_keeps_its_schedule_and_records_keys(tmp_path):
    record_demos = script("record_demos")
    clock = FakeTime()
    down: set[int] = set()
    reader = KeyReader(is_down=lambda vk: vk in down)

    def sleep(s):  # the human presses and releases jump while the recorder waits
        clock.sleep(s)
        down.symmetric_difference_update({VIRTUAL_KEYS[JUMP][0]})
        reader.poll()

    with Writer(tmp_path / "demo", {}, scale=1.0) as w:
        stats = record_demos.record(FakeScreen(clock), reader, w, rate_hz=60, seconds=1.0,
                                    still_every_s=0.5, now=clock.now, sleep=sleep)
    rec = Recording(tmp_path / "demo")
    cols = rec.columns()
    assert stats["rows"] == len(rec) == 61 and stats["late"] == stats["dropped"] == stats["skipped"] == 0
    assert np.allclose(np.diff(cols["t"]), 1 / 60)
    assert (cols["pressed"] == to_mask({JUMP})).sum() >= 29  # every other wait starts a press
    assert len(rec.stills()) == stats["stills"] == 3          # at 0, 0.5 and 1.0 s


# -- the label store ------------------------------------------------------------------------------
def test_labels_are_complete_only_when_both_fighters_are_resolved(tmp_path):
    label = script("label")
    labels = label.Labels(tmp_path)
    labels.set_box("a.png", "wu_shang", (50, 80, 10, 20))
    assert labels.data["a.png"]["wu_shang"]["box"] == [10, 20, 50, 80]
    assert not labels.complete("a.png")
    labels.cycle("a.png", "wu_shang", "facing")
    labels.cycle("a.png", "wu_shang", "weapon")
    labels.toggle_absent("a.png", "mordex")
    assert labels.complete("a.png")
    assert labels.data["a.png"]["wu_shang"] == {"box": [10, 20, 50, 80], "facing": "right", "weapon": "spear"}
    labels.save()
    assert json.loads((tmp_path / "labels.json").read_text())["a.png"]["mordex"] == {"absent": True}
    assert label.Labels(tmp_path).complete("a.png")
