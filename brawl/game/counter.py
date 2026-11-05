"""The training toolbar's frame counter: ground truth for every frame step and rewind.

While training mode is paused, the toolbar shows a frame number between "F5 Prev" and "F6 Next".
It restarts at 0 on every pause, counts up with F6 and down with F5, and goes negative when F5
replays the history from before the pause.

The game drops some presses and lands others late (gate T3: a press that arrived after the
clock had given up on it and pressed again moved the game two frames). So the clock reads the
counter's *number*, not just whether it changed: it then always knows exactly which frame the
game is on, and can undo a press that landed twice.

The number is read by matching: the digits are a fixed font on a plain cell, so each character
is cut out by its columns of ink and compared with exemplars taken from labelled crops (``python scripts/gate.py counter``, stored in ``assets/counter/glyphs.npz``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from brawl.paths import ASSETS

COUNTER_BOX = (700, 40, 760, 62)  # x0, y0, x1, y1 at 1920 x 1080
CELL_COLUMNS = (6, 52)            # the number's cell inside COUNTER_BOX, without its border
INK = 100.0                       # grey level of digit pixels (text ~255, cell ~33)
CANVAS = 12                       # glyphs are compared on a canvas this wide
BACKGROUND = 33.0                 # grey level of the cell behind the digits
GLYPHS_PATH = ASSETS / "counter" / "glyphs.npz"


def crop_counter(image: Any, box: tuple[int, int, int, int] = COUNTER_BOX) -> np.ndarray:
    x0, y0, x1, y1 = box
    return np.asarray(image)[y0:y1, x0:x1].astype(np.float32).mean(axis=2)


def counter_changed(a: np.ndarray, b: np.ndarray) -> bool:
    """The counter is static UI text: a digit change moves dozens of pixels by about 50 grey
    levels; an unchanged counter moves none. Six pixels is a safe margin."""
    return int((np.abs(a - b) > 20).sum()) >= 6


# -- reading the number --------------------------------------------------------------------
def segment(crop: np.ndarray) -> list[np.ndarray]:
    """Cut the counter into characters: maximal runs of columns that contain ink."""
    cell = crop[:, CELL_COLUMNS[0]:CELL_COLUMNS[1]]
    ink = (cell > INK).any(axis=0)
    glyphs, start = [], None
    for x, on in enumerate([*ink, False]):
        if on and start is None:
            start = x
        elif not on and start is not None:
            glyphs.append(cell[:, start:x])
            start = None
    return glyphs


def _canvas(glyph: np.ndarray, shift: int = 0) -> np.ndarray:
    """A glyph centred on a fixed-width background canvas, moved ``shift`` columns."""
    out = np.full((glyph.shape[0], CANVAS), BACKGROUND, dtype=np.float32)
    x = min(max(0, (CANVAS - glyph.shape[1]) // 2 + shift), CANVAS - glyph.shape[1])
    out[:, x:x + glyph.shape[1]] = glyph[:, :CANVAS]
    return out


@dataclass
class Digits:
    """Nearest-neighbour reader for the counter's font.

    The digits are anti-aliased and drawn at sub-pixel positions (the number is centred), so one
    character has a few slightly different renderings. Every labelled rendering is kept as an
    exemplar and a glyph takes the character of its closest exemplar, allowing a one-column shift.
    A glyph whose closest exemplar is not clearly closer than every other character's is
    unreadable, so a doubtful crop reads as None rather than as a wrong number.
    """

    chars: np.ndarray        # (n,) the character of each exemplar
    exemplars: np.ndarray    # (n, height, CANVAS) uint8
    max_ratio: float = 0.5   # best distance / best other-character distance

    def char(self, glyph: np.ndarray) -> Optional[str]:
        if glyph.shape[0] != self.exemplars.shape[1] or glyph.shape[1] > CANVAS:
            return None
        ex = self.exemplars.astype(np.float32)
        d = np.min([np.abs(ex - _canvas(glyph, s)).mean(axis=(1, 2)) for s in (-1, 0, 1)], axis=0)
        best = int(np.argmin(d))
        others = d[self.chars != self.chars[best]]
        if others.size and d[best] > self.max_ratio * float(others.min()):
            return None
        return str(self.chars[best])

    def read(self, crop: np.ndarray) -> Optional[int]:
        """The number in a counter crop, or None if any character is unreadable."""
        text = ""
        for glyph in segment(crop):
            ch = self.char(glyph)
            if ch is None:
                return None
            text += ch
        try:
            return int(text)
        except ValueError:  # empty, or a sign in the wrong place
            return None

    @classmethod
    def fit(cls, samples: Iterable[tuple[np.ndarray, int]]) -> tuple["Digits", int]:
        """Exemplars from crops labelled with the number they show (duplicates dropped). Returns
        the reader and the number of crops skipped because their glyph count did not match."""
        seen: dict[bytes, str] = {}
        skipped = 0
        for crop, label in samples:
            text, glyphs = str(label), segment(crop)
            if len(glyphs) != len(text) or any(g.shape[1] > CANVAS for g in glyphs):
                skipped += 1
                continue
            for ch, glyph in zip(text, glyphs):
                seen.setdefault(_canvas(glyph).clip(0, 255).astype(np.uint8).tobytes(), ch)
        exemplars = np.stack([np.frombuffer(k, dtype=np.uint8).reshape(-1, CANVAS) for k in seen])
        return cls(np.array(list(seen.values())), exemplars), skipped

    def save(self, path: Path = GLYPHS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, chars=self.chars, exemplars=self.exemplars, max_ratio=np.array(self.max_ratio))

    @classmethod
    def load(cls, path: Path = GLYPHS_PATH) -> "Digits":
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: calibrate the counter with `python scripts/gate.py counter`")
        z = np.load(path)
        return cls(z["chars"], z["exemplars"], float(z["max_ratio"]))


def held_out_accuracy(samples: list[tuple[np.ndarray, int]], fold: int = 3) -> tuple[int, int]:
    """Fit on the numbers not divisible by ``fold``, read the others: a check that the reader
    generalises to numbers it has not seen, not just to repeated renderings of the same number."""
    train = [(c, v) for c, v in samples if v % fold]
    test = [(c, v) for c, v in samples if not v % fold]
    digits, _ = Digits.fit(train)
    return sum(digits.read(c) == v for c, v in test), len(test)


_digits: Optional[Digits] = None


def read_counter(image: Any) -> Optional[int]:
    """The frame number on a full screen image (exemplars loaded once)."""
    global _digits
    if _digits is None:
        _digits = Digits.load()
    return _digits.read(crop_counter(image))


# -- the counter as the clock sees it --------------------------------------------------------
class FrameCounter:
    """Reads the counter from the screen: its pixels (``read``) and its number (``value``)."""

    def __init__(self, screen: Any, *, extract: Callable[[Any], np.ndarray] = crop_counter,
                 changed: Callable[[np.ndarray, np.ndarray], bool] = counter_changed,
                 value_of: Callable[[Any], Optional[int]] = read_counter,
                 timeout_s: float = 0.15, now: Callable[[], float] = time.perf_counter) -> None:
        self.screen = screen
        self.extract = extract
        self.changed = changed
        self.value_of = value_of
        self.timeout_s = timeout_s
        self.now = now

    def read(self) -> np.ndarray:
        return self.extract(self.screen.grab(fresh=False).image)

    def value(self) -> Optional[int]:
        return self.value_of(self.screen.grab(fresh=False).image)

    def wait_change(self, before: np.ndarray, timeout_s: Optional[float] = None) -> bool:
        """Poll fresh frames until the counter's pixels differ from ``before``; False on timeout."""
        deadline = self.now() + (self.timeout_s if timeout_s is None else timeout_s)
        while True:
            frame = self.screen.grab(fresh=True)
            if self.changed(before, self.extract(frame.image)):
                return True
            if self.now() >= deadline:
                return False

    def wait_value(self, other: int, timeout_s: Optional[float] = None) -> Optional[int]:
        """Poll fresh frames until the counter shows a readable number other than ``other``;
        None on timeout."""
        deadline = self.now() + (self.timeout_s if timeout_s is None else timeout_s)
        while True:
            value = self.value_of(self.screen.grab(fresh=True).image)
            if value is not None and value != other:
                return value
            if self.now() >= deadline:
                return None
