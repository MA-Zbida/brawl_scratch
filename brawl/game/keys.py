"""The only module that touches the keyboard: sending keys to the game, and reading the human's.

Bindings are Brawlhalla's in-game controls as this project configures them. Training-mode
controls (pause, frame step) are the F-keys; on a laptop they are FN+F5/F6/F7, but software
sends the F-key scan codes directly, so FN is not needed (Gate T1 checks this).
"""
from __future__ import annotations

import time
from typing import Callable, Iterable, Protocol

# Fighting controls.
LEFT, RIGHT, UP, DOWN = "a", "d", "w", "s"
JUMP, DODGE = "space", "e"
LIGHT, HEAVY, THROW = "num4", "num6", "num5"

# Training-mode controls.
REWIND, STEP, PAUSE = "f5", "f6", "f7"
MENU = "tab"
DIFFICULTY_UP, DIFFICULTY_DOWN = "d", "a"  # pressed together with MENU

FIGHT_KEYS = frozenset({LEFT, RIGHT, UP, DOWN, JUMP, DODGE, LIGHT, HEAVY, THROW})


class Backend(Protocol):
    def key_down(self, key: str) -> None: ...

    def key_up(self, key: str) -> None: ...


class DirectInput:
    """Scan-code key events through pydirectinput (what DirectX games read)."""

    _NUMPAD = {"num4": 0x4B, "num5": 0x4C, "num6": 0x4D}

    def __init__(self) -> None:
        import pydirectinput

        for name, code in self._NUMPAD.items():
            pydirectinput.KEYBOARD_MAPPING.setdefault(name, code)
        pydirectinput.PAUSE = 0
        pydirectinput.FAILSAFE = False
        self._pdi = pydirectinput

    def key_down(self, key: str) -> None:
        self._pdi.keyDown(key)

    def key_up(self, key: str) -> None:
        self._pdi.keyUp(key)


class Keys:
    """Keeps track of which keys are held, so every change is the minimal set of key events."""

    def __init__(self, backend: Backend, *, tap_s: float = 0.02,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.backend = backend
        self.tap_s = tap_s
        self.sleep = sleep
        self.held: frozenset[str] = frozenset()

    def hold(self, keys: Iterable[str]) -> None:
        """Hold exactly ``keys``: release what is no longer wanted, press what is new."""
        target = frozenset(keys)
        for key in sorted(self.held - target):
            self.backend.key_up(key)
        for key in sorted(target - self.held):
            self.backend.key_down(key)
        self.held = target

    def tap(self, key: str, hold_s: float | None = None) -> None:
        """Press and release one key that is not held, keeping it down for ``hold_s`` (default
        ``tap_s``)."""
        if key in self.held:
            raise ValueError(f"{key!r} is held; release it before tapping")
        self.backend.key_down(key)
        self.sleep(self.tap_s if hold_s is None else hold_s)
        self.backend.key_up(key)

    def chord(self, keys: Iterable[str]) -> None:
        """Press several keys together (in order), then release them (in reverse)."""
        keys = list(keys)
        if any(key in self.held for key in keys):
            raise ValueError("release held keys before a chord")
        for key in keys:
            self.backend.key_down(key)
        self.sleep(self.tap_s)
        for key in reversed(keys):
            self.backend.key_up(key)

    def release_all(self) -> None:
        self.hold(())


# -- the bitmask used in recordings --------------------------------------------------------
KEY_ORDER: tuple[str, ...] = tuple(sorted(FIGHT_KEYS))  # bit i of a key mask is KEY_ORDER[i]


def to_mask(keys: Iterable[str]) -> int:
    return sum(1 << KEY_ORDER.index(key) for key in keys)


def from_mask(mask: int) -> frozenset[str]:
    return frozenset(key for i, key in enumerate(KEY_ORDER) if mask >> i & 1)


# -- reading the human's keyboard ------------------------------------------------------------
# Windows virtual-key codes of the fight keys. The numpad sends VK_NUMPAD4/5/6 with Num Lock on
# and VK_LEFT/CLEAR/RIGHT with it off; both are accepted (so are the arrow keys, which share
# those codes).
VIRTUAL_KEYS: dict[str, tuple[int, ...]] = {
    LEFT: (0x41,), RIGHT: (0x44,), UP: (0x57,), DOWN: (0x53,), JUMP: (0x20,), DODGE: (0x45,),
    LIGHT: (0x64, 0x25), HEAVY: (0x66, 0x27), THROW: (0x65, 0x0C),
}
STOP_KEY = 0x79  # F10: ends a recording


def _async_key_down(vk: int) -> bool:
    import ctypes

    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


class KeyReader:
    """Which fight keys the human holds, and which were pressed since the last ``take``.

    ``poll`` samples the keyboard once; ``start`` runs it every ``poll_s`` in a thread, so a tap
    between two recorded frames is still seen as a press. Presses matter: a jump or an attack
    starts on the press, not on the held state.
    """

    def __init__(self, is_down: Callable[[int], bool] = _async_key_down, poll_s: float = 0.001) -> None:
        import threading

        self.is_down = is_down
        self.poll_s = poll_s
        self._held: frozenset[str] = frozenset()
        self._pressed: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def poll(self) -> None:
        held = frozenset(key for key, codes in VIRTUAL_KEYS.items() if any(self.is_down(c) for c in codes))
        with self._lock:
            self._pressed |= held - self._held
            self._held = held

    def take(self) -> tuple[frozenset[str], frozenset[str]]:
        """(held now, pressed since the previous take)."""
        with self._lock:
            pressed, self._pressed = frozenset(self._pressed), set()
            return self._held, pressed

    def stop_requested(self) -> bool:
        return self.is_down(STOP_KEY)

    def start(self) -> None:
        import threading

        def run() -> None:
            while not self._stop.is_set():
                self.poll()
                time.sleep(self.poll_s)

        self._thread = threading.Thread(target=run, name="key-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
