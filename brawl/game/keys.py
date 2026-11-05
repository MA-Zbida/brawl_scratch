"""The only module that touches the keyboard.

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

    def force_release(self, keys: Iterable[str] = FIGHT_KEYS) -> None:
        """Send a release for every key, held or not. The game can drop a release like it drops a
        press, and then believes a key is still down (which blocks frame stepping)."""
        for key in sorted(keys):
            self.backend.key_up(key)
        self.held = frozenset()
