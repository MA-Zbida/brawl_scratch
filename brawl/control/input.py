"""Keyboard injection.

Tap-type inputs are HELD across a latch window rather than pulsed: the game polls
at roughly 60 Hz, so a keyDown/keyUp pair emitted inside one control step is
frequently shorter than a poll interval and gets dropped entirely.
"""

from __future__ import annotations

from typing import Iterable

KeySet = Iterable[str]


class NullInputController:
    def set_pressed(self, keys: KeySet) -> None:
        return

    def key_down(self, key: str) -> None:
        return

    def key_up(self, key: str) -> None:
        return

    def tap(self, keys: KeySet) -> None:
        return

    def reset(self) -> None:
        return


class PyDirectInputController:
    _HOLDABLE_KEYS: frozenset[str] = frozenset({"a", "d", "s"})

    def __init__(self):
        try:
            import pydirectinput
        except Exception as exc:
            raise RuntimeError("pydirectinput is required for PyDirectInputController") from exc
        self._pydirectinput = pydirectinput
        # Add explicit numpad aliases so actions can target numpad-only keys.
        mapping = getattr(self._pydirectinput, "KEYBOARD_MAPPING", None)
        if isinstance(mapping, dict):
            mapping.setdefault("num4", 0x4B)
            mapping.setdefault("num5", 0x4C)
            mapping.setdefault("num6", 0x4D)
            mapping.setdefault("num_4", mapping["num4"])
            mapping.setdefault("num_5", mapping["num5"])
            mapping.setdefault("num_6", mapping["num6"])
        if hasattr(self._pydirectinput, "PAUSE"):
            self._pydirectinput.PAUSE = 0
        if hasattr(self._pydirectinput, "FAILSAFE"):
            self._pydirectinput.FAILSAFE = False
        if hasattr(self._pydirectinput, "MINIMUM_DURATION"):
            setattr(self._pydirectinput, "MINIMUM_DURATION", 0)
        if hasattr(self._pydirectinput, "MINIMUM_SLEEP"):
            setattr(self._pydirectinput, "MINIMUM_SLEEP", 0)
        if hasattr(self._pydirectinput, "DARWIN_CATCH_UP_TIME"):
            setattr(self._pydirectinput, "DARWIN_CATCH_UP_TIME", 0)
        self._pressed: set[str] = set()
        # Tap-type keys are tracked separately from held movement keys so that
        # set_pressed() never releases a tap that is still inside its hold window.
        self._tap_held: set[str] = set()

    def set_pressed(self, keys: KeySet) -> None:
        target = set(keys)

        # Release holdable keys only when they are currently pressed.
        # Avoiding redundant keyUp() calls significantly reduces input overhead.
        for key in self._HOLDABLE_KEYS:
            if key not in target and key in self._pressed:
                self._pydirectinput.keyUp(key)
                self._pressed.discard(key)

        # Release any other tracked key not in target
        for key in list(self._pressed):
            if key not in target and key not in self._HOLDABLE_KEYS:
                self._pydirectinput.keyUp(key)
                self._pressed.discard(key)

        # Press keys that should be held
        for key in target:
            if key not in self._pressed:
                self._pydirectinput.keyDown(key)
                self._pressed.add(key)

    def key_down(self, key: str) -> None:
        if key not in self._tap_held:
            self._pydirectinput.keyDown(key)
            self._tap_held.add(key)

    def key_up(self, key: str) -> None:
        if key in self._tap_held:
            self._pydirectinput.keyUp(key)
            self._tap_held.discard(key)

    def tap(self, keys: KeySet) -> None:
        """Immediate press-release.

        Retained for compatibility only. Brawlhalla samples input at ~60 Hz, so a
        keyDown/keyUp pair emitted inside a single step is frequently shorter than
        one poll interval and is dropped by the game. The control loop uses
        key_down/key_up with a multi-step hold window instead.
        """
        for key in keys:
            self._pydirectinput.keyDown(key)
        for key in keys:
            self._pydirectinput.keyUp(key)

    def reset(self) -> None:
        for key in list(self._tap_held):
            self._pydirectinput.keyUp(key)
        self._tap_held.clear()
        for key in list(self._pressed):
            self._pydirectinput.keyUp(key)
        for key in self._HOLDABLE_KEYS:
            self._pydirectinput.keyUp(key)
        self._pressed.clear()
