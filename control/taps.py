"""Poll-visible tap edges; consecutive requests never extend a press forever."""
from __future__ import annotations


class TapSequencer:
    VERSION = "poll_visible_edges_v2"
    # Both levels must survive a nominal 60 Hz poll, even if control runs faster.
    MIN_LEVEL_S = .020

    def __init__(self, hold_steps: int = 1) -> None:
        self.hold_steps = max(1, int(hold_steps))
        self.held: dict[str, tuple[int, float]] = {}
        self.released_until: dict[str, float] = {}

    def advance(self, requested: set[str], *, now: float) -> tuple[set[str], set[str]]:
        down, up = set(), set()
        for key, (remaining, until) in list(self.held.items()):
            if remaining > 0 or now < until:
                self.held[key] = (max(0, remaining - 1), until)
            else:
                up.add(key)
                del self.held[key]
                self.released_until[key] = now + self.MIN_LEVEL_S
        for key in requested:
            if key not in self.held and now >= self.released_until.get(key, float('-inf')):
                down.add(key)
                self.held[key] = (self.hold_steps - 1, now + self.MIN_LEVEL_S)
        # No request queue: an obsolete jump/throw must not fire after a policy change.
        return down, up

    def release(self, *, now: float) -> set[str]:
        keys = set(self.held)
        for key in keys:
            self.released_until[key] = now + self.MIN_LEVEL_S
        self.held.clear()
        return keys
