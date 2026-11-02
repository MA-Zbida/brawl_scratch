"""Non-blocking input sequences for semantic control qualification.

Reports injection, never claims the game accepted a move. No inference, sleeping,
or background input thread. The caller must advance frequently and log outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from control.semantic_actions import ACTIONS, BY_NAME, ControlContext, SemanticAction, action_mask


class InputSink(Protocol):
    def set_pressed(self, keys: set[str]) -> None: ...
    def key_down(self, key: str) -> None: ...
    def key_up(self, key: str) -> None: ...
    def reset(self) -> None: ...


@dataclass(frozen=True)
class ExecutionResult:
    action: str | None
    status: str
    reason: str = ""


@dataclass(frozen=True)
class ExecutionTiming:
    level_s: float = .020  # A level must survive a nominal 60 Hz input poll.
    gc_delay_s: float = .040  # Candidate neutral-dodge-to-attack separation, not verified game frame data.
    gc_deadline_s: float = .100  # Missed deadline aborts, rather than silently emitting an aerial attack.
    post_dodge_guard_s: float = .150  # Keep ordinary air attacks outside the candidate cancel guard.

    def __post_init__(self) -> None:
        values = (self.level_s, self.gc_delay_s, self.gc_deadline_s, self.post_dodge_guard_s)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Execution timing must be finite and positive")
        if not self.level_s <= self.gc_delay_s < self.gc_deadline_s <= self.post_dodge_guard_s:
            raise ValueError("Invalid execution timing order")


class SemanticExecutor:
    VERSION = "semantic_sequence_candidate_v2"

    def __init__(self, sink: InputSink, timing: ExecutionTiming | None = None) -> None:
        self.sink = sink
        self.timing = timing or ExecutionTiming()
        self._action: SemanticAction | None = None
        self._phase = "idle"
        self._taps: set[str] = set()
        self._mirrored = False
        self._deadline = 0.
        self._started = 0.
        self._last_now = float("-inf")
        self._last_dodge = float("-inf")

    @property
    def busy(self) -> bool:
        return self._action is not None

    def available_actions(self, context: ControlContext, *, now: float) -> tuple[bool, ...]:
        if self.busy:
            # NOOP here means advance the in-flight sequence, not cancel it.
            return (True,) + (False,) * (len(ACTIONS) - 1)
        mask = list(action_mask(context))
        if now - self._last_dodge < self.timing.post_dodge_guard_s:
            for action in ACTIONS:
                if action.context == "air" and action.button in {"num4", "num6"} and not action.gravity_cancel:
                    mask[action.id] = False
        return tuple(mask)

    def observation_state(self, *, now: float, mirrored: bool) -> tuple[float, ...]:
        active = [0.] * len(ACTIONS)
        if self._action is not None:
            active[self._action.id] = 1.
        phases = [float(self._phase == phase) for phase in ("idle", "dodge", "press", "release")]
        remaining = max(0., min(1., (self._deadline - now) / self.timing.post_dodge_guard_s)) if self.busy else 0.
        guard = max(0., min(1., 1. - (now - self._last_dodge) / self.timing.post_dodge_guard_s))
        # Relative to the CURRENT canonical frame, not an absolute left/right bit.
        flipped = float(self.busy and mirrored != self._mirrored)
        return tuple(active + phases + [remaining, guard, flipped])

    def _check_time(self, now: float) -> None:
        if not math.isfinite(now) or now < self._last_now:
            raise ValueError("Executor needs finite monotonic timestamps")
        self._last_now = now

    def _release_taps(self) -> None:
        for key in sorted(self._taps):
            self.sink.key_up(key)
        self._taps.clear()

    def _press(self, held: set[str], taps: set[str]) -> None:
        self.sink.set_pressed(held)
        for key in sorted(taps):
            self.sink.key_down(key)
            self._taps.add(key)

    def submit(self, name: str, context: ControlContext, *, now: float) -> ExecutionResult:
        self._check_time(now)
        action = BY_NAME[name]
        if self.busy:
            return ExecutionResult(name, "rejected", "busy")
        if not action_mask(context)[action.id]:
            return ExecutionResult(name, "rejected", "context_mask")
        if (action.context == "air" and action.button in {"num4", "num6"}
                and not action.gravity_cancel
                and now - self._last_dodge < self.timing.post_dodge_guard_s):
            return ExecutionResult(name, "rejected", "post_dodge_guard")
        self._action = action
        self._mirrored = context.mirrored
        self._started = now
        self._phase = "dodge" if action.gravity_cancel else "press"
        self._deadline = now + (self.timing.gc_delay_s if action.gravity_cancel else self.timing.level_s)
        held, taps = (set(), {"e"}) if action.gravity_cancel else action.keys(mirrored=self._mirrored)
        try:
            self._press(held, taps)
        except Exception:
            self.close()
            raise
        if "e" in taps:
            self._last_dodge = now
        return ExecutionResult(name, "started")

    def advance(self, context: ControlContext, *, now: float) -> ExecutionResult:
        self._check_time(now)
        action = self._action
        if action is None:
            return ExecutionResult(None, "idle")
        # Release is cleanup, not a new input decision. A lost triangle or a
        # landing cannot invalidate a pulse whose keys are already up.
        if self._phase == "release":
            if now < self._deadline:
                return ExecutionResult(action.name, "pending")
            self._action = None
            self._phase = "idle"
            return ExecutionResult(action.name, "completed")
        # Context remains mandatory before a deferred GC attack is injected.
        changed_context = (not context.controllable
                           or (action.context == "air" and context.grounded is not False)
                           or (action.context == "ground" and context.grounded is not True))
        if changed_context:
            injected = self._phase == "press"
            minimum_hold_met = injected and now >= self._deadline
            self.close()
            if minimum_hold_met:
                return ExecutionResult(action.name, "completed", "context_changed_after_input")
            if injected:
                return ExecutionResult(action.name, "cancelled", "context_changed_before_min_hold")
            return ExecutionResult(action.name, "cancelled", "context_changed")
        if self._phase == "dodge" and now - self._started > self.timing.gc_deadline_s:
            self.close()
            return ExecutionResult(action.name, "cancelled", "sequence_deadline")
        if now < self._deadline:
            return ExecutionResult(action.name, "pending")
        try:
            if self._phase == "dodge":
                self._release_taps()
                self._press(*action.keys(mirrored=self._mirrored))
                self._phase = "press"
                self._deadline = now + self.timing.level_s
                return ExecutionResult(action.name, "attack_injected")
            if self._phase == "press":
                self._release_taps()
                self.sink.set_pressed(set())
                self._phase = "release"
                # Do not backdate the release after a delayed caller tick.
                self._deadline = now + self.timing.level_s
                return ExecutionResult(action.name, "released")
            self._action = None
            self._phase = "idle"
            return ExecutionResult(action.name, "completed")
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self.sink.reset()
        self._taps.clear()
        self._action = None
        self._phase = "idle"

    def __enter__(self) -> SemanticExecutor:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
