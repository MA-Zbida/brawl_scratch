"""Non-blocking handoff between perception and the control loop.

This module intentionally imports only the standard library. Importing concrete
capture or detector implementations here could pull in torch before the caller has
acquired DXGI duplication via ``capture_first``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Protocol, Self


class _FrameProvider(Protocol):
    def get_frame(self) -> Any: ...


class _Extractor(Protocol):
    def predict(self, frame: Any) -> list[dict]: ...


@dataclass(frozen=True, slots=True)
class PerceptionStamp:
    """Timing and freshness metadata for one perception result."""

    frame_id: int
    captured_at: float
    age_s: float
    is_stale: bool


@dataclass(frozen=True, slots=True)
class _Snapshot:
    frame: Any
    detections: list[dict]
    frame_id: int
    captured_at: float


class AsyncPerception:
    """Run capture and detection on one background thread.

    ``latest()`` returns the most recently completed pair immediately. Before the
    first detection completes, that pair is ``(None, [])`` with frame ID zero and
    infinite age.
    """

    def __init__(
        self,
        frame_provider: _FrameProvider,
        extractor: _Extractor,
        *,
        poll_interval: float = 0.0,
    ) -> None:
        if poll_interval < 0.0:
            raise ValueError("poll_interval must be non-negative")

        self._frame_provider = frame_provider
        self._extractor = extractor
        self._poll_interval = float(poll_interval)

        self._state_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._snapshot = _Snapshot(None, [], 0, 0.0)
        self._last_seen_frame_id: int | None = None
        self._error: Exception | None = None

    def start(self) -> None:
        """Start the sole perception worker.

        Repeated calls are harmless and cannot create a second detector thread.
        A stopped instance is intentionally not restartable because TensorRT engine
        lifecycle is owned by the injected extractor.
        """
        with self._lifecycle_lock:
            if self._thread is not None:
                return
            if self._stop_event.is_set():
                raise RuntimeError("AsyncPerception cannot be restarted after stop()")

            worker = threading.Thread(
                target=self._run,
                name="AsyncPerception",
                daemon=False,
            )
            self._thread = worker
            try:
                worker.start()
            except BaseException:
                self._thread = None
                raise

    def latest(self) -> tuple[Any, list[dict], PerceptionStamp]:
        """Return the current completed result without waiting for a new frame."""
        # Capture and inference happen outside this lock. The worker holds it only
        # while swapping a few references, so consumer latency is independent of
        # detector latency.
        with self._state_lock:
            error = self._error
            snapshot = self._snapshot
            is_stale = snapshot.frame_id == self._last_seen_frame_id
            if error is None:
                self._last_seen_frame_id = snapshot.frame_id

        if error is not None:
            # Re-raising on every read keeps a failed detector from silently feeding
            # the policy its last valid state forever.
            raise error

        now = time.perf_counter()
        age_s = (
            now - snapshot.captured_at
            if snapshot.captured_at > 0.0
            else float("inf")
        )
        stamp = PerceptionStamp(
            frame_id=snapshot.frame_id,
            captured_at=snapshot.captured_at,
            age_s=max(0.0, age_s),
            is_stale=is_stale,
        )
        return snapshot.frame, snapshot.detections, stamp

    def stop(self) -> None:
        """Request shutdown and wait for the worker; safe to call repeatedly."""
        self._stop_event.set()
        with self._lifecycle_lock:
            worker = self._thread

        if worker is not None and worker is not threading.current_thread():
            worker.join()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                frame = self._frame_provider.get_frame()
                captured_at = time.perf_counter()
                detections = self._extractor.predict(frame)

                # The pair is published only after inference completes, so consumers
                # can never combine detections from one frame with pixels from another.
                # Provider frames and detector outputs are treated as borrowed,
                # immutable buffers; copying a full screen frame here would spend the
                # latency this pipeline exists to recover.
                with self._state_lock:
                    frame_id = self._snapshot.frame_id + 1
                    self._snapshot = _Snapshot(
                        frame=frame,
                        detections=detections,
                        frame_id=frame_id,
                        captured_at=captured_at,
                    )

                if self._poll_interval > 0.0:
                    # Event.wait makes even a long polling interval interruptible.
                    self._stop_event.wait(self._poll_interval)
        # This thread is the exception boundary: provider and detector failures of
        # any ordinary type must cross it and become visible to the consumer.
        except Exception as error:  # noqa: BLE001
            with self._state_lock:
                self._error = error
            self._stop_event.set()


__all__ = ["AsyncPerception", "PerceptionStamp"]
