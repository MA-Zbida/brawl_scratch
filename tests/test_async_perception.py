"""Behavioral tests for the asynchronous perception handoff."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from perception.async_pipeline import AsyncPerception


class CountingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def get_frame(self) -> int:
        self.calls += 1
        return self.calls


class SlowExtractor:
    def __init__(self, delay_s: float = 0.02) -> None:
        self.calls = 0
        self.delay_s = delay_s
        self.entered = threading.Event()

    def predict(self, frame: Any) -> list[dict]:
        self.calls += 1
        self.entered.set()
        time.sleep(self.delay_s)
        return [{"frame": frame}]


def _next_frame(
    pipeline: AsyncPerception,
    *,
    after: int = 0,
    timeout_s: float = 1.0,
):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        result = pipeline.latest()
        if result[2].frame_id > after:
            return result
        time.sleep(0.001)
    pytest.fail(f"no frame newer than {after} arrived within {timeout_s:.3f}s")


def test_latest_does_not_wait_for_slow_detector() -> None:
    extractor = SlowExtractor(delay_s=0.15)
    pipeline = AsyncPerception(CountingProvider(), extractor, poll_interval=0.001)
    pipeline.start()
    assert extractor.entered.wait(timeout=0.5)

    try:
        started_at = time.perf_counter()
        frame, detections, stamp = pipeline.latest()
        elapsed_s = time.perf_counter() - started_at

        assert elapsed_s < 0.02
        assert frame is None
        assert detections == []
        assert stamp.frame_id == 0
    finally:
        pipeline.stop()


def test_repeat_reads_expose_staleness_and_keep_frame_id() -> None:
    pipeline = AsyncPerception(CountingProvider(), SlowExtractor(delay_s=0.03))
    pipeline.start()

    try:
        first = _next_frame(pipeline)
        repeated = pipeline.latest()

        assert first[2].is_stale is False
        assert repeated[2].is_stale is True
        assert repeated[2].frame_id == first[2].frame_id
    finally:
        pipeline.stop()


def test_frame_ids_are_monotonic_and_match_completed_detections() -> None:
    extractor = SlowExtractor(delay_s=0.004)
    pipeline = AsyncPerception(CountingProvider(), extractor)
    pipeline.start()
    observed_ids: list[int] = []

    try:
        deadline = time.perf_counter() + 0.08
        while time.perf_counter() < deadline:
            observed_ids.append(pipeline.latest()[2].frame_id)
            time.sleep(0.001)
    finally:
        pipeline.stop()

    final_id = pipeline.latest()[2].frame_id
    assert observed_ids == sorted(observed_ids)
    assert final_id == extractor.calls
    assert final_id >= 2


class GatedProvider:
    """Hold the second capture so its timestamp is predictably recent."""

    def __init__(self) -> None:
        self.calls = 0
        self.second_capture_waiting = threading.Event()
        self.release_second = threading.Event()

    def get_frame(self) -> int:
        self.calls += 1
        if self.calls == 2:
            self.second_capture_waiting.set()
            self.release_second.wait(timeout=1.0)
        return self.calls


class ImmediateExtractor:
    def predict(self, frame: Any) -> list[dict]:
        return [{"frame": frame}]


def test_age_grows_between_reads_and_drops_after_refresh() -> None:
    provider = GatedProvider()
    pipeline = AsyncPerception(provider, ImmediateExtractor())
    pipeline.start()

    try:
        first = _next_frame(pipeline)
        assert provider.second_capture_waiting.wait(timeout=0.5)
        time.sleep(0.03)
        older = pipeline.latest()[2]

        provider.release_second.set()
        refreshed = _next_frame(pipeline, after=first[2].frame_id)[2]

        assert older.age_s > first[2].age_s
        assert refreshed.age_s < older.age_s
    finally:
        provider.release_second.set()
        pipeline.stop()


class ExplodingExtractor:
    def __init__(self) -> None:
        self.raised = threading.Event()

    def predict(self, frame: Any) -> list[dict]:
        self.raised.set()
        raise ValueError("detector failed")


def test_worker_exception_surfaces_without_deadlocking_latest() -> None:
    extractor = ExplodingExtractor()
    pipeline = AsyncPerception(CountingProvider(), extractor)
    pipeline.start()
    assert extractor.raised.wait(timeout=0.5)
    worker = pipeline._thread
    assert worker is not None
    worker.join(timeout=0.5)
    assert not worker.is_alive()

    started_at = time.perf_counter()
    with pytest.raises(ValueError, match="detector failed"):
        pipeline.latest()
    assert time.perf_counter() - started_at < 0.02

    pipeline.stop()


def test_stop_joins_thread_is_idempotent_and_thread_is_not_daemon() -> None:
    pipeline = AsyncPerception(CountingProvider(), SlowExtractor(delay_s=0.01))
    pipeline.start()
    worker = pipeline._thread

    assert worker is not None
    assert worker.daemon is False
    pipeline.start()
    assert pipeline._thread is worker
    pipeline.stop()
    assert not worker.is_alive()
    pipeline.stop()
    assert not worker.is_alive()


def test_context_manager_cleans_up_when_body_raises() -> None:
    pipeline = AsyncPerception(CountingProvider(), SlowExtractor(delay_s=0.01))

    with pytest.raises(RuntimeError, match="body failed"), pipeline:
        worker = pipeline._thread
        assert worker is not None and worker.is_alive()
        raise RuntimeError("body failed")

    assert worker is not None
    assert not worker.is_alive()


def test_consumer_throughput_is_not_limited_by_detector() -> None:
    extractor = SlowExtractor(delay_s=0.02)
    iterations = 0

    with AsyncPerception(CountingProvider(), extractor) as pipeline:
        _next_frame(pipeline)
        deadline = time.perf_counter() + 0.15
        while time.perf_counter() < deadline:
            pipeline.latest()
            iterations += 1

    assert extractor.calls >= 2
    assert iterations > extractor.calls * 100


def test_poll_interval_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="poll_interval"):
        AsyncPerception(CountingProvider(), ImmediateExtractor(), poll_interval=-0.01)
