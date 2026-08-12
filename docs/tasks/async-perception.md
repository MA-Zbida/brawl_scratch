# Task brief: asynchronous perception pipeline

Hand this to another agent verbatim. It is scoped to **new files only** so it cannot
collide with concurrent work on `env.py`.

---

## Context

This repo trains a Brawlhalla agent from screen pixels on a single laptop. The control
loop runs at roughly **40 steps/second** and that ceiling is the project's binding
constraint — every design decision exists to buy sample efficiency, because wall-clock
time cannot be bought.

Right now one environment step does this **synchronously**:

```
grab frame  ->  YOLO inference  ->  update state  ->  reward  ->  policy  ->  inject keys
```

Detection dominates: roughly 10–20 ms of a ~25 ms budget. Because it blocks, the control
rate is pinned to the detector rate even though the policy itself costs well under a
millisecond.

## Goal

Decouple capture and detection from the control loop, so the policy steps at its own rate
while perception refreshes independently in the background. Expected gain is roughly 2× on
control rate.

## Deliverable

**One new module** — `perception/async_pipeline.py` — plus tests. Do **not** modify
`env.py`; integration is a separate one-line change made afterwards.

```python
class AsyncPerception:
    """Runs capture + detection on a background thread.

    Maintains the most recent (frame, detections) pair. `latest()` never blocks:
    it returns whatever is current, possibly the same result twice if the consumer
    is faster than the detector.
    """

    def __init__(self, frame_provider, extractor, *, poll_interval: float = 0.0): ...
    def start(self) -> None: ...
    def latest(self) -> tuple[Any, list[dict], PerceptionStamp]: ...
    def stop(self) -> None: ...
    def __enter__(self); def __exit__(self, *exc)
```

`PerceptionStamp` must carry at minimum:

- `frame_id: int` — monotonic counter, increments once per completed detection
- `captured_at: float` — `time.perf_counter()` when the frame was grabbed
- `age_s: float` — seconds between capture and the `latest()` call
- `is_stale: bool` — True when the consumer has already seen this `frame_id`

## Hard constraints

**1. Import order is load-bearing.** `capture_first` must be imported before anything that
pulls in `torch`. On this hardware, importing torch loads the NVIDIA CUDA libraries, which
moves the process to the discrete GPU, after which DXGI duplication of the integrated-GPU
display output becomes impossible. Measured — see `capture_first.py` and
`python -m tools.check_capture --diagnose-order`. If your module imports anything heavy,
`import capture_first` goes first.

**2. Staleness must be observable, never hidden.** The consumer has to be able to tell that
it received a repeated frame. Silently handing back duplicates as if they were fresh would
inject fake zero-velocity frames into the state estimator — position unchanged across two
"different" steps reads as "the character stopped moving". That corrupts every velocity
feature and is invisible in training. This is the single most important requirement.

**3. One detector, one thread.** The TensorRT engine is not thread-safe and the GPU has
4 GB. Exactly one worker thread, one extractor instance.

**4. Clean shutdown.** No daemon threads left running, no hang on exit. `stop()` must be
idempotent and safe to call from `__exit__` after an exception.

**5. Do not modify** `env.py`, `capture/`, `control/`, `feature_extractor/`, `action_space.py`,
or anything under `train/`. New files only, plus `tests/test_async_perception.py`.

## Tests required

Use fake providers — no GPU, no game, no real capture. The existing suite runs headless and
yours must too.

1. `latest()` returns without blocking even when the detector is slower than the consumer.
2. A consumer faster than the detector sees `is_stale=True` on repeat reads, and the same
   `frame_id`.
3. `frame_id` increases monotonically and never skips backwards.
4. `age_s` grows between refreshes and drops on a new frame.
5. An exception inside the worker does not deadlock the consumer; it surfaces (re-raised on
   the next `latest()`, or exposed as an `error` attribute — your choice, but it must not be
   swallowed).
6. `stop()` joins the thread; calling it twice is safe; `with AsyncPerception(...)` cleans up
   on exception.
7. Throughput: with a detector stubbed at 20 ms and a consumer looping freely, the consumer
   completes markedly more iterations than the detector produces frames. This is the whole
   point of the change — assert it.

Run with:

```
python -m pytest tests/test_async_perception.py -q
```

## Style

Match the existing code. Type hints, `from __future__ import annotations`, comments that
explain *why* rather than restating the code. Where a decision has a non-obvious rationale
(thread-safety, staleness semantics, buffer ownership), say so in a comment — this codebase
has repeatedly been bitten by constraints that were invisible in the source.

## Explicitly out of scope

- Wiring into `env.py`
- Any change to the action space, observation schema, or reward
- Multiprocessing (threads only — the GIL is released during TensorRT inference and cv2
  resize, which is where the time goes)
- Frame interpolation or extrapolation. If perception is stale, say it is stale; do not
  invent state.
