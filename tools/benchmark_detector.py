#!/usr/bin/env python
"""Measure detector latency across models and resolutions.

Detection dominates the control loop. Measured on an RTX 3050 Ti with ``yolo26s-p2``
at ``imgsz=960``::

    total=36.36ms (27.51 hz)  frame=2.32ms  detect=33.72ms  everything-else=0.31ms

That is 93% of the step in the detector and under 1% in all the Python combined, so
detector latency is the only thing worth optimising.

**Run this with the game closed, and read the result accordingly.** Brawlhalla renders
on the same NVIDIA GPU as the detector -- the display is driven by the Intel iGPU, but
the game is not. Measured on this machine::

    benchmark, game closed :  18.2 ms
    live profile, game open:  33.7 ms      1.9x, entirely GPU contention

The game was running uncapped at 120-200 fps; it ticks at 60, so everything above that
is GPU time taken from the detector.

**Plan size does not indicate precision.** A 215 MB engine for a 9.4M-parameter model
looked like an FP32 export; it was already FP16 ("Converted 455/476 nodes to fp16") and
rebuilding moved latency 17.6 -> 18.2 ms, i.e. nothing. The bulk was duplicated shared
constants from the ONNX conversion. Read precision from the build log, never from the
file size.

**Reduce model size before resolution.** The self-indicator is roughly 15 px at
``imgsz=960`` and the P2 head exists to find it; shrinking the input can silently
destroy the feature that agent identity, and so every relational feature, depends on.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: A step must fit in this budget to sustain the target control rate.
TARGET_HZ = 40.0
STEP_BUDGET_MS = 1000.0 / TARGET_HZ
#: Measured non-detector cost per step (frame grab + state + reward + input).
OVERHEAD_MS = 2.6


def _weight_report(path: Path) -> None:
    """Report the plan size without inferring precision from it.

    A large .engine does NOT imply FP32. Measured here: a 215 MB plan for a 9.4M
    parameter model was already FP16 -- the bulk was duplicated shared constants
    from the ONNX conversion ("Total Weights Memory: 210 MB"), and rebuilding it
    changed latency by 3% . Precision has to be read from the build log, not
    guessed from the file size.
    """
    size_mb = path.stat().st_size / 1e6
    print(f"  file          : {path.name}  ({size_mb:.0f} MB)")


def benchmark(model_path: Path, sizes: Sequence[int], warmup: int, iters: int) -> None:
    import numpy as np
    import torch
    from ultralytics import YOLO

    print(f"\n{'=' * 72}")
    print(f"{model_path}")
    print("=" * 72)
    _weight_report(model_path)

    model = YOLO(str(model_path), task="detect")
    names = getattr(model, "names", None)
    print(f"  classes       : {names}")
    print(f"  gpu           : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    print(f"\n{'imgsz':>7}{'ms':>9}{'fps':>8}{'step ms':>10}{'ctrl hz':>9}   verdict")
    for size in sizes:
        # Real frames, not zeros: an empty image can short-circuit post-processing
        # and report a latency the live pipeline never sees.
        frame = (np.random.rand(size, size, 3) * 255).astype(np.uint8)

        try:
            for _ in range(warmup):
                model(frame, imgsz=size, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(iters):
                model(frame, imgsz=size, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            detect_ms = (time.perf_counter() - start) / iters * 1000.0
        except AssertionError:
            # TensorRT plans are built for one fixed input shape. Benchmarking a
            # different imgsz needs a separate engine, not a different argument.
            print(f"{size:>7}{'  n/a':>9}   engine is fixed-shape; rebuild at this imgsz to compare")
            continue
        except Exception as exc:
            print(f"{size:>7}{'  failed':>9}   {type(exc).__name__}: {str(exc).splitlines()[0][:40]}")
            continue

        step_ms = detect_ms + OVERHEAD_MS
        ctrl_hz = 1000.0 / step_ms
        if step_ms <= STEP_BUDGET_MS:
            verdict = f"OK (>= {TARGET_HZ:.0f} Hz)"
        elif ctrl_hz >= 30.0:
            verdict = "tight"
        else:
            verdict = "TOO SLOW"
        print(f"{size:>7}{detect_ms:>9.2f}{1000/detect_ms:>8.1f}{step_ms:>10.2f}{ctrl_hz:>9.1f}   {verdict}")


def breakdown(model_path: Path, iters: int = 50) -> None:
    """Split Extract.predict into preprocessing versus inference.

    The profiler's `detect` timer wraps the whole of `Extract.predict`, not just the
    forward pass, so a gap between this tool's number and the live one is not
    necessarily the GPU. Live capture is 1920x1080; the benchmark feeds an
    already-square array, skipping a full-resolution resample the real pipeline pays
    on every frame.

    Measured here: 18.2 ms benchmark against 33.1 ms live, with the game capped -- so
    roughly 15 ms is unaccounted for by inference. This attributes it.
    """
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO

    print(f"\n{'=' * 72}")
    print("PREPROCESSING BREAKDOWN")
    print("=" * 72)

    full = (np.random.rand(1080, 1920, 3) * 255).astype(np.uint8)
    model = YOLO(str(model_path), task="detect")

    def timeit(fn, n=iters):
        for _ in range(10):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - start) / n * 1000.0

    area = timeit(lambda: cv2.resize(full, (960, 540), interpolation=cv2.INTER_AREA))
    linear = timeit(lambda: cv2.resize(full, (960, 540), interpolation=cv2.INTER_LINEAR))
    nearest = timeit(lambda: cv2.resize(full, (960, 540), interpolation=cv2.INTER_NEAREST))

    print(f"\n  resize 1920x1080 -> 960x540")
    print(f"    INTER_AREA   (current) {area:>8.2f} ms")
    print(f"    INTER_LINEAR           {linear:>8.2f} ms   ({area - linear:+.2f} vs current)")
    print(f"    INTER_NEAREST          {nearest:>8.2f} ms   ({area - nearest:+.2f} vs current)")

    pre = cv2.resize(full, (960, 540), interpolation=cv2.INTER_AREA)
    square = (np.random.rand(960, 960, 3) * 255).astype(np.uint8)

    infer_square = timeit(lambda: model(square, imgsz=960, verbose=False))
    infer_rect = timeit(lambda: model(pre, imgsz=960, verbose=False))
    infer_full = timeit(lambda: model(full, imgsz=960, verbose=False))

    print(f"\n  inference by input shape")
    print(f"    960x960  (benchmark)   {infer_square:>8.2f} ms")
    print(f"    960x540  (pre-resized) {infer_rect:>8.2f} ms   letterboxed internally")
    print(f"    1920x1080 (raw frame)  {infer_full:>8.2f} ms   ultralytics resizes once")

    current = area + infer_rect
    direct = infer_full
    print(f"\n  end-to-end per frame")
    print(f"    current  (INTER_AREA + rect infer) {current:>8.2f} ms")
    print(f"    pass raw frame straight to model   {direct:>8.2f} ms   ({current - direct:+.2f})")
    print(f"    INTER_LINEAR + rect infer          {linear + infer_rect:>8.2f} ms   ({current - (linear + infer_rect):+.2f})")

    best = min(current, direct, linear + infer_rect)
    if best < current - 1.0:
        step = best + 1.5   # frame grab, capped game
        print(f"\n  Best option saves {current - best:.1f} ms -> ~{1000 / step:.0f} Hz control rate.")
        print("  A single resample also preserves more of the ~15 px self-indicator than two.")
    else:
        print("\nPreprocessing is not the gap. Inference itself is the cost;")
        print("  a smaller model (yolo26n-p2) is then the only remaining lever.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", type=str, default="feature_extractor/yolo/best.engine",
                   help="Comma-separated model paths to compare (.engine / .pt / .onnx)")
    p.add_argument("--sizes", type=str, default="640,768,960",
                   help="Comma-separated imgsz values")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--breakdown", action="store_true",
                   help="Attribute the gap between benchmark and live latency")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    paths = [Path(m.strip()) for m in args.models.split(",") if m.strip()]
    missing = [p for p in paths if not p.exists()]
    for p in missing:
        print(f"skipping (not found): {p}", file=sys.stderr)
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("No models to benchmark.", file=sys.stderr)
        return 1

    print(f"Budget: {STEP_BUDGET_MS:.1f} ms/step for {TARGET_HZ:.0f} Hz, "
          f"minus {OVERHEAD_MS:.1f} ms measured overhead "
          f"=> detection must fit in {STEP_BUDGET_MS - OVERHEAD_MS:.1f} ms")

    for path in paths:
        benchmark(path, sizes, args.warmup, args.iters)
        if args.breakdown:
            breakdown(path, args.iters)

    print(f"\n{'=' * 72}")
    print("Read this with the game CLOSED in mind")
    print("=" * 72)
    print("  Brawlhalla renders on the same NVIDIA GPU, so live latency is higher than")
    print("  anything measured here. Observed: 18.2 ms benchmark vs 33.7 ms live (1.9x).")
    print("  Cap the game to 60 fps and lower its graphics BEFORE changing the model.")
    print()
    print("Choosing between the options")
    print("=" * 72)
    print("  1. Cap the game's framerate first. It is free and it was worth ~1.9x here.")
    print("  2. If still too slow, drop model size (yolo26n-p2) before resolution.")
    print("  3. Only lower imgsz as a last resort: the self-indicator is ~15 px at 960,")
    print("     and identity for every relational feature depends on detecting it.")
    print("  4. Re-run tools/calibrate_indicator_geometry.py after any change that")
    print("     alters detection -- the association constants are resolution-sensitive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
