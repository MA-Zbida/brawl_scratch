#!/usr/bin/env python
"""Find out why inference on a *captured* frame costs more than on a synthetic one.

Measured on this machine, same engine and resolution throughout:

    synthetic 1920x1080 array -> predict   ~17 ms
    live capture              -> predict   ~33 ms

Everything else has been ruled out: the engine is FP16, capping the game's framerate
moved only the frame-grab time, and preprocessing is 0.85 ms. The remaining difference
between the two measurements is the *provenance of the array*, not the work done on it.

A capture buffer is not an ordinary numpy array. It can be

* **non-contiguous** -- a BGRA capture sliced to BGR keeps a stride of 4, so every
  per-pixel pass reads with gaps and loses cache locality;
* **mapped device memory** -- a staging texture surfaced as an array, where reads are
  uncached and cross PCIe at a fraction of RAM bandwidth.

Either makes the resize-and-normalise pass that precedes inference far more expensive
while the GPU work itself is unchanged. Both are fixed the same way: one bulk copy into
ordinary memory, which reads the buffer sequentially exactly once.

This script reports the buffer's actual layout and times inference on it directly
against a copy, so the answer is measured rather than assumed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

import capture_first  # noqa: E402,F401  (must precede torch)


def _describe(name: str, arr) -> None:
    import numpy as np

    flags = arr.flags
    owns = arr.base is None
    print(f"  {name}")
    print(f"    shape/dtype   : {arr.shape} {arr.dtype}")
    print(f"    strides       : {arr.strides}")
    print(f"    C-contiguous  : {flags['C_CONTIGUOUS']}")
    print(f"    owns its data : {owns}   (False = a view into someone else's buffer)")
    expected = arr.shape[2] if arr.ndim == 3 else 1
    if arr.ndim == 3 and arr.strides[1] != expected:
        print(f"    *** stride {arr.strides[1]} for {expected} channels: this is a sliced view;")
        print("        every per-pixel pass reads with gaps")


def _time(fn, warmup: int, iters: int) -> float:
    import torch

    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=Path("feature_extractor/yolo/best.engine"))
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=40)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    import numpy as np
    from ultralytics import YOLO

    from capture import DxcamFrameProvider

    print("=" * 72)
    print("FRAME PATH DIAGNOSIS")
    print("=" * 72)

    provider = DxcamFrameProvider()
    frame = None
    for _ in range(60):                      # let the capture thread produce one
        frame = provider.get_frame()
        if frame is not None:
            break
        time.sleep(0.02)
    if frame is None:
        print("No frame captured. Is anything on screen?", file=sys.stderr)
        return 1

    print("\nbuffer layout")
    _describe("captured frame", frame)

    contiguous = np.ascontiguousarray(frame)
    synthetic = (np.random.rand(*frame.shape) * 255).astype(np.uint8)
    _describe("after ascontiguousarray", contiguous)

    model = YOLO(str(args.model), task="detect")
    kw = dict(imgsz=args.imgsz, verbose=False)

    # Time the copy on a held frame. Calling get_frame() in the timed loop would
    # measure DXCam's target_fps cadence (get_latest_frame blocks for a NEW frame),
    # not the copy -- which is what made an earlier run report 16 ms for a no-op.
    copy_ms = _time(lambda: np.ascontiguousarray(frame), args.warmup, args.iters)

    print("\ninference by array provenance")
    rows = [
        ("captured frame (as-is)", lambda: model(frame, **kw)),
        ("captured frame, copied", lambda: model(contiguous, **kw)),
        ("synthetic array", lambda: model(synthetic, **kw)),
    ]
    results = {}
    for label, fn in rows:
        try:
            ms = _time(fn, args.warmup, args.iters)
        except Exception as exc:
            print(f"  {label:<26}   failed: {type(exc).__name__}: {exc}")
            continue
        results[label] = ms
        print(f"  {label:<26} {ms:>8.2f} ms")

    print(f"\n  bulk copy alone            {copy_ms:>8.2f} ms")

    print("\n" + "=" * 72)
    raw = results.get("captured frame (as-is)")
    copied = results.get("captured frame, copied")
    synth = results.get("synthetic array")

    if raw is None or synth is None:
        print("Not enough measurements to conclude.")
    elif copied is not None and copied < raw - 2.0:
        print(f"THE BUFFER IS THE PROBLEM. Copying saves {raw - copied:.1f} ms/frame.")
        print("The capture buffer is expensive to read repeatedly; one bulk copy reads it")
        print("sequentially once and every later pass runs at RAM speed.")
        print("\nFix: copy in the frame provider, right after get_frame().")
    elif raw - synth > 3.0:
        print(f"Captured frames cost {raw - synth:.1f} ms more than synthetic ones, and copying")
        print("does not recover it. The cost is in acquiring the frame rather than reading it")
        print("-- try a smaller capture region, or a different target_fps.")
    else:
        print("Captured and synthetic frames cost the same here.")
        print("The live gap is NOT the frame buffer. Remaining suspects: contention from the")
        print("policy/optimiser sharing the GPU, or thermal throttling during a long session")
        print("-- re-run the live profile immediately after a cold start to check.")

    provider.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
