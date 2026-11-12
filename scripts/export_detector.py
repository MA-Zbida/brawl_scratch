"""Build the TensorRT engine of a trained detector, for this machine's GPU.

    python scripts/export_detector.py runs/detector/<name>/weights/best.pt

Writes best.engine beside the weights: half precision, one fixed 16:9 input (544 x 960 for a model
trained at 960 px), so the game frame is scaled once and nothing is padded to a square. The engine
is tied to this GPU, driver and TensorRT version: build it again after any of them changes. Half
precision can change a few borderline boxes, so score the engine like any candidate:

    python scripts/gate_p1.py runs/detector/<name>/weights/best.engine

Needs the ``tensorrt`` and ``onnx`` packages (README, Install).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", type=Path, help="best.pt of a trained detector")
    p.add_argument("--imgsz", type=int, default=None, help="input width; default: the size it was trained at")
    p.add_argument("--full-precision", action="store_true", help="32-bit floats instead of 16-bit")
    args = p.parse_args(argv)

    os.environ.setdefault("YOLO_AUTOINSTALL", "False")  # never let ultralytics install packages
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    width = args.imgsz or int(model.overrides.get("imgsz", 1280))
    height = -(-width * 9 // 16 // 32) * 32  # 16:9, rounded up to the network's stride
    engine = model.export(format="engine", imgsz=[height, width], half=not args.full_precision, batch=1,
                          simplify=False, device=0)
    print(f"\nengine: {engine} ({height} x {width})\nnext: python scripts/gate_p1.py {engine}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
