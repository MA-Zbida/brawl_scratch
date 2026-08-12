#!/usr/bin/env python
"""Diagnose screen capture and the TensorRT engine — the two things that must work
before anything touches the live game.

Both failures in this area are misleading in the same way: they look like project bugs
but are environment mismatches.

**Capture.** DXGI Desktop Duplication only works on the adapter that owns the display
output. On a hybrid-graphics laptop the panel is driven by the Intel iGPU while CUDA
runs on the NVIDIA dGPU, so duplicating from device 0 fails with a bare COMError about
feature levels. This enumerates every adapter/output pair and reports which actually
capture. Picking the Intel adapter here does not affect CUDA — torch and TensorRT
address the NVIDIA GPU directly, not through D3D.

**Engine.** A TensorRT plan is tied to the TensorRT version that built it. Loading a
plan built by a different version fails, and Ultralytics reports it as a non-fatal
error, then hands back a model with no class-name metadata. Every detection then
carries a numeric id, the 3-class schema matches nothing, and the pipeline yields an
empty game state while reporting success.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EXPECTED_CLASSES = {"character", "indicator_self", "weapon"}


def check_capture() -> int:
    print("=" * 72)
    print("SCREEN CAPTURE")
    print("=" * 72)
    try:
        import dxcam
    except Exception as exc:
        print(f"dxcam not importable: {exc}")
        return 1

    try:
        print("adapters:"); print(dxcam.device_info())
    except Exception as exc:
        print(f"  could not enumerate adapters: {exc}")
    try:
        print("outputs:"); print(dxcam.output_info())
    except Exception as exc:
        print(f"  could not enumerate outputs: {exc}")

    working: list[tuple[int, int]] = []
    print(f"{'device':>8}{'output':>8}   result")
    for dev in range(4):
        for out in range(3):
            try:
                cam = dxcam.create(device_idx=dev, output_idx=out, output_color="BGR")
            except Exception as exc:
                msg = str(exc).strip().splitlines()[0][:60] if str(exc) else type(exc).__name__
                print(f"{dev:>8}{out:>8}   FAIL  {msg}")
                continue
            if cam is None:
                print(f"{dev:>8}{out:>8}   none returned")
                continue
            try:
                frame = cam.grab()
                shape = "no frame yet" if frame is None else f"{frame.shape}"
                print(f"{dev:>8}{out:>8}   OK    {shape}")
                working.append((dev, out))
            finally:
                try:
                    cam.release()
                except Exception:
                    pass

    print()
    if working:
        dev, out = working[0]
        print(f"Usable: {working}")
        print(f"DxcamFrameProvider auto-probes and will select device {dev}, output {out}.")
        print(f"To pin it: DxcamFrameProvider(device_idx={dev}, output_idx={out})")
        return 0

    print("No adapter/output combination can duplicate the desktop.")
    print("  - Set python.exe's GPU preference (Windows Graphics Settings) to the adapter")
    print("    driving your panel, usually the Intel iGPU. CUDA is unaffected.")
    print("  - Fullscreen-exclusive can block duplication; use borderless windowed.")
    print("  - Remote Desktop sessions cannot duplicate at all.")
    return 1


def check_engine(model: Path, rebuild: bool, imgsz: int) -> int:
    print()
    print("=" * 72)
    print("DETECTOR")
    print("=" * 72)

    try:
        import tensorrt
        print("tensorrt :", tensorrt.__version__)
    except Exception as exc:
        print("tensorrt : not importable —", exc)

    if rebuild:
        pt = model.with_suffix(".pt")
        if not pt.exists():
            print(f"Cannot rebuild: {pt} not found.")
            return 1
        print(f"\nrebuilding engine from {pt} at imgsz={imgsz} against the installed TensorRT...")
        from ultralytics import YOLO
        YOLO(str(pt)).export(format="engine", half=True, imgsz=imgsz, device=0)
        print("done — the new .engine sits beside the .pt")

    if not model.exists():
        print(f"\n{model} not found.")
        return 1

    print(f"\nloading {model} ...")
    try:
        from feature_extractor.yolo.extract import Extract
        ex = Extract(yolo_model=str(model), imgsz=imgsz)
    except RuntimeError as exc:
        print(f"\nFAILED:\n{exc}")
        return 1

    names = getattr(ex.yolo, "names", None)
    print("class names:", names)
    found = set(str(v) for v in (names.values() if isinstance(names, dict) else names or []))
    if EXPECTED_CLASSES.issubset(found):
        print("Schema OK — matches the 3-class detection schema.")
        return 0
    print(f"Schema MISMATCH: expected {sorted(EXPECTED_CLASSES)}, model reports {sorted(found)}")
    return 1


_PROBE = (
    "import sys\n"
    "{preamble}"
    "import dxcam\n"
    "try:\n"
    "    cam = dxcam.create(device_idx=0, output_idx=0, output_color='BGR')\n"
    "    print('OK' if cam is not None else 'NONE')\n"
    "except Exception as exc:\n"
    "    print('FAIL: ' + type(exc).__name__)\n"
)

_ORDERINGS = [
    ("dxcam alone", ""),
    ("after `import torch`", "import torch\n"),
    ("after `import ultralytics`", "import ultralytics\n"),
    ("after CUDA context", "import torch\ntorch.zeros(8, device='cuda').sum().item()\n"),
    ("after `import env`", "sys.path.insert(0, r'{root}')\nimport env\n"),
]


def diagnose_order() -> int:
    """Find which import first makes desktop duplication impossible.

    Each ordering runs in a fresh subprocess, because the whole hypothesis is that
    something loaded earlier changes the process's adapter affinity permanently --
    testing them in one process would contaminate every case after the first.
    """
    import subprocess

    print()
    print("=" * 72)
    print("IMPORT-ORDER DIAGNOSIS")
    print("=" * 72)
    print("Duplication can fail purely because CUDA libraries were loaded first:")
    print("on an Optimus laptop that flips the process onto the discrete GPU, which")
    print("does not own the display output.\n")

    root = str(Path(__file__).resolve().parent.parent)
    first_failure = None
    for label, preamble in _ORDERINGS:
        code = _PROBE.format(preamble=preamble.format(root=root))
        try:
            out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                 text=True, timeout=180)
            result = (out.stdout or out.stderr).strip().splitlines()[-1] if (out.stdout or out.stderr) else "no output"
        except Exception as exc:
            result = f"probe error: {exc}"
        print(f"  {label:<28} {result}")
        if first_failure is None and not result.startswith("OK"):
            first_failure = label

    print()
    if first_failure is None:
        print("Duplication survives every ordering. The failure is elsewhere --")
        print("check for another process holding the duplicator, or fullscreen-exclusive mode.")
        return 0
    if first_failure == "dxcam alone":
        print("Duplication fails even in a bare process: not an ordering problem.")
        print("Check Windows Graphics Settings for THIS interpreter specifically --")
        print(f"  {sys.executable}")
        print("A venv's python.exe is a separate binary from the base install, and the")
        print("per-app GPU preference is keyed on the executable path, so setting it on")
        print("one does not cover the other.")
    else:
        print(f"Duplication breaks at: {first_failure}")
        print("Nothing can be imported before capture is created. The GDI fallback")
        print("(MssFrameProvider) sidesteps this entirely; EnvConfig.capture_backend='mss'")
        print("forces it.")
    return 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path,
                   default=Path("feature_extractor/yolo/best.engine"))
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--rebuild-engine", action="store_true",
                   help="Re-export the .engine from the sibling .pt against the installed TensorRT")
    p.add_argument("--skip-capture", action="store_true")
    p.add_argument("--skip-engine", action="store_true")
    p.add_argument("--diagnose-order", action="store_true",
                   help="Find which import first breaks desktop duplication")
    return p.parse_args(argv)


def check_capture_after_cuda() -> int:
    """Re-probe capture once CUDA/TensorRT is initialised.

    This is the ordering trap. Initialising CUDA binds the process to the NVIDIA
    adapter, and duplicating the Intel-owned display output then fails -- so a probe
    that ran before the engine loaded can pass while the real pipeline fails on the
    identical device/output pair. `BrawlDeepEnv` creates capture first for this
    reason; this verifies the hazard is what we think it is.
    """
    print()
    print("=" * 72)
    print("CAPTURE AFTER CUDA INIT  (ordering check)")
    print("=" * 72)
    try:
        import dxcam
        import torch
    except Exception as exc:
        print(f"skipped: {exc}")
        return 0

    if not torch.cuda.is_available():
        print("skipped: no CUDA")
        return 0

    torch.zeros(8, device="cuda").sum().item()   # force context creation
    print("CUDA context created; retrying duplication on device 0, output 0 ...")
    try:
        cam = dxcam.create(device_idx=0, output_idx=0, output_color="BGR")
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {str(exc).splitlines()[0]}")
        print("\n  Confirms the ordering hazard: capture must be created BEFORE the")
        print("  detector. BrawlDeepEnv already does this. Any new entry point that")
        print("  touches CUDA before building the frame provider will hit this.")
        return 1
    if cam is None:
        print("  FAILED: create() returned None")
        return 1
    try:
        print("  OK — duplication still works after CUDA init on this machine.")
    finally:
        try:
            cam.release()
        except Exception:
            pass
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.diagnose_order:
        return diagnose_order()
    rc = 0
    if not args.skip_capture:
        rc |= check_capture()
    if not args.skip_engine:
        rc |= check_engine(args.model, args.rebuild_engine, args.imgsz)
    if not args.skip_capture:
        # Deliberately last: it must run with CUDA already initialised.
        rc |= check_capture_after_cuda()
    print()
    print("ALL CHECKS PASSED" if rc == 0 else "SOME CHECKS FAILED — see above")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
