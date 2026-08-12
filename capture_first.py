"""Acquire the screen duplicator before anything loads CUDA.

**Import this before torch, ultralytics, stable-baselines3, or ``env``.**

On an Optimus laptop the display output belongs to the Intel iGPU. Merely importing
``torch`` loads the NVIDIA CUDA libraries, which flips the process onto the discrete
GPU -- and DXGI Desktop Duplication then refuses to duplicate an output that adapter
does not own, failing with::

    COMError: The specified device interface or feature level is not supported

Measured on this machine with ``python -m tools.check_capture --diagnose-order``::

    dxcam alone                  OK
    after `import torch`         FAIL
    after `import ultralytics`   FAIL
    after CUDA context           FAIL

The flip happens at import, not at CUDA-context creation, so no amount of reordering
*inside* the environment helps -- by the time any of its code runs, torch is already
loaded. A duplicator acquired beforehand keeps working, so this module grabs one at
import time and hands it to ``DxcamFrameProvider`` later.

The alternative was falling back to GDI capture, which sustains a fraction of the
frame rate. Control rate is the binding constraint on this project, so that trade is
not worth making.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

__all__ = ["take_camera", "status", "acquire", "is_available"]

_camera: Optional[Any] = None
_error: Optional[BaseException] = None
_selected: Optional[tuple[int, int]] = None
_torch_already_loaded: bool = False
_attempted: bool = False

#: Set to 1 to skip the eager grab (tests, headless machines, CI).
DISABLE_ENV_VAR = "BRAWL_NO_EARLY_CAPTURE"


def _candidates(dxcam) -> list[tuple[int, int]]:
    try:
        n_devices = max(1, len(dxcam.device_info().strip().splitlines()))
    except Exception:
        n_devices = 2
    try:
        n_outputs = max(1, len(dxcam.output_info().strip().splitlines()))
    except Exception:
        n_outputs = 2
    return [(d, o) for d in range(n_devices) for o in range(n_outputs)]


def acquire(force: bool = False) -> bool:
    """Create the duplicator now. Returns True when one is held."""
    global _camera, _error, _selected, _torch_already_loaded, _attempted

    if _camera is not None and not force:
        return True
    if _attempted and not force:
        return False
    _attempted = True

    _torch_already_loaded = "torch" in sys.modules

    try:
        import dxcam
    except Exception as exc:
        _error = exc
        return False

    for dev, out in _candidates(dxcam):
        try:
            cam = dxcam.create(device_idx=dev, output_idx=out, output_color="BGR")
        except Exception as exc:
            _error = exc
            continue
        if cam is not None:
            _camera, _selected, _error = cam, (dev, out), None
            return True

    return False


def take_camera() -> Optional[Any]:
    """Hand over the held camera exactly once; ownership transfers to the caller."""
    global _camera
    cam, _camera = _camera, None
    return cam


def is_available() -> bool:
    return _camera is not None


def status() -> str:
    if _camera is not None:
        dev, out = _selected or (-1, -1)
        warn = ""
        if _torch_already_loaded:
            warn = ("  (torch was ALREADY imported when this ran -- it worked anyway, but the "
                    "ordering is fragile; import capture_first earlier)")
        return f"held: device {dev}, output {out}{warn}"
    if _torch_already_loaded:
        return (
            "NOT HELD, and torch was already imported before this module ran.\n"
            "  That is almost certainly the cause: importing torch loads the NVIDIA CUDA\n"
            "  libraries, which moves the process to the discrete GPU, and the display\n"
            "  output belongs to the integrated one.\n"
            "  Fix: make `import capture_first` the FIRST import in the entry point,\n"
            "  before torch / ultralytics / stable_baselines3 / env."
        )
    return f"NOT HELD: {type(_error).__name__ if _error else 'no adapter/output worked'}: {_error}"


# Eager acquisition on import — the entire point of the module. Skipped under pytest
# and when explicitly disabled, so test runs do not hold a real duplicator.
if os.environ.get(DISABLE_ENV_VAR, "") not in ("1", "true", "True") and "pytest" not in sys.modules:
    acquire()
