"""Test bootstrap.

`env.py` imports the YOLO wrapper at module scope, which pulls in `ultralytics`.
That makes the environment untestable on any interpreter without the full live
stack installed — which is why no test imported `env` before this file existed.

Heavy, hardware-facing dependencies are stubbed **only when genuinely missing**,
so a machine with the real stack installed still exercises the real modules.
Capture, input injection and detection are all injectable on `BrawlDeepEnv`, so
tests that need them pass in their own fakes.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _install_stub(name: str, build: callable) -> None:
    """Register a stand-in module if the real one cannot be imported."""
    if name in sys.modules:
        return
    try:
        __import__(name)
    except Exception:
        sys.modules[name] = build()


def _ultralytics_stub() -> types.ModuleType:
    module = types.ModuleType("ultralytics")

    class YOLO:  # noqa: D401 - minimal stand-in
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "ultralytics is stubbed for testing; construct BrawlDeepEnv with an "
                "explicit extractor instead of relying on the default."
            )

    module.YOLO = YOLO
    return module


def _dxcam_stub() -> types.ModuleType:
    module = types.ModuleType("dxcam")

    def create(*args, **kwargs):
        raise RuntimeError("dxcam is stubbed for testing; inject a frame provider.")

    module.create = create
    return module


def _pydirectinput_stub() -> types.ModuleType:
    module = types.ModuleType("pydirectinput")
    module.KEYBOARD_MAPPING = {}
    module.PAUSE = 0
    module.FAILSAFE = False
    module.keyDown = lambda *a, **k: None
    module.keyUp = lambda *a, **k: None
    return module


def _keyboard_stub() -> types.ModuleType:
    module = types.ModuleType("keyboard")
    module.is_pressed = lambda *a, **k: False
    return module


_install_stub("ultralytics", _ultralytics_stub)
_install_stub("dxcam", _dxcam_stub)
_install_stub("pydirectinput", _pydirectinput_stub)
_install_stub("keyboard", _keyboard_stub)
