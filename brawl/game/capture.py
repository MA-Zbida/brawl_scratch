"""Screen capture through DXGI desktop duplication.

**Import this module before anything that loads torch.** On a hybrid-graphics laptop the display
belongs to the integrated GPU; importing torch loads the CUDA libraries, which moves the process
to the discrete GPU, and DXGI then refuses to duplicate a display that adapter does not own
(``COMError: The specified device interface or feature level is not supported``). A duplicator
created before that keeps working, so this module creates one at import time.

``Screen.grab(fresh=True)`` waits for a frame the game presented after the previous grab. In the
stepped regime that is the frame produced by the last step: the paused game presents nothing new
until it is stepped.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

DISABLE_ENV_VAR = "BRAWL_NO_EARLY_CAPTURE"  # set to 1 to skip the early grab (tests, CI)

_camera: Optional[Any] = None
_error: Optional[BaseException] = None
_torch_was_loaded = False


def _acquire() -> None:
    global _camera, _error, _torch_was_loaded
    _torch_was_loaded = "torch" in sys.modules
    try:
        import dxcam
    except Exception as exc:  # noqa: BLE001 - reported by status()
        _error = exc
        return
    try:
        devices = max(1, len(dxcam.device_info().strip().splitlines()))
        outputs = max(1, len(dxcam.output_info().strip().splitlines()))
    except Exception:  # noqa: BLE001
        devices, outputs = 2, 2
    # The adapter that owns the display is often not device 0 on hybrid graphics: try them all.
    for device in range(devices):
        for output in range(outputs):
            try:
                camera = dxcam.create(device_idx=device, output_idx=output, output_color="BGR")
            except Exception as exc:  # noqa: BLE001
                _error = exc
                continue
            if camera is not None:
                _camera, _error = camera, None
                return


def status() -> str:
    if _camera is not None:
        return "duplicator held" + (" (torch was already loaded; ordering is fragile)" if _torch_was_loaded else "")
    if _torch_was_loaded:
        return ("no duplicator: torch was imported before brawl.game.capture. Make "
                "`import brawl.game.capture` the first import of the entry point.")
    return f"no duplicator: {type(_error).__name__ if _error else 'no adapter worked'}: {_error}"


class Camera(Protocol):
    def grab(self) -> Optional[Any]: ...


@dataclass
class Frame:
    image: Any            # H x W x 3, BGR
    fresh: bool           # False if no new frame arrived before the timeout
    captured_at: float    # perf_counter time of the grab


class Screen:
    """The one place frames come from."""

    def __init__(self, camera: Optional[Camera] = None, *, timeout_s: float = 0.25,
                 poll_s: float = 0.001, now: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if camera is None:
            camera = _camera
            if camera is None:
                raise RuntimeError(status())
        self.camera = camera
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.now = now
        self.sleep = sleep
        self._last: Optional[Any] = None

    def grab(self, *, fresh: bool = True) -> Frame:
        """Return the newest frame. With ``fresh``, wait (up to ``timeout_s``) for one presented
        after the previous grab; otherwise return the last frame seen if nothing new arrived."""
        deadline = self.now() + self.timeout_s
        while True:
            image = self.camera.grab()  # None when nothing changed since the last grab
            if image is not None:
                self._last = image
                return Frame(image, True, self.now())
            if not fresh or self.now() >= deadline:
                if self._last is None:
                    raise RuntimeError("no frame captured yet")
                return Frame(self._last, False, self.now())
            self.sleep(self.poll_s)


if os.environ.get(DISABLE_ENV_VAR, "") not in ("1", "true", "True") and "pytest" not in sys.modules:
    _acquire()
