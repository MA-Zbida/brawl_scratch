"""GDI screen capture — the slow, opt-in fallback."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class MssFrameProvider:
    """GDI-based screen capture fallback.

    DXGI Desktop Duplication is fast but refuses to run when the process is bound to
    an adapter that does not own the display output -- which on an Optimus laptop can
    happen simply because CUDA libraries were loaded, flipping the process to the
    discrete GPU before any capture code runs.

    ``mss`` uses GDI BitBlt instead, which has no such affinity requirement. It is
    slower (roughly 30-60 fps at 1080p versus 60+ for duplication), so it is a
    fallback rather than the default, but it is close enough to the ~40 Hz control
    loop to be usable and it works where duplication cannot.
    """

    def __init__(self, monitor: int = 1, region: Optional[Tuple[int, int, int, int]] = None):
        try:
            import mss
        except Exception as exc:
            raise RuntimeError(
                "mss is required for the capture fallback: python -m pip install mss"
            ) from exc

        self._sct = mss.mss()
        monitors = self._sct.monitors
        if region is not None:
            left, top, right, bottom = region
            self._region = {"left": left, "top": top,
                            "width": right - left, "height": bottom - top}
        else:
            idx = monitor if 0 <= monitor < len(monitors) else (1 if len(monitors) > 1 else 0)
            self._region = monitors[idx]
        self._last_good_frame = None
        print(f"[mss] capturing {self._region['width']}x{self._region['height']} (GDI fallback)")

    def get_frame(self):
        try:
            raw = self._sct.grab(self._region)
        except Exception:
            return self._last_good_frame
        # mss yields BGRA; the pipeline expects BGR.
        frame = np.asarray(raw, dtype=np.uint8)[:, :, :3]
        self._last_good_frame = frame
        return frame

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass
