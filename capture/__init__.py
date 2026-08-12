"""Screen capture backends.

Duplication is the only backend used by default. GDI exists for machines where
duplication is impossible, and must be requested explicitly -- it runs at a
fraction of the frame rate, and control rate is this project's binding
constraint, so a silent fallback would quietly invalidate every timing
assumption downstream.
"""

from __future__ import annotations

from typing import Optional, Tuple

from capture.dxgi import DxcamFrameProvider
from capture.gdi import MssFrameProvider

__all__ = ["DxcamFrameProvider", "MssFrameProvider", "create_frame_provider"]


def create_frame_provider(
    region: Optional[Tuple[int, int, int, int]] = None,
    target_fps: int = 60,
    prefer: str = "auto",
):
    """Build the fastest capture backend that actually works on this machine.

    ``prefer`` is "auto" (duplication, then GDI), "dxcam", or "mss".
    """
    prefer = str(prefer).strip().lower()
    if prefer == "mss":
        return MssFrameProvider(region=region)

    try:
        return DxcamFrameProvider(region=region, target_fps=target_fps)
    except Exception as exc:
        if prefer == "dxcam":
            raise
        print(
            "[capture] DXGI duplication unavailable, falling back to GDI capture.\n"
            f"          reason: {str(exc).splitlines()[0]}\n"
            "          Run `python -m tools.check_capture --diagnose-order` to find out why."
        )
        return MssFrameProvider(region=region)
