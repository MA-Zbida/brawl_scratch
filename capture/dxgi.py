"""DXGI Desktop Duplication capture — the fast path."""

from __future__ import annotations

from typing import Optional, Tuple

import capture_first


class DxcamFrameProvider:
    """Desktop capture via DXGI Desktop Duplication.

    Duplication only works on the adapter that actually **owns the display output**.
    On a hybrid-graphics laptop the panel is usually driven by the Intel iGPU while
    CUDA work runs on the NVIDIA dGPU, so the correct capture adapter is often *not*
    device 0, and a hardcoded index fails with:

        COMError: The specified device interface or feature level is not supported

    Rather than making the caller guess, unspecified indices are probed until a
    combination works. CUDA is unaffected by the choice -- torch and TensorRT talk to
    the NVIDIA GPU directly, not through the D3D adapter this picks.
    """

    def __init__(self, region: Optional[Tuple[int, int, int, int]] = None,
                 output_idx: Optional[int] = None, device_idx: Optional[int] = None,
                 target_fps: int = 60):
        try:
            import dxcam
        except Exception as exc:
            raise RuntimeError("dxcam is required for DxcamFrameProvider") from exc

        # Prefer a duplicator acquired before torch was loaded; creating one now
        # fails on hybrid graphics once the process has been moved to the dGPU.
        self._camera = capture_first.take_camera()
        if self._camera is None:
            self._camera = self._create_camera(dxcam, device_idx, output_idx)
        else:
            print(f"[dxcam] using pre-acquired duplicator ({capture_first.status()})")
        # Prefer non-blocking capture mode when available: it returns the latest
        # frame immediately (possibly duplicate) instead of waiting for a fresh one.
        # This is critical for RL control loops where policy steps can be faster
        # than the capture thread.
        if region is None:
            try:
                self._camera.start(target_fps=target_fps, video_mode=True)
            except TypeError:
                self._camera.start(target_fps=target_fps)
        else:
            try:
                self._camera.start(region=region, target_fps=target_fps, video_mode=True)
            except TypeError:
                self._camera.start(region=region, target_fps=target_fps)
        self._last_good_frame = None

    @staticmethod
    def _candidate_indices(dxcam, device_idx, output_idx) -> list[tuple[int, int]]:
        """Device/output pairs to try, honouring anything explicitly requested."""
        if device_idx is not None and output_idx is not None:
            return [(int(device_idx), int(output_idx))]

        try:
            n_devices = len(dxcam.device_info().strip().splitlines())
        except Exception:
            n_devices = 2
        try:
            n_outputs = max(1, len(dxcam.output_info().strip().splitlines()))
        except Exception:
            n_outputs = 2

        devices = [int(device_idx)] if device_idx is not None else list(range(max(1, n_devices)))
        outputs = [int(output_idx)] if output_idx is not None else list(range(max(1, n_outputs)))
        return [(d, o) for d in devices for o in outputs]

    @classmethod
    def _create_camera(cls, dxcam, device_idx, output_idx):
        errors: list[str] = []
        for dev, out in cls._candidate_indices(dxcam, device_idx, output_idx):
            try:
                camera = dxcam.create(device_idx=dev, output_idx=out, output_color="BGR")
            except Exception as exc:
                errors.append(f"  device {dev}, output {out}: {type(exc).__name__}: {exc}")
                continue
            if camera is not None:
                print(f"[dxcam] capturing on device {dev}, output {out}")
                return camera
            errors.append(f"  device {dev}, output {out}: create() returned None")

        detail = "\n".join(errors) if errors else "  (no device/output combinations enumerated)"
        raise RuntimeError(
            "Could not start desktop duplication on any adapter/output.\n"
            f"{detail}\n\n"
            "Desktop duplication only works on the adapter driving the display. On a\n"
            "hybrid-graphics laptop that is usually the Intel iGPU, not the NVIDIA GPU.\n"
            "Run `python -m tools.check_capture` to see what this machine exposes.\n"
            "If duplication is blocked entirely, set the app's GPU preference for\n"
            "python.exe in Windows Graphics Settings to the adapter that owns the panel;\n"
            "this does not affect CUDA, which addresses the NVIDIA GPU directly."
        )

    def get_frame(self):
        """Return latest frame, falling back to the last good one if DXCam has no new frame."""
        frame = self._camera.get_latest_frame()
        if frame is not None:
            self._last_good_frame = frame
            return frame
        return self._last_good_frame

    def close(self) -> None:
        self._camera.stop()
