"""Construction order in BrawlDeepEnv.

Screen capture must be created before the detector. On a hybrid-graphics laptop the
display output belongs to the Intel iGPU while CUDA runs on the NVIDIA dGPU;
initialising CUDA/TensorRT first binds the process to the NVIDIA adapter and
duplicating the Intel-owned output then fails with DXGI_ERROR_UNSUPPORTED.

The symptom is badly misleading -- "The specified device interface or feature level
is not supported on this system" reads like a driver or hardware problem, and a
capture probe run on its own passes on the very same device/output pair. Nothing in
the code makes the dependency visible, so this test is what keeps a future tidy-up
from reordering two innocuous-looking lines and resurrecting it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import BrawlDeepEnv, EnvConfig, NullInputController


def test_frame_provider_is_constructed_before_the_extractor():
    """Order is load-bearing; assert it on the source rather than trusting comments."""
    import inspect

    source = inspect.getsource(BrawlDeepEnv.__init__)
    assert source.index("self.frame_provider = frame_provider") < source.index(
        "self.extractor = extractor"
    ), "capture must be created before CUDA touches the process"


def test_default_construction_order_observed_at_runtime():
    """Injected providers are still built in the documented order."""
    events: list[str] = []

    class RecordingFrames:
        def __init__(self):
            events.append("capture")

        def get_frame(self):
            return np.zeros((1080, 1920, 3), dtype=np.uint8)

        def close(self):
            pass

    class RecordingExtractor:
        def __init__(self):
            events.append("detector")

        def predict(self, frame):
            return []

    # Construct in the same order BrawlDeepEnv would, then hand them over.
    frames = RecordingFrames()
    extractor = RecordingExtractor()

    env = BrawlDeepEnv(
        extractor=extractor,
        frame_provider=frames,
        input_controller=NullInputController(),
        stocks_health_provider=None,
        config=EnvConfig(ui_regions=None),
    )

    assert events == ["capture", "detector"]
    assert env.frame_provider is frames


def test_capture_failure_message_names_the_real_cause():
    """A bare COMError sent the last debugging session down the wrong path."""
    import types

    dxcam = types.SimpleNamespace(
        device_info=lambda: "Device[0]:<Intel>",
        output_info=lambda: "Device[0] Output[0]: Res:(1920,1080)",
        create=lambda **kwargs: (_ for _ in ()).throw(
            OSError("The specified device interface or feature level is not supported")
        ),
    )

    from env import DxcamFrameProvider

    provider = DxcamFrameProvider.__new__(DxcamFrameProvider)
    with pytest.raises(RuntimeError) as excinfo:
        provider._create_camera(dxcam, None, None)

    message = str(excinfo.value)
    assert "hybrid-graphics" in message
    assert "check_capture" in message
    assert "device 0, output 0" in message, "must report which combinations were tried"
