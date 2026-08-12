"""Detection extraction must not scale its GPU transfers with detection count.

Reading ``box.cls`` / ``box.xywhn`` / ``box.conf`` inside a ``for box in res.boxes``
loop costs three device-to-host synchronisations *per detection*. Each stalls the
pipeline until the GPU drains, so the cost grows with how many objects are on screen
-- the price rises exactly when the scene gets busy, which is when latency matters.

This was invisible to every latency benchmark, because those call the model and throw
the results away: the transfers only happen when something reads the tensors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extractor.yolo.extract import Extract


class CountingTensor:
    """Minimal tensor stand-in that records every device-to-host transfer."""

    def __init__(self, array, counter):
        self._array = np.asarray(array)
        self._counter = counter

    def cpu(self):
        self._counter["transfers"] += 1
        return self

    def numpy(self):
        return self._array

    def astype(self, dtype):
        return self._array.astype(dtype)

    def __len__(self):
        return len(self._array)

    def __getitem__(self, idx):
        return CountingTensor(self._array[idx], self._counter)

    def __int__(self):
        self._counter["transfers"] += 1
        return int(self._array)

    def __float__(self):
        self._counter["transfers"] += 1
        return float(self._array)


class FakeBoxes:
    def __init__(self, xywhn, cls, conf, counter):
        self.xywhn = CountingTensor(xywhn, counter)
        self.cls = CountingTensor(cls, counter)
        self.conf = CountingTensor(conf, counter)
        self._n = len(cls)

    def __len__(self):
        return self._n


class FakeResult:
    def __init__(self, n, counter):
        rng = np.random.default_rng(0)
        self.boxes = FakeBoxes(
            rng.random((n, 4)).astype(np.float32),
            np.arange(n) % 3,
            np.full(n, 0.9, dtype=np.float32),
            counter,
        )
        self.names = {0: "character", 1: "indicator_self", 2: "weapon"}


def _extract_with(n_detections):
    """Run conversion over n detections, counting transfers."""
    counter = {"transfers": 0}
    extractor = Extract.__new__(Extract)
    extractor.class_names = None
    extractor.yolo = type("M", (), {"names": {0: "character", 1: "indicator_self", 2: "weapon"}})()

    detections = extractor._results_to_detections([FakeResult(n_detections, counter)])
    return detections, counter["transfers"]


def test_transfers_do_not_grow_with_detection_count():
    """The regression: 3 transfers per box became 15 stalls in a busy scene."""
    _, few = _extract_with(1)
    _, many = _extract_with(8)

    assert many == few, (
        f"{few} transfers for 1 detection but {many} for 8 -- the cost scales with "
        "scene complexity, which is precisely when it must not"
    )


def test_transfer_count_is_small_and_fixed():
    _, transfers = _extract_with(8)
    assert transfers <= 3, f"expected at most 3 transfers (xywhn, cls, conf), got {transfers}"


@pytest.mark.parametrize("n", [0, 1, 3, 8])
def test_conversion_is_correct_for_any_count(n):
    detections, _ = _extract_with(n)

    assert len(detections) == n
    for det in detections:
        assert det["class_name"] in {"character", "indicator_self", "weapon"}
        assert len(det["bbox"]) == 4
        assert isinstance(det["confidence"], float)
        assert all(isinstance(v, float) for v in det["bbox"])


def test_empty_result_is_handled():
    detections, transfers = _extract_with(0)
    assert detections == []
    assert transfers == 0, "no detections should mean no transfers at all"


def test_no_results_object_is_handled():
    extractor = Extract.__new__(Extract)
    extractor.class_names = None
    extractor.yolo = type("M", (), {"names": {}})()
    assert extractor._results_to_detections(None) == []
    assert extractor._results_to_detections([]) == []
