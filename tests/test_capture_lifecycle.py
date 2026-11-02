"""The capture camera is borrowed, and must be returned.

dxcam keeps one camera per (device, output) *per process*. Any entry point that
builds more than one environment -- the retention evaluator builds one per phase --
gets "DXCamera instance already exists" on the second unless the first hands its
camera back. That failure surfaced only mid-run, after the first phase had already
been evaluated.
"""

from __future__ import annotations

import brawl.capture_first as capture_first


def test_take_then_release_allows_a_second_take() -> None:
    sentinel = object()
    original = capture_first._camera
    try:
        capture_first._camera = sentinel

        first = capture_first.take_camera()
        assert first is sentinel
        assert capture_first.take_camera() is None, "a camera can only be taken once"

        capture_first.release_camera(first)
        assert capture_first.take_camera() is sentinel, (
            "a returned camera must be available to the next environment"
        )
    finally:
        capture_first._camera = original


def test_releasing_none_is_a_no_op() -> None:
    original = capture_first._camera
    try:
        capture_first._camera = None
        capture_first.release_camera(None)
        assert capture_first.take_camera() is None
    finally:
        capture_first._camera = original


def test_release_does_not_clobber_a_camera_already_held() -> None:
    """Two providers closing must not overwrite whichever camera is parked."""
    held, stale = object(), object()
    original = capture_first._camera
    try:
        capture_first._camera = held
        capture_first.release_camera(stale)
        assert capture_first.take_camera() is held
    finally:
        capture_first._camera = original
