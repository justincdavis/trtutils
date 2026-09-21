# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/parallel/image/_detector.py -- Buffer handling in ParallelDetector."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import DATA_DIR
from trtutils.core import Buffer, MemoryLocation
from trtutils.parallel.image._detector import EngineInfo, ParallelDetector, _to_queueable_images

YOLOV10_ONNX = DATA_DIR / "yolov10" / "yolov10n_640.onnx"


def test_to_queueable_images_rejects_device_buffer(random_images) -> None:
    """A device Buffer raises TypeError instead of being queued for a worker thread."""
    img = random_images(1)[0]
    buf = Buffer.from_array(img, MemoryLocation.DEVICE)
    try:
        with pytest.raises(TypeError, match="device Buffer"):
            _to_queueable_images([img, buf])
    finally:
        buf.free()


def test_to_queueable_images_unwraps_host_buffer(random_images) -> None:
    """A host Buffer is unwrapped to its array; a plain ndarray passes through unchanged."""
    img = random_images(1)[0]
    buf = Buffer.from_array(img, MemoryLocation.HOST)
    try:
        result = _to_queueable_images([img, buf])
        assert result[0] is img
        assert isinstance(result[1], np.ndarray)
        np.testing.assert_array_equal(result[1], img)
    finally:
        buf.free()


def test_parallel_detector_rejects_device_buffer(build_test_engine, random_images) -> None:
    """ParallelDetector.submit_model raises TypeError when given a device Buffer."""
    if not YOLOV10_ONNX.exists():
        pytest.skip(f"missing {YOLOV10_ONNX}")

    engine_path = build_test_engine(YOLOV10_ONNX)
    detector = ParallelDetector([EngineInfo(engine_path=engine_path)], warmup=False)
    try:
        img = random_images(1)[0]
        buf = Buffer.from_array(img, MemoryLocation.DEVICE)
        try:
            with pytest.raises(TypeError, match="device Buffer"):
                detector.submit_model([buf], modelid=0)
        finally:
            buf.free()
    finally:
        detector.stop()
