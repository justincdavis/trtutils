# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Image test fixtures -- built engines, a CUDA stream, and device outputs for the image tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests._gpu_fixtures import engine_cache_path
from tests.conftest import DATA_DIR, ENGINES_DIR
from trtutils.builder import build_engine
from trtutils.core import create_stream, destroy_stream
from trtutils.core._buffer import Buffer, MemoryLocation

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

YOLOV10_ONNX = DATA_DIR / "yolov10" / "yolov10n_640.onnx"
YOLOV10_DYN_ONNX = DATA_DIR / "yolov10" / "yolov10n_640_dyn.onnx"


@pytest.fixture(scope="session")
def yolov10_engine(build_test_engine) -> Path:
    """Build and cache a YOLOv10n engine, skipping if the ONNX is missing."""
    if not YOLOV10_ONNX.exists():
        pytest.skip(f"missing {YOLOV10_ONNX}")
    return build_test_engine(YOLOV10_ONNX)


@pytest.fixture(scope="session")
def yolov10_dynamic_engine() -> Path:
    """Build a YOLOv10n engine with a (1, 4, 8) batch profile, skipping if the ONNX is missing."""
    if not YOLOV10_DYN_ONNX.exists():
        pytest.skip(f"missing {YOLOV10_DYN_ONNX} (trtutils download --model yolov10n --dynamic)")
    engine_path = engine_cache_path(YOLOV10_DYN_ONNX.stem, "1-4-8")
    if not engine_path.exists():
        ENGINES_DIR.mkdir(parents=True, exist_ok=True)
        shape = (3, 640, 640)
        build_engine(
            YOLOV10_DYN_ONNX,
            engine_path,
            optimization_level=1,
            shapes=[("images", ((1, *shape), (4, *shape), (8, *shape)))],
        )
    return engine_path


@pytest.fixture(scope="session")
def task_engine(build_test_engine) -> Callable[[str], Path]:
    """Build and cache a YOLOv11n task engine ("seg", "pose", "obb", "cls"), skipping if missing."""

    def _get(task: str, batch_size: int = 1) -> Path:
        suffix = f"-b{batch_size}" if batch_size > 1 else ""
        onnx_path = DATA_DIR / "yolov11_tasks" / f"yolov11n-{task}{suffix}.onnx"
        if not onnx_path.exists():
            pytest.skip(f"missing {onnx_path} (export with `trtutils download`)")
        return build_test_engine(onnx_path, batch_size=batch_size)

    return _get


@pytest.fixture
def cuda_stream():
    """Create a CUDA stream for the test, destroy after."""
    stream = create_stream()
    yield stream
    destroy_stream(stream)


@pytest.fixture
def to_device() -> Callable[[np.ndarray], Buffer]:
    """Copy a host array into a new device Buffer, as an engine output for the CUDA postprocessors."""
    return lambda array: Buffer.from_array(np.ascontiguousarray(array), MemoryLocation.DEVICE)
