# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Image test fixtures -- a built YOLOv10 engine shared by the image model tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import tensorrt as trt

from tests.conftest import DATA_DIR, ENGINES_DIR
from trtutils.builder import build_engine

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
    engine_path = ENGINES_DIR / f"yolov10n_640_dyn_b1-8_{trt.__version__}.engine"
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
