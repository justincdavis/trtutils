# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.conftest import DATA_DIR
from trtutils.download import download

if TYPE_CHECKING:
    from pathlib import Path


YOLOV10_ONNX = DATA_DIR / "yolov10" / "yolov10n_640.onnx"


@pytest.fixture(scope="session")
def yolov10_onnx_path() -> Path:
    """Path to the YOLOv10n ONNX model, downloaded if not present."""
    if not YOLOV10_ONNX.exists():
        YOLOV10_ONNX.parent.mkdir(parents=True, exist_ok=True)
        download("yolov10n", YOLOV10_ONNX, imgsz=640, make_static=True)
    return YOLOV10_ONNX


@pytest.fixture(scope="session")
def yolov10_engine_path(build_test_engine, yolov10_onnx_path) -> Path:
    """Build and return path to a YOLOv10n test engine."""
    return build_test_engine(yolov10_onnx_path)
