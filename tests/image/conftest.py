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
from trtutils.builder.hooks import yolo_efficient_nms_hook

if TYPE_CHECKING:
    from pathlib import Path

YOLOV10_ONNX = DATA_DIR / "yolov10" / "yolov10n_640.onnx"
YOLOV10_DYN_ONNX = DATA_DIR / "yolov10" / "yolov10n_640_dyn.onnx"
YOLOV8_ONNX = DATA_DIR / "yolov8" / "yolov8n_640.onnx"


@pytest.fixture(scope="session")
def yolov10_engine(build_test_engine) -> Path:
    """Build and cache a YOLOv10n engine, skipping if the ONNX is missing."""
    if not YOLOV10_ONNX.exists():
        pytest.skip(f"missing {YOLOV10_ONNX}")
    return build_test_engine(YOLOV10_ONNX)


@pytest.fixture(scope="session")
def yolov10_dynamic_engine() -> Path:
    """
    Build a YOLOv10n engine with a (1, 4, 8) dynamic batch profile.

    Needs the dynamic-axes export from the download tool
    (``trtutils download --model yolov10n --imgsz 640 --dynamic``); the
    static export bakes the batch dimension into internal ops, so it cannot
    be made dynamic after the fact. Skips if that ONNX is missing.
    """
    if not YOLOV10_DYN_ONNX.exists():
        pytest.skip(f"missing {YOLOV10_DYN_ONNX} (export with `trtutils download --dynamic`)")

    engine_path = ENGINES_DIR / f"yolov10n_640_dyn_b8_{trt.__version__}.engine"
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
def yolov8n_effnms_engine() -> Path:
    """
    Build a YOLOv8n engine with an EfficientNMS_TRT plugin grafted on at build time.

    Needs the raw (no-NMS) yolov8n ONNX export; skips if it is missing.
    """
    if not YOLOV8_ONNX.exists():
        pytest.skip(f"missing {YOLOV8_ONNX}")

    engine_path = ENGINES_DIR / f"yolov8n_640_effnms_{trt.__version__}.engine"
    if not engine_path.exists():
        ENGINES_DIR.mkdir(parents=True, exist_ok=True)
        build_engine(
            YOLOV8_ONNX,
            engine_path,
            hooks=[yolo_efficient_nms_hook(num_classes=80)],
            optimization_level=1,
            fp16=True,
        )
    return engine_path
