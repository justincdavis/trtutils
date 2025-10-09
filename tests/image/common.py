# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

import trtutils

from .paths import ENGINE_PATHS, ONNX_PATHS

if TYPE_CHECKING:
    from pathlib import Path


def build_detector_engine(version: int = 8) -> Path:
    """
    Build a TensorRT detector engine from ONNX model.

    Parameters
    ----------
    version : int
        The YOLO version to use (7, 8, 9, 10, or 0 for YOLOX).
        Default is 8 (YOLOv8).

    Returns
    -------
    Path
        The compiled engine.

    """
    engine_path = ENGINE_PATHS[version]
    if engine_path.exists():
        return engine_path

    onnx_path = ONNX_PATHS[version]
    if not onnx_path.exists():
        err_msg = f"ONNX file not found: {onnx_path}"
        raise FileNotFoundError(err_msg)

    engine_path.parent.mkdir(parents=True, exist_ok=True)

    trtutils.builder.build_engine(
        onnx_path,
        engine_path,
        fp16=True,
    )

    return engine_path


def build_classifier_engine() -> Path:
    """
    Build a TensorRT classifier engine.

    Returns
    -------
    Path
        The compiled engine.

    """
    from pathlib import Path

    base = Path(__file__).parent.parent.parent
    engine_path = base / "data" / "engines" / "resnet18.engine"
    onnx_path = base / "data" / "resnet18.onnx"

    if engine_path.exists():
        return engine_path

    if not onnx_path.exists():
        err_msg = f"ONNX file not found: {onnx_path}. Run scripts/export_torchvision_classifier.py to create it."
        raise FileNotFoundError(err_msg)

    engine_path.parent.mkdir(parents=True, exist_ok=True)

    trtutils.builder.build_engine(
        onnx_path,
        engine_path,
        fp16=True,
    )

    return engine_path



