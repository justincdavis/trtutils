# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule containing tools for building TensorRT engines.

Classes
-------
:class:`EngineCalibrator`
    Calibrates an engine during quantization.
:class:`ImageBatcher`
    Batches images for calibration during engine building.

Functions
---------
:func:`build_engine`
    Build a TensorRT engine from an ONNX file.

"""

from __future__ import annotations

from ._batcher import ImageBatcher
from ._build import build_engine
from ._calibrator import EngineCalibrator

__all__ = [
    "EngineCalibrator",
    "ImageBatcher",
    "build_engine",
]
