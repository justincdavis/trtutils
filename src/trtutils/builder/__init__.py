# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule containing tools for building TensorRT engines.

Submodules
----------
:mod:`hooks`
    Submodule containing hooks for building TensorRT engines.
:mod:`onnx`
    Submodule containing tools for working with ONNX models.

Classes
-------
:class:`EngineCalibrator`
    Calibrates an engine during quantization.
:class:`ImageBatcher`
    Batches images for calibration during engine building.
:class:`SyntheticBatcher`
    Generates synthetic data batches for calibration during engine building.
:class:`ProgressBar`
    Progress bar implementation for TensorRT engine building.

Functions
---------
:func:`build_engine`
    Build a TensorRT engine from an ONNX file.
:func:`build_dla_engine`
    Build an efficient TensorRT engine for DLA.
:func:`can_run_on_dla`
    Evaluate if the model can run on a DLA.
:func:`read_onnx`
    Read an ONNX file and get TensorRT objects.

"""

from __future__ import annotations

from . import hooks
from ._batcher import ImageBatcher, SyntheticBatcher
from ._build import build_engine
from ._calibrator import EngineCalibrator
from ._dla import build_dla_engine, can_run_on_dla
from ._onnx import read_onnx

__all__ = [
    "EngineCalibrator",
    "ImageBatcher",
    "SyntheticBatcher",
    "build_dla_engine",
    "build_engine",
    "can_run_on_dla",
    "hooks",
    "read_onnx",
]

import contextlib

with contextlib.suppress(ImportError):
    from . import onnx

    __all__ += ["onnx"]

with contextlib.suppress(AttributeError):
    from ._progress import ProgressBar

    __all__ += ["ProgressBar"]
