# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Preprocessors for images.

Classes
--------
:class:`CPUPreprocessor`
    CPU-based preprocessor.
:class:`CUDAPreprocessor`
    CUDA-based preprocessor.
:class:`TRTPreprocessor`
    TensorRT-based preprocessor.
:class:`ImagePreprocessor`
    Abstract base class for image preprocessors.
:class:`GPUImagePreprocessor`
    Abstract base class for GPU-based image preprocessors.

Functions
---------
:func:`preprocess`
    Preprocess an image for a model.

"""

from __future__ import annotations

from ._abc import GPUImagePreprocessor, ImagePreprocessor
from ._cpu import CPUPreprocessor
from ._cuda import CUDAPreprocessor
from ._process import preprocess
from ._trt import TRTPreprocessor

__all__ = [
    "CPUPreprocessor",
    "CUDAPreprocessor",
    "GPUImagePreprocessor",
    "ImagePreprocessor",
    "TRTPreprocessor",
    "preprocess",
]
