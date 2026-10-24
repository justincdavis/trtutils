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
:func:`preprocess_text_detection`
    Preprocess an image for a DB text detection model.
:func:`preprocess_text_recognition`
    Preprocess text line images for a CTC text recognition model.
:func:`resize_text_detection`
    Resize an image to a valid DB text detection input size.
:func:`crop_text_region`
    Crop and rectify a detected text region.

"""

from __future__ import annotations

from ._cpu import CPUPreprocessor
from ._cuda import CUDAPreprocessor
from ._image_preproc import GPUImagePreprocessor, ImagePreprocessor
from ._ocr import (
    crop_text_region,
    preprocess_text_detection,
    preprocess_text_recognition,
    resize_text_detection,
)
from ._process import preprocess
from ._trt import TRTPreprocessor

__all__ = [
    "CPUPreprocessor",
    "CUDAPreprocessor",
    "GPUImagePreprocessor",
    "ImagePreprocessor",
    "TRTPreprocessor",
    "crop_text_region",
    "preprocess",
    "preprocess_text_detection",
    "preprocess_text_recognition",
    "resize_text_detection",
]
