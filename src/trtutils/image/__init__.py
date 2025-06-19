# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Utilities for using TensorRT on images.

Submodules
----------
:mod:`common`
    Common utilities for image processing with TensorRT.
:mod:`kernels`
    Kernels for image processing with TensorRT.
:mod:`parallel`
    Parallel implementations of image models.
:mod:`preprocessors`
    Preprocessors for images.
:mod:`postprocessors`
    Postprocessors for images.
:mod:`onnx_models`
    Base ONNX models for creating 'micro-engines' for image processing.

Classes
-------
:class:`Classifer`
    Wrapper around classification models.
:class:`Detector`
    Wrapper around detection models.
:class:`SAHI`
    Simple implementation of SAHI.

"""

from __future__ import annotations

from . import common, kernels, onnx_models, parallel, postprocessors, preprocessors
from ._classifier import Classifier
from ._detector import Detector
from ._sahi import SAHI

__all__ = [
    "SAHI",
    "Classifier",
    "Detector",
    "common",
    "kernels",
    "onnx_models",
    "parallel",
    "postprocessors",
    "preprocessors",
]
