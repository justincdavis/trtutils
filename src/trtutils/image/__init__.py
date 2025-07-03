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
:mod:`interfaces`
    Interfaces for image models.
:mod:`onnx_models`
    Base ONNX models for creating 'micro-engines' for image processing.

Classes
-------
:class:`Classifer`
    Wrapper around classification models.
:class:`Detector`
    Wrapper around detection models.

"""

from __future__ import annotations

from . import common, kernels, onnx_models, parallel, postprocessors, preprocessors
from ._classifier import Classifier
from ._detector import Detector

__all__ = [
    "Classifier",
    "Detector",
    "common",
    "kernels",
    "interfaces",
    "onnx_models",
    "parallel",
    "postprocessors",
    "preprocessors",
]
