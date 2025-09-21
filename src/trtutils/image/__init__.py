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
:mod:`sahi`
    SAHI (Slicing Aided Hyper Inference) for object detection.

Classes
-------
:class:`Classifer`
    Wrapper around classification models.
:class:`Detector`
    Wrapper around detection models.
:class:`SAHI`
    SAHI wrapper for slicing aided inference.
:class:`ImageModel`
    Base class for models which process images.

"""

from __future__ import annotations

from . import (
    common,
    interfaces,
    kernels,
    onnx_models,
    parallel,
    postprocessors,
    preprocessors,
    sahi,
)
from ._classifier import Classifier
from ._detector import Detector
from ._image_model import ImageModel
from .sahi import SAHI

__all__ = [
    "SAHI",
    "Classifier",
    "Detector",
    "ImageModel",
    "common",
    "interfaces",
    "kernels",
    "onnx_models",
    "parallel",
    "postprocessors",
    "preprocessors",
    "sahi",
]
