# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Utilities for using TensorRT on images.

Submodules
----------
:mod:`kernels`
    Kernels for image processing with TensorRT.
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
:class:`DepthEstimator`
    Wrapper around depth estimation models.
:class:`Detector`
    Wrapper around detection models.
:class:`HandInteractionDetector`
    Wrapper around hand-object interaction models.
:class:`SAHI`
    SAHI wrapper for slicing aided inference.
:class:`ImageModel`
    Base class for models which process images.

Type Aliases
------------
:data:`ImageInput`
    A single image, either an HWC uint8 ``np.ndarray`` or a ``Buffer``
    (host or device) holding one.

"""

from __future__ import annotations

from . import (
    interfaces,
    kernels,
    onnx_models,
    postprocessors,
    preprocessors,
    sahi,
)
from ._classifier import Classifier
from ._depth_estimator import DepthEstimator
from ._detector import Detector
from ._hand_interaction import HandInteractionDetector
from ._image_model import ImageModel
from .interfaces import ImageInput
from .sahi import SAHI

__all__ = [
    "SAHI",
    "Classifier",
    "DepthEstimator",
    "Detector",
    "HandInteractionDetector",
    "ImageInput",
    "ImageModel",
    "interfaces",
    "kernels",
    "onnx_models",
    "postprocessors",
    "preprocessors",
    "sahi",
]
