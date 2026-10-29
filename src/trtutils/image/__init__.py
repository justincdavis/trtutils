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
:mod:`types`
    Type aliases for image model results.

Classes
-------
:class:`AutotuneTrial`
    One timed configuration from :func:`autotune`.
:class:`Classifer`
    Wrapper around classification models.
:class:`DepthEstimator`
    Wrapper around depth estimation models.
:class:`Detector`
    Wrapper around detection models.
:class:`HandInteractionDetector`
    Wrapper around hand-object interaction models.
:class:`TextDetector`
    Wrapper around DB text detection models (e.g. PP-OCR).
:class:`TextRecognizer`
    Wrapper around CTC text recognition models (e.g. PP-OCR).
:class:`OCR`
    Two-stage OCR pipeline of a TextDetector and a TextRecognizer.
:class:`SAHI`
    SAHI wrapper for slicing aided inference.
:class:`OBBDetector`
    Wrapper around oriented bounding box detection models.
:class:`PoseEstimator`
    Wrapper around pose estimation models.
:class:`Segmenter`
    Wrapper around instance segmentation models.
:class:`ImageModel`
    Base class for models which process images.

Functions
---------
:func:`autotune`
    Time an image model's runtime options on your own images.

Constants
---------
:data:`COCO91_TO_COCO80`
    COCO category ids (1-90) to 80-class indices, for use as a detector ``class_map``.

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
from ._autotune import AutotuneTrial, autotune
from ._class_maps import COCO91_TO_COCO80
from ._classifier import Classifier
from ._depth_estimator import DepthEstimator
from ._detector import Detector
from ._hand_interaction import HandInteractionDetector
from ._image_model import ImageModel
from ._obb_detector import OBBDetector
from ._ocr import OCR, TextDetector, TextRecognizer
from ._pose_estimator import PoseEstimator
from ._segmenter import Segmenter
from .sahi import SAHI

__all__ = [
    "COCO91_TO_COCO80",
    "OCR",
    "SAHI",
    "AutotuneTrial",
    "Classifier",
    "DepthEstimator",
    "Detector",
    "HandInteractionDetector",
    "ImageModel",
    "OBBDetector",
    "PoseEstimator",
    "Segmenter",
    "TextDetector",
    "TextRecognizer",
    "autotune",
    "interfaces",
    "kernels",
    "onnx_models",
    "postprocessors",
    "preprocessors",
    "sahi",
]
