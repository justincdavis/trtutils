# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Postprocessors for images.

Submodules
----------
classification
    Classification postprocessing module.
detection
    Detection postprocessing module.

Functions
---------
:func:`postprocess_detections`
    Postprocess the output of a detection model.
:func:`get_detections`
    Get the detections from the output of a detection model.
:func:`postprocess_classifications`
    Postprocess the output of a classification model.
:func:`get_classifications`
    Get the classifications from the output of a classification model.
:func:`postprocess_efficient_nms`
    Postprocess detections from model outputs.
:func:`decode_efficient_nms`
    Decode detections from model outputs.

"""

from __future__ import annotations

from . import classification, detection
from .classification._process import get_classifications, postprocess_classifications
from .detection._process import (
    get_detections,
    postprocess_detections,
    postprocess_efficient_nms,
    decode_efficient_nms,
)

__all__ = [
    "classification",
    "detection",
    "get_classifications",
    "get_detections",
    "postprocess_classifications",
    "postprocess_detections",
    "postprocess_efficient_nms",
    "decode_efficient_nms",
]
