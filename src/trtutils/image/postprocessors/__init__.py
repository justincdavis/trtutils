# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Postprocessors for images.

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

"""

from __future__ import annotations

from ._classifier import get_classifications, postprocess_classifications
from ._detection import get_detections, postprocess_detections

__all__ = [
    "get_classifications",
    "get_detections",
    "postprocess_classifications",
    "postprocess_detections",
]
