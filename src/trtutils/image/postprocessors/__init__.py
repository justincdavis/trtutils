# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Postprocessors for images.

Functions
---------
:func:`get_classifications`
    Get the classifications from the output of a classification model.
:func:`postprocess_classifications`
    Postprocess the output of a classification model.
:func:`get_detections`
    Get the detections from unified postprocessed outputs.
:func:`postprocess_yolov10`
    Postprocess the output of a YOLO-v10 model.
:func:`postprocess_rfdetr`
    Postprocess the output of a RF-DETR model.
:func:`postprocess_detr`
    Postprocess the output of a DETR-based model.
:func:`postprocess_efficient_nms`
    Postprocess the output of an EfficientNMS model.

"""

from __future__ import annotations

from ._classifier import get_classifications, postprocess_classifications
from ._detection import (
    postprocess_yolov10,
    postprocess_rfdetr,
    postprocess_detr,
    postprocess_efficient_nms,
    get_detections,
)

__all__ = [
    # classification
    "get_classifications",
    "postprocess_classifications",
    # detection
    "postprocess_yolov10",
    "postprocess_rfdetr",
    "postprocess_detr",
    "postprocess_efficient_nms",
    "get_detections",
]
