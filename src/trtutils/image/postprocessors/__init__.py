# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Postprocessors for images.

Functions
---------
:func:`get_classifications`
    Get the classifications from the output of a classification model.
:func:`get_depth_maps`
    Get the depth maps from the output of a depth estimation model.
:func:`postprocess_classifications`
    Postprocess the output of a classification model.
:func:`postprocess_depth`
    Postprocess the output of a depth estimation model.
:func:`get_detections`
    Get the detections from unified postprocessed outputs.
:func:`get_interactions`
    Pair hands with objects from postprocessed hand-object interaction outputs.
:func:`postprocess_hand_interactions`
    Postprocess the output of a hand-object interaction model.
:func:`postprocess_yolov10`
    Postprocess the output of a YOLO-v10 model.
:func:`postprocess_rfdetr`
    Postprocess the output of a RF-DETR model.
:func:`postprocess_detr`
    Postprocess the output of a DETR-based model.
:func:`postprocess_detr_lbs`
    Postprocess the output of a DETR-based model with LBS output order.
:func:`postprocess_rtdetrv3`
    Postprocess the output of an RT-DETR v3 model.
:func:`postprocess_efficient_nms`
    Postprocess the output of an EfficientNMS model.

"""

from __future__ import annotations

from ._classifier import get_classifications, postprocess_classifications
from ._depth import get_depth_maps, postprocess_depth
from ._detection import (
    get_detections,
    postprocess_detr,
    postprocess_detr_lbs,
    postprocess_efficient_nms,
    postprocess_rfdetr,
    postprocess_rtdetrv3,
    postprocess_yolov10,
)
from ._hand_interaction import HandInteraction, get_interactions, postprocess_hand_interactions

__all__ = [
    "HandInteraction",
    "get_classifications",
    "get_depth_maps",
    "get_detections",
    "get_interactions",
    "postprocess_classifications",
    "postprocess_depth",
    "postprocess_detr",
    "postprocess_detr_lbs",
    "postprocess_efficient_nms",
    "postprocess_hand_interactions",
    "postprocess_rfdetr",
    "postprocess_rtdetrv3",
    "postprocess_yolov10",
]
