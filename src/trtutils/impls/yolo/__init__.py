# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Implementations of YOLO object detectors for TRTModel.

Classes
-------
YOLO
    TRTModel implementation for YOLO object detectors

Functions
---------
get_detections
    Get the detections from a YOLO V7/8/9 network.
get_detections_v10
    Get the detections from a YOLO V10 network.
preprocess
    Preprocess some input for a YOLO network.
postprocess
    Postprocess the output for a YOLO V7/8/9 network.
postprocess_v10
    Postprocess the output for a YOLO V10 network.

"""

from __future__ import annotations

from ._process import get_detections, get_detections_v10, preprocess, postprocess, postprocess_v10

__all__ = [
    "get_detections",
    "get_detections_v10",
    "preprocess",
    "postprocess",
    "postprocess_v10",
]
