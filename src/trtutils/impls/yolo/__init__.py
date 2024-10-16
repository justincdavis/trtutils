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
    Get the detections from a YOLO network.
preprocess
    Preprocess some input for a YOLO network.
postprocess
    Postprocess the output for a YOLO network.

"""

from __future__ import annotations

from ._process import get_detections, postprocess, preprocess
from ._yolo import YOLO

__all__ = [
    "YOLO",
    "get_detections",
    "postprocess",
    "preprocess",
]
