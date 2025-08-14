# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Detection postprocessing module.

Classes
-------
DetectionPostprocessor
    Abstract base class for detection postprocessors.
CPUDetectionPostprocessor
    CPU-based detection postprocessor.
CUDADetectionPostprocessor
    CUDA-based detection postprocessor.

Functions
---------
get_detections
    Get detections from model outputs.
postprocess_detections
    Postprocess detections from model outputs.
postprocess_efficient_nms
    Postprocess detections from model outputs.
decode_efficient_nms
    Decode detections from model outputs.

"""

from __future__ import annotations

from ._abc import DetectionPostprocessor
from ._cpu import CPUDetectionPostprocessor
from ._cuda import CUDADetectionPostprocessor
from ._process import (
    get_detections,
    postprocess_detections,
    postprocess_efficient_nms,
    decode_efficient_nms,
)

__all__ = [
    "CPUDetectionPostprocessor",
    "CUDADetectionPostprocessor",
    "DetectionPostprocessor",
    "get_detections",
    "postprocess_detections",
    "postprocess_efficient_nms",
    "decode_efficient_nms",
]
