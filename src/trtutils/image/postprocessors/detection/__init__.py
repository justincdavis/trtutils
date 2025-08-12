# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Detection postprocessing module.

Classes
-------
DetectionPostprocessor
    Abstract base class for detection postprocessors.
GPUDetectionPostprocessor
    GPU-based detection postprocessor.
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

"""

from __future__ import annotations

from ._abc import DetectionPostprocessor, GPUDetectionPostprocessor
from ._cpu import CPUDetectionPostprocessor
from ._cuda import CUDADetectionPostprocessor
from ._process import get_detections, postprocess_detections

__all__ = [
    "CPUDetectionPostprocessor",
    "CUDADetectionPostprocessor",
    "DetectionPostprocessor",
    "GPUDetectionPostprocessor",
    "get_detections",
    "postprocess_detections",
]
