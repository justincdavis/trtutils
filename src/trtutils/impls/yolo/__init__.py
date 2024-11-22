# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Implementations of YOLO object detectors for TRTModel.

Classes
-------
:class:`CPUPreprocessor`
    Preprocess an image for YOLO on CPU.
:class:`CUDAPreprocessor`
    Preprocess an image for YOLO using CUDA.
:class:`ParallelYOLO`
    Multi-threaded YOLO models, useful for multi-accelerator systems.
:class:`YOLO`
    TRTModel implementation for YOLO object detectors
:class:`YOLO7`
    Alias for YOLO with args preset
:class:`YOLO8`
    Alias for YOLO with args preset
:class:`YOLO9`
    Alias for YOLO with args preset
:class:`YOLO10`
    Alias for YOLO with args preset
:class:`YOLOX`
    Alias for YOLO with args preset

Functions
---------
:func:`get_detections`
    Get the detections from a YOLO network.
:func:`preprocess`
    Preprocess some input for a YOLO network.
:func:`postprocess`
    Postprocess the output for a YOLO network.

"""

from __future__ import annotations

from ._parallel import ParallelYOLO
from ._preprocessors import CPUPreprocessor, CUDAPreprocessor
from ._process import get_detections, postprocess, preprocess
from ._yolo import YOLO
from ._yolos import YOLO7, YOLO8, YOLO9, YOLO10, YOLOX

__all__ = [
    "YOLO",
    "YOLO7",
    "YOLO8",
    "YOLO9",
    "YOLO10",
    "YOLOX",
    "CPUPreprocessor",
    "CUDAPreprocessor",
    "ParallelYOLO",
    "get_detections",
    "postprocess",
    "preprocess",
]
