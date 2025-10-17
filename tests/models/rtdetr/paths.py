# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""RT-DETR model path definitions - imports from consolidated paths."""
from __future__ import annotations

from ..paths import (
    GROUND_TRUTHS,
    HORSE_IMAGE_PATH,
    IMAGE_PATHS,
    PEOPLE_IMAGE_PATH,
    RTDETR_ENGINE_PATHS as ENGINE_PATHS,
    RTDETR_ONNX_PATHS as ONNX_PATHS,
)

__all__ = [
    "ENGINE_PATHS",
    "ONNX_PATHS",
    "HORSE_IMAGE_PATH",
    "PEOPLE_IMAGE_PATH",
    "IMAGE_PATHS",
    "GROUND_TRUTHS",
]
