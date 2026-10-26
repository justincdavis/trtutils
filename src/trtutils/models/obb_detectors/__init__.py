# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Oriented bounding box (OBB) detector model implementations."""

from __future__ import annotations

from ._yolo import YOLOv8OBB, YOLOv11OBB, YOLOv26OBB

__all__ = [
    "YOLOv8OBB",
    "YOLOv11OBB",
    "YOLOv26OBB",
]
