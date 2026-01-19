# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule containing hooks for building TensorRT engines.

Functions
---------
:func:`yolo_efficient_nms_hook`
    Hook for building YOLO models with EfficientNMS.

"""

from __future__ import annotations

from ._yolo import yolo_efficient_nms_hook

__all__ = ["yolo_efficient_nms_hook"]
