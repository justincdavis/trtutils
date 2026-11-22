# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule containing image model hooks for building TensorRT engines.

Functions
---------
:func:`semantic_argmax_hook`
    Hook that replaces segmentation logits with an in-engine class map.
:func:`yolo_efficient_nms_hook`
    Hook for building YOLO models with EfficientNMS.

"""

from __future__ import annotations

from ._semantic import semantic_argmax_hook
from ._yolo import yolo_efficient_nms_hook

__all__ = ["semantic_argmax_hook", "yolo_efficient_nms_hook"]
