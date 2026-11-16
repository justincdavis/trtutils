# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule containing hooks for building TensorRT engines.

Functions
---------
:func:`fp16_io_hook`
    Hook making every float32 network input and output float16.
:func:`int8_input_hook`
    Hook making every float network input int8 at scale 1.
:func:`int8_ranges_hook`
    Hook setting INT8 dynamic ranges from the input ranges, without calibration data.
:func:`yolo_efficient_nms_hook`
    Hook for building YOLO models with EfficientNMS.

"""

from __future__ import annotations

from ._io import fp16_io_hook, int8_input_hook
from ._ranges import int8_ranges_hook
from ._yolo import yolo_efficient_nms_hook

__all__ = ["fp16_io_hook", "int8_input_hook", "int8_ranges_hook", "yolo_efficient_nms_hook"]
