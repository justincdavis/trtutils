# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule for ONNX post-training quantization using NVIDIA modelopt.

Functions
---------
:func:`quantize_onnx`
    Quantize an ONNX model using NVIDIA modelopt.

"""

from __future__ import annotations

from ._quantize import quantize_onnx

__all__ = [
    "quantize_onnx",
]
