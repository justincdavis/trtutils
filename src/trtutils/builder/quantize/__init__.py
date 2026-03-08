# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule for ONNX post-training quantization using NVIDIA modelopt.

Functions
---------
:func:`generate_calibration_data`
    Generate calibration data from a batcher and save to a .npy file.
:func:`quantize_onnx`
    Quantize an ONNX model using NVIDIA modelopt.

"""

from __future__ import annotations

from ._calibration import generate_calibration_data
from ._quantize import quantize_onnx

__all__ = [
    "generate_calibration_data",
    "quantize_onnx",
]
