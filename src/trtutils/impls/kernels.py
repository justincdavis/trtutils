# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
CUDA kernel implementations for various preprocessing functions.

Attributes
----------
:attribute:`SCALE_SWAP_TRANSPOSE` : tuple[str, str]
    Rescales an image, swaps channels, and transposes HWC -> CHW
:attribute:`LETTERBOX_RESIZE` : tuple[str, str]
    Resizes an image using the letterbox method.
:attribute:`LINEAR_RESIZE` : tuple[str, str]
    Resizes and image using bilinear interpolation.

"""

from __future__ import annotations

from pathlib import Path

_KERNEL_DIR = Path(__file__).parent / "_kernels"
_SST_FILE = _KERNEL_DIR / "sst.cu"
_LETTERBOX_FILE = _KERNEL_DIR / "letterbox.cu"
_LINEAR_FILE = _KERNEL_DIR / "linear.cu"

with _SST_FILE.open("r") as f:
    _SCALE_SWAP_TRANSPOSE_KERNEL_CODE = f.read()
SCALE_SWAP_TRANSPOSE: tuple[str, str] = (
    _SCALE_SWAP_TRANSPOSE_KERNEL_CODE,
    "scaleSwapTranspose",
)

with _LETTERBOX_FILE.open("r") as f:
    _LETTERBOX_RESIZE_KERNEL_CODE = f.read()
LETTERBOX_RESIZE: tuple[str, str] = (
    _LETTERBOX_RESIZE_KERNEL_CODE,
    "letterboxResize",
)

with _LINEAR_FILE.open("r") as f:
    _LINEAR_RESIZE_KERNEL_CODE = f.read()
LINEAR_RESIZE: tuple[str, str] = (
    _LINEAR_RESIZE_KERNEL_CODE,
    "linearResize",
)
