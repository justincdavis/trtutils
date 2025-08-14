# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
CUDA kernel implementations for various preprocessing functions.

Attributes
----------
:attribute:`SCALE_SWAP_TRANSPOSE` : tuple[Path, str]
    Rescales an image, swaps channels, and transposes HWC -> CHW
:attribute:`SST_FAST` : tuple[Path, str]
    Rescales an image, swaps channels, and transposes HWC -> CHW
:attribute:`LETTERBOX_RESIZE` : tuple[Path, str]
    Resizes an image using the letterbox method.
:attribute:`LINEAR_RESIZE` : tuple[Path, str]
    Resizes and image using bilinear interpolation.
:attribute:`RESCALE_V10_DETECTIONS` : tuple[Path, str]
    Rescales bounding boxes for YOLOv10 style output.
:attribute:`RESCALE_EFF_NMS_DETECTIONS` : tuple[Path, str]
    Rescales bounding boxes for EfficientNMS style output.

"""

from __future__ import annotations

from pathlib import Path

_KERNEL_DIR = Path(__file__).parent / "_kernels"
_SST_FILE = _KERNEL_DIR / "sst.cu"
_SST_FAST_FILE = _KERNEL_DIR / "sst_opt.cu"
_LETTERBOX_FILE = _KERNEL_DIR / "letterbox.cu"
_LINEAR_FILE = _KERNEL_DIR / "linear.cu"
_RESCALE_V10_FILE = _KERNEL_DIR / "rescale_v10.cu"
_RESCALE_EFF_NMS_FILE = _KERNEL_DIR / "rescale_eff_nms.cu"


SST_FAST: tuple[Path, str] = (
    _SST_FAST_FILE,
    "scaleSwapTranspose_opt",
)

SCALE_SWAP_TRANSPOSE: tuple[Path, str] = (
    _SST_FILE,
    "scaleSwapTranspose",
)

LETTERBOX_RESIZE: tuple[Path, str] = (
    _LETTERBOX_FILE,
    "letterboxResize",
)

LINEAR_RESIZE: tuple[Path, str] = (
    _LINEAR_FILE,
    "linearResize",
)

RESCALE_V10_DETECTIONS: tuple[Path, str] = (
    _RESCALE_V10_FILE,
    "rescaleV10Detections",
)

RESCALE_EFF_NMS_DETECTIONS: tuple[Path, str] = (
    _RESCALE_EFF_NMS_FILE,
    "rescaleEffNMSDetections",
)
