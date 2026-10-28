# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
CUDA kernel implementations for various preprocessing functions.

Attributes
----------
:attribute:`SCALE_SWAP_TRANSPOSE` : tuple[Path, str]
    Rescales an image, swaps channels, and transposes HWC -> CHW
:attribute:`SST_FAST` : tuple[Path, str]
    Rescales an image, swaps channels, and transposes HWC -> CHW (float32 output)
:attribute:`SST_FAST_F16` : tuple[Path, str]
    Rescales an image, swaps channels, and transposes HWC -> CHW (float16 output)
:attribute:`LETTERBOX_RESIZE` : tuple[Path, str]
    Resizes an image using the letterbox method.
:attribute:`LINEAR_RESIZE` : tuple[Path, str]
    Resizes and image using bilinear interpolation.
:attribute:`IMAGENET_SST` : tuple[Path, str]
    ImageNet normalization with scale/swap/transpose (float32 output)
:attribute:`IMAGENET_SST_F16` : tuple[Path, str]
    ImageNet normalization with scale/swap/transpose (float16 output)
:attribute:`YOLO_POST` : tuple[Path, str]
    YOLO-style head postprocessing (decode, NMS, segmentation masks); the
    file holds several kernels, the name is the first.
:attribute:`OCR_POST` : tuple[Path, str]
    OCR postprocessing (CTC argmax, DB binarization); the file holds
    several kernels, the name is the first.

"""

from __future__ import annotations

from pathlib import Path

_KERNEL_DIR = Path(__file__).parent / "_kernels"
_SST_FILE = _KERNEL_DIR / "sst.cu"
_SST_FAST_FILE = _KERNEL_DIR / "sst_opt.cu"
_LETTERBOX_FILE = _KERNEL_DIR / "letterbox.cu"
_LINEAR_FILE = _KERNEL_DIR / "linear.cu"
_SST_IMAGENET_FILE = _KERNEL_DIR / "sst_imagenet.cu"
_YOLO_POST_FILE = _KERNEL_DIR / "yolo_post.cu"
_OCR_POST_FILE = _KERNEL_DIR / "ocr_post.cu"

SST_FAST: tuple[Path, str] = (
    _SST_FAST_FILE,
    "scaleSwapTranspose_opt",
)

SST_FAST_F16: tuple[Path, str] = (
    _SST_FAST_FILE,
    "scaleSwapTranspose_opt_f16",
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

IMAGENET_SST: tuple[Path, str] = (
    _SST_IMAGENET_FILE,
    "scaleSwapTransposeImagenet",
)

IMAGENET_SST_F16: tuple[Path, str] = (
    _SST_IMAGENET_FILE,
    "scaleSwapTransposeImagenet_f16",
)

YOLO_POST: tuple[Path, str] = (
    _YOLO_POST_FILE,
    "yoloDecode",
)

OCR_POST: tuple[Path, str] = (
    _OCR_POST_FILE,
    "ctcArgmax",
)
