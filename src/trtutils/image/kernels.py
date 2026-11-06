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
:attribute:`DET_CLS_POST` : Path
    Postprocess kernels for detection unletterbox and classification softmax.
:attribute:`YOLO_NMS` : Path
    Postprocess kernels for YOLO-style heads: decode, sort, NMS, and gather.
:attribute:`YOLO_OBB` : Path
    Postprocess kernels for oriented boxes: decode, probiou NMS, and keep.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_KERNEL_DIR = Path(__file__).parent / "_kernels"
_SST_FILE = _KERNEL_DIR / "sst.cu"
_SST_FAST_FILE = _KERNEL_DIR / "sst_opt.cu"
_LETTERBOX_FILE = _KERNEL_DIR / "letterbox.cu"
_LINEAR_FILE = _KERNEL_DIR / "linear.cu"
_SST_IMAGENET_FILE = _KERNEL_DIR / "sst_imagenet.cu"

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

DET_CLS_POST: Path = _KERNEL_DIR / "det_cls_post.cu"
YOLO_NMS: Path = _KERNEL_DIR / "yolo_nms.cu"
YOLO_OBB: Path = _KERNEL_DIR / "yolo_obb.cu"

# loadIndex() in _kernels/postprocess.cuh
_INDEX_KIND = {np.dtype(np.int32): 0, np.dtype(np.float32): 1, np.dtype(np.int64): 2}


def index_kind(dtype: np.dtype | type) -> int:
    """Return the loadIndex() kind of an index tensor (class ids, counts)."""
    kind = _INDEX_KIND.get(np.dtype(dtype))
    if kind is None:
        err_msg = f"CUDA postprocess index tensors must be int32, int64, or float32, got {dtype}"
        raise ValueError(err_msg)
    return kind
