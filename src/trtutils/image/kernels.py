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
:attribute:`COMPACT_BOXES` : tuple[Path, str]
    Batched detection compaction (conf/finite filter, letterbox remap, compact)
:attribute:`COMPACT_V10` : tuple[Path, str]
    Batched YOLO-v10 (B, N, 6) detection compaction
:attribute:`COMPACT_RTDETRV3` : tuple[Path, str]
    Batched RT-DETR v3 (T, 6) + per-image counts detection compaction
:attribute:`RFDETR_SIGMOID` : tuple[Path, str]
    Sigmoid over RF-DETR logits
:attribute:`RFDETR_TOPK` : tuple[Path, str]
    Top-Q selection over RF-DETR sigmoid scores (iterative block argmax)
:attribute:`RFDETR_GATHER` : tuple[Path, str]
    RF-DETR selected box denormalization, remap, and compaction
:attribute:`SOFTMAX_ROWS` : tuple[Path, str]
    Per-row softmax for classification logits
:attribute:`DEPTH_MINMAX` : tuple[Path, str]
    Per-image min/max reduction for depth maps
:attribute:`DEPTH_NORMALIZE` : tuple[Path, str]
    In-place min-max normalization of depth maps to [0, 1]
:attribute:`HAND_NMS` : tuple[Path, str]
    Class-aware NMS + pair-probability gather for hand interactions

"""

from __future__ import annotations

from pathlib import Path

_KERNEL_DIR = Path(__file__).parent / "_kernels"
_SST_FILE = _KERNEL_DIR / "sst.cu"
_SST_FAST_FILE = _KERNEL_DIR / "sst_opt.cu"
_LETTERBOX_FILE = _KERNEL_DIR / "letterbox.cu"
_LINEAR_FILE = _KERNEL_DIR / "linear.cu"
_SST_IMAGENET_FILE = _KERNEL_DIR / "sst_imagenet.cu"
_PP_COMPACT_FILE = _KERNEL_DIR / "pp_compact.cu"
_PP_COMPACT_V10_FILE = _KERNEL_DIR / "pp_compact_v10.cu"
_PP_COMPACT_RTDETRV3_FILE = _KERNEL_DIR / "pp_compact_rtdetrv3.cu"
_PP_RFDETR_FILE = _KERNEL_DIR / "pp_rfdetr.cu"
_PP_SOFTMAX_FILE = _KERNEL_DIR / "pp_softmax.cu"
_PP_DEPTH_FILE = _KERNEL_DIR / "pp_depth.cu"
_PP_HAND_NMS_FILE = _KERNEL_DIR / "pp_hand_nms.cu"

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

COMPACT_BOXES: tuple[Path, str] = (_PP_COMPACT_FILE, "compactBoxes")
COMPACT_V10: tuple[Path, str] = (_PP_COMPACT_V10_FILE, "compactV10")
COMPACT_RTDETRV3: tuple[Path, str] = (_PP_COMPACT_RTDETRV3_FILE, "compactRtdetrV3")
RFDETR_SIGMOID: tuple[Path, str] = (_PP_RFDETR_FILE, "rfdetrSigmoid")
RFDETR_TOPK: tuple[Path, str] = (_PP_RFDETR_FILE, "rfdetrTopk")
RFDETR_GATHER: tuple[Path, str] = (_PP_RFDETR_FILE, "rfdetrGather")
SOFTMAX_ROWS: tuple[Path, str] = (_PP_SOFTMAX_FILE, "softmaxRows")
DEPTH_MINMAX: tuple[Path, str] = (_PP_DEPTH_FILE, "depthMinMax")
DEPTH_NORMALIZE: tuple[Path, str] = (_PP_DEPTH_FILE, "depthNormalize")
HAND_NMS: tuple[Path, str] = (_PP_HAND_NMS_FILE, "handNms")
