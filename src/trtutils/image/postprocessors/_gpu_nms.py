# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Shared CUDA NMS for YOLO-style heads."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from trtutils.compat._libs import cudart
from trtutils.core._buffer import Buffer, MemoryLocation
from trtutils.core._cuda import cuda_call
from trtutils.core._kernels import Kernel, cached_args
from trtutils.image.kernels import YOLO_NMS

if TYPE_CHECKING:
    from typing_extensions import Self

# column stripes per row block of the NMS mask kernel
_NMS_STRIPES = 16
_RANK_SLICES = 16  # RANK_SLICES in yolo_nms.cu
# NMS_MAX_WORDS in yolo_nms.cu
_MAX_CANDIDATES = 64 * 1024
# NMS bit matrix rows before a frame needs more: (N, N / 64) words is 141 MB at N = 33600
_NMS_ROWS = 4096


def _device(shape: tuple[int, ...], dtype: type) -> Buffer:
    return Buffer.empty(shape, dtype, MemoryLocation.DEVICE)


def _mapped(shape: tuple[int, ...], dtype: type) -> Buffer:
    # pinned host memory the kernels write directly, read after a stream sync
    return Buffer.empty(shape, dtype, MemoryLocation.HOST, pinned=True, mapped=True)


class YoloNms:
    """
    Decode, sort, and class-aware NMS of a (B, C, N) YOLO head on the device.

    Matches the CPU path: candidates scoring above ``conf_thres`` sorted by
    score (ties by anchor), then greedy NMS as cv2.dnn.NMSBoxesBatched.
    """

    def __init__(self: Self) -> None:
        file, _ = YOLO_NMS
        self._decode = Kernel(file, "yoloDecode")
        self._rank = Kernel(file, "yoloRank")
        self._scatter = Kernel(file, "yoloScatter")
        self._mask = Kernel(file, "yoloNmsMask")
        self._reduce = Kernel(file, "yoloNmsReduce")
        self._shape: tuple[int, int, int, int] | None = None

    def _allocate(self: Self, batch: int, num: int, max_det: int) -> None:
        self.count = _device((batch,), np.int32)
        self._cand_anchor = _device((batch, num), np.int32)
        self._cand_score = _device((batch, num), np.float32)
        self._cand_cls = _device((batch, num), np.int32)
        self._cand_rank = _device((batch, num), np.int32)
        self.sort_anchor = _device((batch, num), np.int32)
        self.sort_score = _device((batch, num), np.float32)
        self.sort_cls = _device((batch, num), np.int32)
        self.sort_box = _device((batch, num, 4), np.float32)
        self._bits: Buffer | None = None
        self._rows = min(num, _NMS_ROWS)
        self.keep = _device((batch, max_det), np.int32)
        self.keep_count = _device((batch,), np.int32)

    def rank(
        self: Self,
        head: Buffer,
        cls_start: int,
        nc: int,
        conf_thres: float | None,
        max_det: int,
        stream: cudart.cudaStream_t,
        *,
        inclusive: bool = False,
    ) -> None:
        """
        Enqueue the decode and sort of ``head`` on ``stream``, filling count and the sorted arrays.

        Keeps scores above ``conf_thres`` (at or above with ``inclusive``); None keeps every anchor.
        """
        batch, channels, num = head.shape
        if self._shape != (batch, channels, num, max_det):
            self._allocate(batch, num, max_det)
            self._shape = (batch, channels, num, max_det)
        unfiltered = -math.inf if inclusive else 0.0
        conf = unfiltered if conf_thres is None else float(conf_thres)
        cuda_call(cudart.cudaMemsetAsync(self.count.ptr, 0, self.count.nbytes, stream))
        args = cached_args(
            self._decode,
            head.device_ptr,
            channels,
            num,
            cls_start,
            nc,
            conf,
            int(inclusive),
            self.count.ptr,
            self._cand_anchor.ptr,
            self._cand_score.ptr,
            self._cand_cls.ptr,
            self._cand_rank.ptr,
        )
        self._decode.call((math.ceil(num / 256), batch, 1), (256, 1, 1), stream, args)
        args = cached_args(
            self._rank,
            num,
            self.count.ptr,
            self._cand_anchor.ptr,
            self._cand_score.ptr,
            self._cand_rank.ptr,
        )
        blocks = math.ceil(num / 256)
        self._rank.call((blocks * _RANK_SLICES, batch, 1), (256, 1, 1), stream, args)
        args = cached_args(
            self._scatter,
            head.device_ptr,
            channels,
            num,
            self.count.ptr,
            self._cand_anchor.ptr,
            self._cand_score.ptr,
            self._cand_cls.ptr,
            self._cand_rank.ptr,
            self.sort_anchor.ptr,
            self.sort_score.ptr,
            self.sort_cls.ptr,
            self.sort_box.ptr,
        )
        self._scatter.call((blocks, batch, 1), (256, 1, 1), stream, args)

    def __call__(
        self: Self,
        head: Buffer,
        cls_start: int,
        nc: int,
        conf_thres: float | None,
        nms_iou_thres: float,
        max_det: int,
        stream: cudart.cudaStream_t,
    ) -> None:
        """
        Enqueue NMS of ``head`` on ``stream``, filling keep / keep_count and the sorted arrays.

        keep_count is -1 for an image with more candidates than the NMS holds: call
        :meth:`grow` and run the frame again.
        """
        batch, _, num = head.shape
        if num > _MAX_CANDIDATES:
            err_msg = f"CUDA NMS supports up to {_MAX_CANDIDATES} anchors, got {num}"
            raise ValueError(err_msg)
        self.rank(head, cls_start, nc, conf_thres, max_det, stream)
        rows = self._rows
        words = math.ceil(rows / 64)
        if self._bits is None:
            self._bits = _device((batch, rows, words), np.uint64)
        args = cached_args(
            self._mask,
            rows,
            num,
            self.count.ptr,
            self.sort_cls.ptr,
            self.sort_box.ptr,
            float(nms_iou_thres),
            self._bits.ptr,
        )
        self._mask.call((words, _NMS_STRIPES, batch), (64, 1, 1), stream, args)
        args = cached_args(
            self._reduce,
            rows,
            max_det,
            self.count.ptr,
            self._bits.ptr,
            self.keep.ptr,
            self.keep_count.ptr,
        )
        self._reduce.call((batch, 1, 1), (32, 1, 1), stream, args)

    def grow(self: Self) -> None:
        """Size the NMS for every anchor, after a frame overflowed it."""
        self._rows = self.sort_anchor.shape[1]
        self._bits = None
