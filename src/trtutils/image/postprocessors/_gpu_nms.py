# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Shared CUDA NMS for YOLO-style heads."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from trtutils.compat._libs import cudart
from trtutils.core._cuda import cuda_call
from trtutils.image.kernels import YOLO_NMS

from ._gpu import CudaPostprocessor, device

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.core._buffer import Buffer

# column stripes per row block of the NMS mask kernel
_NMS_STRIPES = 16
_RANK_SLICES = 16  # RANK_SLICES in yolo_nms.cu
# NMS bit matrix rows, NMS_MAX_ROWS in yolo_nms.cu: (N, N / 64) words would be 141 MB at
# N = 33600, so frames with more candidates take the reduce kernel's slower path
_NMS_ROWS = 4096


class YoloNms(CudaPostprocessor):
    """
    Decode, sort, class-aware NMS and gather of a (B, C, N) YOLO head on the device.

    Matches the CPU path: candidates scoring above ``conf_thres`` sorted by
    score (ties by anchor), then greedy NMS as cv2.dnn.NMSBoxesBatched.
    Task postprocessors run :meth:`__call__` then :meth:`gather`, or use
    :meth:`rank` and the sorted arrays for their own suppression.
    """

    _FILE = YOLO_NMS

    def __init__(self: Self, stream: cudart.cudaStream_t) -> None:
        super().__init__(stream)
        self._decode = self._kernel("yoloDecode")
        self._rank = self._kernel("yoloRank")
        self._scatter = self._kernel("yoloScatter")
        self._mask = self._kernel("yoloNmsMask")
        self._reduce = self._kernel("yoloNmsReduce")
        self._gather = self._kernel("yoloGather")

    def _allocate(self: Self, batch: int, num: int, max_det: int) -> None:
        self.count = device((batch,), np.int32)
        self._cand_anchor = device((batch, num), np.int32)
        self._cand_score = device((batch, num), np.float32)
        self._cand_cls = device((batch, num), np.int32)
        self._cand_rank = device((batch, num), np.int32)
        self.sort_anchor = device((batch, num), np.int32)
        self.sort_score = device((batch, num), np.float32)
        self.sort_cls = device((batch, num), np.int32)
        self.sort_box = device((batch, num, 4), np.float32)
        self._rows = min(num, _NMS_ROWS)
        # allocated by the first __call__, rank() alone (OBB) does not use it
        self._bits: Buffer | None = None
        self._keep = device((batch, max_det), np.int32)
        self.keep_count = device((batch,), np.int32)

    def rank(
        self: Self,
        head: Buffer,
        cls_start: int,
        nc: int,
        conf_thres: float | None,
        max_det: int,
        *,
        inclusive: bool = False,
    ) -> None:
        """
        Enqueue the decode and sort of ``head``, filling count and the sorted arrays.

        Keeps scores above ``conf_thres`` (at or above with ``inclusive``); None keeps every anchor.
        """
        batch, channels, num = head.shape
        if self._resized(batch, num, max_det):
            self._allocate(batch, num, max_det)
        unfiltered = -math.inf if inclusive else 0.0
        conf = unfiltered if conf_thres is None else float(conf_thres)
        cuda_call(cudart.cudaMemsetAsync(self.count.ptr, 0, self.count.nbytes, self._stream))
        blocks = math.ceil(num / 256)
        self._launch(
            self._decode,
            (blocks, batch, 1),
            (256, 1, 1),
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
        self._launch(
            self._rank,
            (blocks * _RANK_SLICES, batch, 1),
            (256, 1, 1),
            num,
            self.count.ptr,
            self._cand_anchor.ptr,
            self._cand_score.ptr,
            self._cand_rank.ptr,
        )
        self._launch(
            self._scatter,
            (blocks, batch, 1),
            (256, 1, 1),
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

    def __call__(
        self: Self,
        head: Buffer,
        cls_start: int,
        nc: int,
        conf_thres: float | None,
        nms_iou_thres: float,
        max_det: int,
    ) -> None:
        """
        Enqueue the rank and NMS of ``head``, selecting the boxes :meth:`gather` reads.

        ``keep_count`` holds each image's kept count on the device for later kernels.
        """
        batch, _, num = head.shape
        self.rank(head, cls_start, nc, conf_thres, max_det)
        rows = self._rows
        if self._bits is None:
            self._bits = device((batch, rows, math.ceil(rows / 64)), np.uint64)
        self._launch(
            self._mask,
            (math.ceil(rows / 64), _NMS_STRIPES, batch),
            (64, 1, 1),
            rows,
            num,
            self.count.ptr,
            self.sort_cls.ptr,
            self.sort_box.ptr,
            float(nms_iou_thres),
            self._bits.ptr,
        )
        self._launch(
            self._reduce,
            (batch, 1, 1),
            (32, 1, 1),
            rows,
            num,
            max_det,
            self.count.ptr,
            self.sort_cls.ptr,
            self.sort_box.ptr,
            float(nms_iou_thres),
            self._bits.ptr,
            self._keep.ptr,
            self.keep_count.ptr,
        )

    def gather(
        self: Self,
        head: Buffer,
        letterbox: int,
        dets: int,
        counts: int,
        *,
        extra_start: int = 0,
        extras: tuple[int, int] = (0, 0),
        points: bool = False,
    ) -> None:
        """
        Enqueue the gather of the kept boxes into ``dets`` and their count into ``counts``.

        ``dets`` gets (batch, max_det, 6) rows of unletterboxed (x1, y1, x2, y2, score, class)
        and ``counts`` (batch,) int32 counts. ``extras`` is the (address, channels) of a
        (batch, max_det, channels) array for the head channels from ``extra_start``; with
        ``points`` those are (x, y, visibility) triplets unletterboxed like the boxes.
        """
        batch, channels, num = head.shape
        max_det = self._keep.shape[1]
        extras_ptr, n_extra = extras
        self._launch(
            self._gather,
            (max_det, batch, 1),
            (32, 1, 1),
            head.device_ptr,
            channels,
            num,
            extra_start,
            n_extra,
            int(points),
            max_det,
            self._keep.ptr,
            self.keep_count.ptr,
            self.sort_anchor.ptr,
            self.sort_score.ptr,
            self.sort_cls.ptr,
            self.sort_box.ptr,
            letterbox,
            dets,
            extras_ptr,
            counts,
        )
