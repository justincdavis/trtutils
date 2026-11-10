# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CUDA postprocessing for YOLO-style oriented boxes."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from trtutils.core._buffer import Buffer
from trtutils.image.kernels import YOLO_OBB

from ._cpu_obb import _probiou_reach
from ._cuda import CUDAPostprocessor, split_detections
from ._cuda_nms import YoloNms

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.compat._libs import cudart


class CUDAObbPostprocessor(CUDAPostprocessor):
    """YOLO oriented box postprocessing on the device, see ``postprocess_yolo_obb``."""

    _FILE = YOLO_OBB

    def __init__(self: Self, stream: cudart.cudaStream_t) -> None:
        super().__init__(stream)
        self._nms = YoloNms(stream)
        self._prep = self._kernel("obbPrep")
        self._suppress = self._kernel("obbSuppress")
        self._keep = self._kernel("obbKeep")

    def __call__(
        self: Self,
        outputs: list[Buffer],
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        conf_thres: float | None,
        nms_iou_thres: float,
        *,
        max_det: int,
    ) -> list[list[np.ndarray]]:
        """Postprocess device ``[output0]`` into the per-image lists of ``postprocess_yolo_obb``."""
        if not all(np.isclose(rx, ry) for rx, ry in ratios):
            err_msg = "OBB decoding needs uniform scaling, use resize_method='letterbox'"
            raise ValueError(err_msg)
        head = outputs[0]
        batch, channels, num = head.shape
        if self._resized(batch, num, max_det):
            self._boxes = Buffer.device((batch, num, 8), np.float32)
            self._alive = Buffer.device((batch, num), np.uint8)
            # (enclosing xyxy, score, class, cx, cy, w, h, angle) rows
            self._rows = Buffer.host_mapped((batch, max_det, 11), np.float32)
            self._counts = Buffer.host_mapped((batch,), np.int32)
        letterbox = self._letterbox(ratios, padding, None)
        nms = self._nms
        nms.rank(head, 4, channels - 5, conf_thres, max_det, inclusive=True)
        self._launch(
            self._prep,
            (math.ceil(num / 256), batch, 1),
            (256, 1, 1),
            head.device_ptr,
            channels,
            num,
            nms.count.ptr,
            nms.sort_anchor.ptr,
            letterbox,
            self._boxes.ptr,
        )
        # one warp per box
        self._launch(
            self._suppress,
            (math.ceil(num * 32 / 256), batch, 1),
            (256, 1, 1),
            num,
            nms.count.ptr,
            nms.sort_cls.ptr,
            self._boxes.ptr,
            float(nms_iou_thres),
            _probiou_reach(nms_iou_thres),
            self._alive.ptr,
        )
        self._launch(
            self._keep,
            (1, batch, 1),
            (1, 1, 1),
            num,
            max_det,
            nms.count.ptr,
            self._alive.ptr,
            nms.sort_cls.ptr,
            nms.sort_score.ptr,
            self._boxes.ptr,
            self._rows.device_ptr,
            self._counts.device_ptr,
        )
        self._sync()
        rows = self._rows.array
        return split_detections(rows[..., :6], self._counts.array, rows[..., 6:])
