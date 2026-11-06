# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CUDA postprocessing for YOLO pose heads."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from trtutils.core._buffer import Buffer

from ._cuda import CUDAPostprocessor, split_detections
from ._cuda_nms import YoloNms

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.compat._libs import cudart


class CUDAPosePostprocessor(CUDAPostprocessor):
    """YOLO pose NMS and keypoint unletterbox on the device, see ``postprocess_yolo_pose``."""

    def __init__(self: Self, stream: cudart.cudaStream_t) -> None:
        super().__init__(stream)
        self._nms = YoloNms(stream)

    def __call__(
        self: Self,
        outputs: list[Buffer],
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        conf_thres: float | None,
        nms_iou_thres: float,
        *,
        max_det: int,
        input_size: tuple[int, int],
    ) -> list[list[np.ndarray]]:
        """Postprocess a device ``(batch, 5 + K*3, N)`` pose head into ``[bboxes, scores, class_ids, keypoints]``."""
        head = outputs[0]
        batch, channels, _ = head.shape
        nk = (channels - 5) // 3
        if self._resized(batch, nk, max_det):
            self._dets = Buffer.host_mapped((batch, max_det, 6), np.float32)
            self._kpts = Buffer.host_mapped((batch, max_det, nk * 3), np.float32)
            self._counts = Buffer.host_mapped((batch,), np.int32)
        self._nms(head, 4, 1, conf_thres, nms_iou_thres, max_det)
        self._nms.gather(
            head,
            self._letterbox(ratios, padding, input_size),
            self._dets.device_ptr,
            self._counts.device_ptr,
            extra_start=5,
            extras=(self._kpts.device_ptr, nk * 3),
            points=True,
        )
        self._sync()
        kpts = self._kpts.array.reshape(batch, max_det, nk, 3)
        return split_detections(self._dets.array, self._counts.array, kpts)
