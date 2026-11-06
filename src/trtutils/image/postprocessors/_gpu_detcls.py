# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CUDA postprocessing for detection unletterbox and classification softmax."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from trtutils.image.kernels import DET_CLS_POST

from ._gpu import CudaPostprocessor, index_kind, mapped, split_detections

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.compat._libs import cudart
    from trtutils.core._buffer import Buffer

_FLOAT = 4  # bytes


class CudaDetPostprocessor(CudaPostprocessor):
    """
    Unletterbox YOLO-v10 or EfficientNMS outputs on the device.

    Same boxes, scores and class ids as ``postprocess_yolov10`` and
    ``postprocess_efficient_nms``, kept in row order.
    """

    _FILE = DET_CLS_POST

    def __init__(self: Self, stream: cudart.cudaStream_t) -> None:
        super().__init__(stream)
        self._unletterbox = self._kernel("detUnletterbox")

    def __call__(
        self: Self,
        outputs: list[Buffer],
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        conf_thres: float | None,
        input_size: tuple[int, int],
    ) -> list[list[np.ndarray]]:
        """Postprocess YOLO-v10 ``[(batch, n, 6)]`` or EfficientNMS ``[num_dets, boxes, scores, classes]``."""
        if len(outputs) == 1:
            # (x1, y1, x2, y2, score, class) rows
            rows = outputs[0]
            batch, n, _ = rows.shape
            ptr = rows.device_ptr
            strided = (ptr, 6, ptr + 4 * _FLOAT, 6, ptr + 5 * _FLOAT, 6, index_kind(np.float32))
            limits = (0, 0)
        else:
            num_dets, boxes, scores, classes = outputs
            batch, n, _ = boxes.shape
            strided = (boxes.device_ptr, 4, scores.device_ptr, 1, classes.device_ptr, 1)
            strided = (*strided, index_kind(classes.dtype))
            limits = (num_dets.device_ptr, index_kind(num_dets.dtype))
        if self._resized(batch, n):
            self._dets = mapped((batch, n, 6), np.float32)
            self._counts = mapped((batch,), np.int32)
        self._launch(
            self._unletterbox,
            (batch, 1, 1),
            (1, 1, 1),
            *strided,
            *limits,
            n,
            self._letterbox(ratios, padding, input_size),
            -math.inf if conf_thres is None else float(conf_thres),
            self._dets.device_ptr,
            self._counts.device_ptr,
        )
        self._sync()
        return split_detections(self._dets.array, self._counts.array)


class CudaClsPostprocessor(CudaPostprocessor):
    """Softmax (or a copy) of a classification head, one block per image."""

    _FILE = DET_CLS_POST

    def __init__(self: Self, stream: cudart.cudaStream_t) -> None:
        super().__init__(stream)
        self._softmax = self._kernel("clsSoftmax")

    def __call__(self: Self, outputs: list[Buffer], *, softmax: bool) -> list[list[np.ndarray]]:
        """
        Match ``postprocess_classifications`` for one contiguous float32 head.

        The last axis is the class axis. Each image is returned as a ``(1, C)`` copy.
        """
        logits = outputs[0]
        classes = int(logits.shape[-1])
        batch = int(logits.size // classes)
        if self._resized(batch, classes):
            self._probs = mapped((batch, classes), np.float32)
        self._launch(
            self._softmax,
            (batch, 1, 1),
            (256, 1, 1),
            logits.device_ptr,
            classes,
            int(softmax),
            self._probs.device_ptr,
        )
        self._sync()
        probs = self._probs.array
        return [[probs[i].reshape(1, classes).copy()] for i in range(batch)]
