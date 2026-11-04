# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CUDA postprocessing for detection unletterbox and classification softmax."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from trtutils.core._buffer import Buffer, MemoryLocation
from trtutils.core._kernels import Kernel, cached_args
from trtutils.core._stream import stream_synchronize
from trtutils.image.kernels import DET_CLS_POST

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.compat._libs import cudart


def _mapped(shape: tuple[int, ...], dtype: type) -> Buffer:
    return Buffer.empty(shape, dtype, MemoryLocation.HOST, pinned=True, mapped=True)


def _letterbox(
    ratios: list[tuple[float, float]],
    padding: list[tuple[float, float]],
    input_size: tuple[int, int],
) -> np.ndarray:
    # width/height match clip_boxes: round((input - 2 * pad) / ratio)
    rows = np.empty((len(ratios), 6), np.float32)
    input_w, input_h = input_size
    for i, ((rx, ry), (px, py)) in enumerate(zip(ratios, padding)):
        width = max(round((input_w - 2 * px) / rx), 1)
        height = max(round((input_h - 2 * py) / ry), 1)
        rows[i] = (rx, ry, px, py, width, height)
    return rows


class CudaDetPostprocessor:
    """
    Unletterbox YOLO-v10 or EfficientNMS boxes on the engine stream.

    Same boxes, scores and class ids as the CPU functions, kept in row order.
    """

    def __init__(self: Self) -> None:
        file, _ = DET_CLS_POST
        self._v10 = Kernel(file, "unletterboxV10")
        self._eff = Kernel(file, "unletterboxEff")
        self._shape: tuple[str, int, int] | None = None

    def _allocate(self: Self, kind: str, batch: int, n: int) -> None:
        self._boxes = _mapped((batch, n, 4), np.float32)
        self._scores = _mapped((batch, n), np.float32)
        self._classes = _mapped((batch, n), np.int32)
        self._counts = _mapped((batch,), np.int32)
        self._letter = _mapped((batch, 6), np.float32)
        self._shape = (kind, batch, n)

    def _pack(self: Self, batch: int) -> list[list[np.ndarray]]:
        boxes = self._boxes.array
        scores = self._scores.array
        classes = self._classes.array
        counts = self._counts.array
        results: list[list[np.ndarray]] = []
        for i in range(batch):
            n = int(counts[i])
            results.append([boxes[i, :n].copy(), scores[i, :n].copy(), classes[i, :n].copy()])
        return results

    def yolov10(
        self: Self,
        output: Buffer,
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        conf_thres: float | None,
        input_size: tuple[int, int],
        stream: cudart.cudaStream_t,
    ) -> list[list[np.ndarray]]:
        """Unletterbox a device ``(batch, N, 6)`` YOLOv10 output."""
        batch, n, _six = output.shape
        if self._shape != ("v10", batch, n):
            self._allocate("v10", batch, n)
        self._letter.array[:] = _letterbox(ratios, padding, input_size)
        conf = -1.0e30 if conf_thres is None else float(conf_thres)
        args = cached_args(
            self._v10,
            output.device_ptr,
            int(n),
            self._letter.device_ptr,
            conf,
            self._boxes.device_ptr,
            self._scores.device_ptr,
            self._classes.device_ptr,
            self._counts.device_ptr,
        )
        self._v10.call((batch, 1, 1), (1, 1, 1), stream, args)
        stream_synchronize(stream)
        return self._pack(batch)

    def efficient_nms(
        self: Self,
        outputs: list[Buffer],
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        conf_thres: float | None,
        input_size: tuple[int, int],
        stream: cudart.cudaStream_t,
    ) -> list[list[np.ndarray]]:
        """Unletterbox device EfficientNMS ``[num_dets, boxes, scores, classes]``."""
        num_dets, boxes, scores, classes = outputs
        batch, n, _four = boxes.shape
        if self._shape != ("eff", batch, n):
            self._allocate("eff", batch, n)
        self._letter.array[:] = _letterbox(ratios, padding, input_size)
        conf = -1.0e30 if conf_thres is None else float(conf_thres)
        num_is_64 = int(num_dets.dtype == np.int64)
        cls_is_float = int(classes.dtype == np.float32)
        args = cached_args(
            self._eff,
            num_dets.device_ptr,
            num_is_64,
            boxes.device_ptr,
            scores.device_ptr,
            classes.device_ptr,
            cls_is_float,
            int(n),
            self._letter.device_ptr,
            conf,
            self._boxes.device_ptr,
            self._scores.device_ptr,
            self._classes.device_ptr,
            self._counts.device_ptr,
        )
        self._eff.call((batch, 1, 1), (1, 1, 1), stream, args)
        stream_synchronize(stream)
        return self._pack(batch)


class CudaClsPostprocessor:
    """Softmax (or a copy) of a classification head, one block per image."""

    def __init__(self: Self) -> None:
        file, _ = DET_CLS_POST
        self._kernel = Kernel(file, "clsSoftmax")
        self._shape: tuple[int, int] | None = None

    def __call__(
        self: Self,
        output: Buffer,
        *,
        softmax: bool,
        stream: cudart.cudaStream_t,
    ) -> list[list[np.ndarray]]:
        """
        Match ``postprocess_classifications`` for one contiguous float32 head.

        The last axis is the class axis. Each image is returned as a ``(1, C)`` copy.
        """
        classes = int(output.shape[-1])
        batch = int(output.size // classes)
        if self._shape != (batch, classes):
            self._probs = _mapped((batch, classes), np.float32)
            self._shape = (batch, classes)
        args = cached_args(
            self._kernel,
            output.device_ptr,
            classes,
            int(softmax),
            self._probs.device_ptr,
        )
        threads = 256
        self._kernel.call((batch, 1, 1), (threads, 1, 1), stream, args)
        stream_synchronize(stream)
        probs = self._probs.array
        return [[probs[i].reshape(1, classes).copy()] for i in range(batch)]
