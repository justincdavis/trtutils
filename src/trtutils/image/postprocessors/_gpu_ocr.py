# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CUDA postprocessing for OCR heads, run on the engine stream."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from trtutils.compat._libs import cudart
from trtutils.core._buffer import Buffer, MemoryLocation
from trtutils.core._cuda import cuda_call
from trtutils.core._kernels import Kernel
from trtutils.core._stream import stream_synchronize
from trtutils.image.kernels import OCR_POST

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Self


def _mapped(count: int, dtype: type) -> Buffer:
    # pinned host memory the kernels write directly, read after a stream sync
    return Buffer.empty((count,), dtype, MemoryLocation.HOST, pinned=True, mapped=True)


class CtcArgmax:
    """Greedy CTC decode of a (B, T, C) float32 head: best class and its probability per step."""

    def __init__(self: Self) -> None:
        file, _ = OCR_POST
        self._kernel = Kernel(file, "ctcArgmax")
        self._capacity = 0

    def __call__(
        self: Self,
        probs: Buffer,
        stream: cudart.cudaStream_t,
    ) -> list[list[np.ndarray]]:
        """Return [indices (T,) int64, probs (T,) float32] per image, as postprocess_text_recognition."""
        batch, steps, classes = probs.shape
        rows = batch * steps
        if rows > self._capacity:
            self._idx = _mapped(rows, np.int32)
            self._prob = _mapped(rows, np.float32)
            self._capacity = rows
        args = self._kernel.create_args(
            probs.device_ptr, classes, self._idx.device_ptr, self._prob.device_ptr
        )
        self._kernel.call((rows, 1, 1), (256, 1, 1), stream, args)
        stream_synchronize(stream)
        idx = self._idx.array[:rows].astype(np.int64).reshape(batch, steps)
        prob = self._prob.array[:rows].reshape(batch, steps).copy()
        return [[i, p] for i, p in zip(idx, prob)]


class DbBinarize:
    """
    Threshold a (B, 1, H, W) float32 DB probability map on the device.

    The 0/1 bitmap is ready on return; the probability map itself is copied
    to the host in the background and ``wait`` blocks until it has landed.
    """

    def __init__(self: Self) -> None:
        file, _ = OCR_POST
        self._kernel = Kernel(file, "dbBinarize")
        self._capacity = 0

    def __call__(
        self: Self,
        probs: Buffer,
        thresh: float,
        stream: cudart.cudaStream_t,
    ) -> tuple[np.ndarray, np.ndarray, Callable[[], None]]:
        """Return the bitmap and probability map, both (B, 1, H, W), and the ``wait`` callable."""
        count = math.prod(probs.shape)
        if count > self._capacity:
            self._bitmap = _mapped(count, np.uint8)
            self._pred = Buffer.empty((count,), np.float32, MemoryLocation.HOST, pinned=True)
            self._capacity = count
        args = self._kernel.create_args(
            probs.device_ptr, count, float(thresh), self._bitmap.device_ptr
        )
        self._kernel.call((math.ceil(count / 256), 1, 1), (256, 1, 1), stream, args)
        stream_synchronize(stream)
        cuda_call(
            cudart.cudaMemcpyAsync(
                self._pred.ptr,
                probs.device_ptr,
                count * 4,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            ),
        )
        shape = probs.shape
        return (
            self._bitmap.array[:count].reshape(shape),
            self._pred.array[:count].reshape(shape),
            lambda: stream_synchronize(stream),
        )
