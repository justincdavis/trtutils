# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CUDA postprocessing for OCR heads: CTC argmax and DB binarization."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from trtutils.core._buffer import Buffer, MemoryLocation
from trtutils.core._memory import memcpy_device_to_host_async
from trtutils.image.kernels import OCR_POST

from ._cuda import CUDAPostprocessor

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Self

    from trtutils.compat._libs import cudart

_THREADS = 256


class CTCArgmax(CUDAPostprocessor):
    """Greedy CTC decode of a (B, T, C) float32 head: best class and its probability per step."""

    _FILE = OCR_POST

    def __init__(self: Self, stream: cudart.cudaStream_t) -> None:
        super().__init__(stream)
        self._argmax = self._kernel("ctcArgmax")
        # (indices, probs) per slot, so several batches can be decoded before one sync;
        # the input width changes per batch, so the buffers only grow
        self._slots: list[tuple[Buffer, Buffer]] = []

    def __call__(self: Self, outputs: list[Buffer]) -> list[list[np.ndarray]]:
        """Match ``postprocess_text_recognition``: [indices (T,) int64, probs (T,) float32] per image."""
        self.enqueue(outputs[0])
        self._sync()
        return self.read(outputs[0].shape)

    def enqueue(self: Self, probs: Buffer, slot: int = 0) -> None:
        """Enqueue the decode of ``probs`` into ``slot``, read it with :meth:`read` after a sync."""
        batch, steps, classes = probs.shape
        rows = batch * steps
        while len(self._slots) <= slot:
            self._slots.append(
                (Buffer.host_mapped((1,), np.int32), Buffer.host_mapped((1,), np.float32))
            )
        if rows > self._slots[slot][0].size:
            self._slots[slot] = (
                Buffer.host_mapped((rows,), np.int32),
                Buffer.host_mapped((rows,), np.float32),
            )
        idx, prob = self._slots[slot]
        self._launch(
            self._argmax,
            (rows, 1, 1),
            (_THREADS, 1, 1),
            probs.device_ptr,
            classes,
            idx.device_ptr,
            prob.device_ptr,
        )

    def read(self: Self, shape: tuple[int, ...], slot: int = 0) -> list[list[np.ndarray]]:
        """Return the decode of a ``shape`` (B, T, C) head in ``slot``, after a stream sync."""
        batch, steps = shape[0], shape[1]
        rows = batch * steps
        idx_buf, prob_buf = self._slots[slot]
        idx = idx_buf.array[:rows].astype(np.int64).reshape(batch, steps)
        prob = prob_buf.array[:rows].reshape(batch, steps).copy()
        return [[i, p] for i, p in zip(idx, prob)]


class DBBinarize(CUDAPostprocessor):
    """
    Threshold a (B, 1, H, W) float32 DB probability map on the device.

    The 0/1 bitmap is ready on return; the probability map itself is copied
    to the host in the background and ``wait`` blocks until it has landed.
    Each slot has its own host buffers, so a pipeline can binarize the next
    image while the previous one's maps are still read.
    """

    _FILE = OCR_POST

    def __init__(self: Self, stream: cudart.cudaStream_t) -> None:
        super().__init__(stream)
        self._binarize = self._kernel("dbBinarize")
        # slot -> (mapped bitmap, pinned probability map); the input shape changes per
        # image, so the buffers only grow
        self._slots: dict[int, tuple[Buffer, Buffer]] = {}

    def _enqueue(
        self: Self, probs: Buffer, thresh: float, slot: int
    ) -> tuple[np.ndarray, np.ndarray]:
        count = probs.size
        held = self._slots.get(slot)
        if held is None or held[0].size < count:
            pred = Buffer.empty((count,), np.float32, MemoryLocation.HOST, pinned=True)
            held = (Buffer.host_mapped((count,), np.uint8), pred)
            self._slots[slot] = held
        bitmap, pred = held
        self._launch(
            self._binarize,
            (math.ceil(count / _THREADS), 1, 1),
            (_THREADS, 1, 1),
            probs.device_ptr,
            count,
            float(thresh),
            bitmap.device_ptr,
        )
        shape = probs.shape
        return bitmap.array[:count].reshape(shape), pred.array[:count].reshape(shape)

    def enqueue(
        self: Self, probs: Buffer, thresh: float, slot: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Enqueue the bitmap and the probability map copy into ``slot``, read after a stream sync."""
        bitmaps, pred = self._enqueue(probs, thresh, slot)
        memcpy_device_to_host_async(pred, probs.device_ptr, self._stream)
        return bitmaps, pred

    def __call__(
        self: Self,
        outputs: list[Buffer],
        thresh: float,
    ) -> tuple[np.ndarray, np.ndarray, Callable[[], None]]:
        """Return the bitmap and probability map, both (B, 1, H, W), and the ``wait`` callable."""
        probs = outputs[0]
        bitmaps, pred = self._enqueue(probs, thresh, 0)
        # the bitmap is read first, the map copy overlaps finding its contours
        self._sync()
        memcpy_device_to_host_async(pred, probs.device_ptr, self._stream)
        return bitmaps, pred, self._sync
