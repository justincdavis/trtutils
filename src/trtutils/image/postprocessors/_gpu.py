# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Shared pieces of the CUDA postprocessors."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from trtutils.core._buffer import Buffer, MemoryLocation
from trtutils.core._kernels import Kernel, cached_args
from trtutils.core._stream import stream_synchronize

from ._common import original_size

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self

    from trtutils.compat._libs import cudart

# loadIndex() in postprocess.cuh
_INDEX_KIND = {np.dtype(np.int32): 0, np.dtype(np.float32): 1, np.dtype(np.int64): 2}


def device(shape: tuple[int, ...], dtype: type) -> Buffer:
    """Device memory for intermediate results."""
    return Buffer.empty(shape, dtype, MemoryLocation.DEVICE)


def mapped(shape: tuple[int, ...], dtype: type) -> Buffer:
    """Pinned host memory the kernels write directly, read after a stream sync."""
    return Buffer.empty(shape, dtype, MemoryLocation.HOST, pinned=True, mapped=True)


def index_kind(dtype: np.dtype | type) -> int:
    """Return the loadIndex() kind of an index tensor (class ids, counts)."""
    kind = _INDEX_KIND.get(np.dtype(dtype))
    if kind is None:
        err_msg = f"CUDA postprocess index tensors must be int32, int64, or float32, got {dtype}"
        raise ValueError(err_msg)
    return kind


def letterbox_table(
    ratios: list[tuple[float, float]],
    padding: list[tuple[float, float]],
    input_size: tuple[int, int] | None,
) -> np.ndarray:
    """(batch, 6) rows of ratio, padding and the size boxes clip to, see Letterbox in postprocess.cuh."""
    rows = np.full((len(ratios), 6), np.inf, dtype=np.float32)
    for i, (ratio, pad) in enumerate(zip(ratios, padding)):
        rows[i, :4] = (*ratio, *pad)
        if input_size is not None:
            rows[i, 4:] = original_size(ratio, pad, input_size)
    return rows


def split_detections(
    dets: np.ndarray,
    counts: np.ndarray,
    *extras: np.ndarray,
) -> list[list[np.ndarray]]:
    """
    Per image ``[bboxes, scores, class_ids, *extras]`` copies of the first ``counts`` rows.

    ``dets`` holds (batch, max_det, 6) rows of (x1, y1, x2, y2, score, class); each extra is
    a (batch, max_det, ...) array of the same rows.
    """
    results = []
    for i, n in enumerate(counts.tolist()):
        rows = dets[i, :n]
        results.append(
            [
                rows[:, :4].copy(),
                rows[:, 4].copy(),
                rows[:, 5].astype(int),
                *(extra[i, :n].copy() for extra in extras),
            ]
        )
    return results


class CudaPostprocessor:
    """
    Base of the CUDA postprocessors: kernels from one file, launched on one stream.

    A task's CUDA postprocessor is called like its CPU postprocess function, with
    the engine outputs as device Buffers, and returns the same per-image host arrays.
    """

    _FILE: ClassVar[Path]

    def __init__(self: Self, stream: cudart.cudaStream_t) -> None:
        self._stream = stream
        self._letter: Buffer | None = None

    def _kernel(self: Self, name: str) -> Kernel:
        return Kernel(self._FILE, name)

    def _launch(
        self: Self,
        kernel: Kernel,
        blocks: tuple[int, int, int],
        threads: tuple[int, int, int],
        *args: float,
    ) -> None:
        kernel.call(blocks, threads, self._stream, cached_args(kernel, *args))

    def _sync(self: Self) -> None:
        stream_synchronize(self._stream)

    def _letterbox(
        self: Self,
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        input_size: tuple[int, int] | None,
    ) -> int:
        """Fill the letterbox table for this batch and return its device address."""
        table = letterbox_table(ratios, padding, input_size)
        if self._letter is None or self._letter.shape != table.shape:
            self._letter = mapped(table.shape, np.float32)
        self._letter.array[:] = table
        return self._letter.device_ptr
