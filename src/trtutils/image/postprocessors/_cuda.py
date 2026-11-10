# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Shared pieces of the CUDA postprocessors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from trtutils.core._buffer import Buffer
from trtutils.image._cuda import CUDAStage

from ._common import original_size

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.compat._libs import cudart


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


class CUDAPostprocessor(CUDAStage):
    """
    Base of the CUDA postprocessors: kernels from one file, launched on one stream.

    A task's CUDA postprocessor is called like its CPU postprocess function, with
    the engine outputs as device Buffers, and returns the same per-image host arrays.
    """

    def __init__(self: Self, stream: cudart.cudaStream_t) -> None:
        super().__init__(stream)
        self._letter: Buffer | None = None

    def _letterbox(
        self: Self,
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        input_size: tuple[int, int] | None,
    ) -> int:
        """Fill the letterbox table for this batch and return its device address."""
        table = letterbox_table(ratios, padding, input_size)
        if self._letter is None or self._letter.shape != table.shape:
            self._letter = Buffer.host_mapped(table.shape, np.float32)
        self._letter.array[:] = table
        return self._letter.device_ptr
