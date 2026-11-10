# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CUDA postprocessing for classification softmax."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from trtutils.core._buffer import Buffer
from trtutils.image.kernels import DET_CLS_POST

from ._cuda import CUDAPostprocessor

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.compat._libs import cudart


class CUDAClsPostprocessor(CUDAPostprocessor):
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
            self._probs = Buffer.host_mapped((batch, classes), np.float32)
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
