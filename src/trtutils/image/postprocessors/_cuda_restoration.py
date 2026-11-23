# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CUDA postprocessing for image restoration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from trtutils.core._buffer import Buffer
from trtutils.core._memory import memcpy_device_to_host_async
from trtutils.image.kernels import RESTORE_POST

from ._cuda import CUDAPostprocessor

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.compat._libs import cudart

_BLOCK = (32, 8, 1)


class CUDARestorePostprocessor(CUDAPostprocessor):
    """Restored images to uint8 BGR on the GPU, copying back a quarter of the float output."""

    _FILE = RESTORE_POST

    def __init__(self: Self, stream: cudart.cudaStream_t) -> None:
        super().__init__(stream)
        self._to_bgr = self._kernel("restoreToBgr")

    def __call__(
        self: Self,
        outputs: list[Buffer],
        sizes: list[tuple[int, int]] | None = None,
        scale: int = 1,
        *,
        no_copy: bool | None = None,  # noqa: ARG002
    ) -> list[list[np.ndarray]]:
        """
        Match ``postprocess_restoration`` for one contiguous float32 (B, 3, H, W) output.

        Raises
        ------
        ValueError
            If ``sizes`` does not have one entry per image.

        """
        restored = outputs[0]
        batch, _, height, width = (int(dim) for dim in restored.shape)
        if sizes is not None and len(sizes) != batch:
            err_msg = f"Expected {batch} image sizes, got {len(sizes)}"
            raise ValueError(err_msg)
        if self._resized(batch, height, width):
            # a device buffer and a pinned copy: byte stores straight into mapped host
            # memory are 20x slower than the float32 copy they replace
            self._images = Buffer.device((batch, height * width * 3), np.uint8)
            self._host = Buffer.empty((batch, height * width * 3), np.uint8, pinned=True)
        crops = [(height, width)] * batch
        if sizes is not None:
            crops = [(min(h * scale, height), min(w * scale, width)) for h, w in sizes]
        image_bytes = 3 * height * width * np.dtype(np.float32).itemsize
        for i, (out_h, out_w) in enumerate(crops):
            self._launch(
                self._to_bgr,
                ((out_w + _BLOCK[0] - 1) // _BLOCK[0], (out_h + _BLOCK[1] - 1) // _BLOCK[1], 1),
                _BLOCK,
                restored.device_ptr + i * image_bytes,
                height,
                width,
                out_h,
                out_w,
                self._images.device_ptr + i * height * width * 3,
            )
        host = self._host.array
        for i, (out_h, out_w) in enumerate(crops):
            memcpy_device_to_host_async(
                host[i, : out_h * out_w * 3],
                self._images.device_ptr + i * height * width * 3,
                self._stream,
            )
        self._sync()
        # the pinned buffer is reused by the next call, so every image is a copy
        return [
            [host[i, : out_h * out_w * 3].reshape(out_h, out_w, 3).copy()]
            for i, (out_h, out_w) in enumerate(crops)
        ]
