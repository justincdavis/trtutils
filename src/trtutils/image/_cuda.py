# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Shared plumbing for classes that launch kernels from one file on one stream."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from trtutils.core._kernels import Kernel, cached_args
from trtutils.core._stream import stream_synchronize

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self

    from trtutils.compat._libs import cudart


class CUDAStage:
    """Kernels from one file, launched with memoized arguments on one stream."""

    _FILE: ClassVar[Path]

    def __init__(self: Self, stream: cudart.cudaStream_t) -> None:
        self._stream = stream
        self._key: tuple[int, ...] | None = None

    def _kernel(self: Self, name: str) -> Kernel:
        return Kernel(self._FILE, name)

    def _resized(self: Self, *key: int) -> bool:
        """Whether the buffers must be (re)allocated for ``key`` (batch, sizes), then remember it."""
        if key == self._key:
            return False
        self._key = key
        return True

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
