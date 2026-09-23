# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Core test fixtures -- cache dir patching, a host-memory CUDA fake, and GPU fixtures."""

from __future__ import annotations

import ctypes
import os
from types import SimpleNamespace

import pytest

from trtutils.core import _buffer, _memory, cache

_CPU_ONLY = os.environ.get("TRTUTILS_IGNORE_MISSING_CUDA", "0") == "1"


@pytest.fixture
def patched_cache_dir(tmp_path, monkeypatch):
    """Provide a temporary cache directory with get_cache_dir patched."""
    cache_dir = tmp_path / "_engine_cache"
    cache_dir.mkdir()
    monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)
    return cache_dir


class _FakeCudart:
    """A CUDA runtime stand-in backed by host memory, recording every call."""

    cudaHostAllocDefault = 0  # noqa: N815
    cudaHostAllocMapped = 2  # noqa: N815
    cudaMemcpyKind = SimpleNamespace(  # noqa: N815
        cudaMemcpyHostToDevice="H2D",
        cudaMemcpyDeviceToHost="D2H",
        cudaMemcpyDeviceToDevice="D2D",
    )

    def __init__(self) -> None:
        self.blocks: dict[int, ctypes.Array] = {}
        self.freed: list[tuple[str, int]] = []
        self.copies: list[tuple[str, bool]] = []
        self.synced: list[int] = []

    def _alloc(self, nbytes: int) -> int:
        # over-allocate so returned addresses are 256-byte aligned like CUDA's
        block = ctypes.create_string_buffer(nbytes + 256)
        ptr = (ctypes.addressof(block) + 255) & ~255
        self.blocks[ptr] = block
        return ptr

    def cudaMalloc(self, nbytes: int) -> int:  # noqa: N802
        return self._alloc(nbytes)

    def cudaHostAlloc(self, nbytes: int, _flags: int) -> int:  # noqa: N802
        return self._alloc(nbytes)

    def cudaHostGetDevicePointer(self, ptr: int, _flags: int) -> int:  # noqa: N802
        return ptr

    def cudaFree(self, ptr: int) -> None:  # noqa: N802
        self.freed.append(("device", ptr))
        self.blocks.pop(ptr)

    def cudaFreeHost(self, ptr: int) -> None:  # noqa: N802
        self.freed.append(("host", ptr))
        self.blocks.pop(ptr)

    def cudaMemcpy(self, dst: int, src: int, nbytes: int, kind: str) -> None:  # noqa: N802
        self.copies.append((kind, False))
        ctypes.memmove(dst, src, nbytes)

    def cudaMemcpyAsync(self, dst: int, src: int, nbytes: int, kind: str, _stream: object) -> None:  # noqa: N802
        self.copies.append((kind, True))
        ctypes.memmove(dst, src, nbytes)

    def cudaStream_t(self, handle: int) -> int:  # noqa: N802
        return handle

    def cudaStreamSynchronize(self, stream: int) -> None:  # noqa: N802
        self.synced.append(stream)


@pytest.fixture
def fake_cuda(monkeypatch) -> _FakeCudart:
    """Route the CUDA calls made by Buffer allocations and copies through a host-memory fake."""
    fake = _FakeCudart()
    for module in (_buffer, _memory):
        monkeypatch.setattr(module, "cudart", fake, raising=False)
        monkeypatch.setattr(module, "cuda_call", lambda result: result)
    return fake


if not _CPU_ONLY:
    from tests.core._gpu_fixtures import *  # noqa: F403
