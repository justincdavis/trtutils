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


def _namespace(attrs: dict[str, object]) -> SimpleNamespace:
    # cuda-python's names are camelCase, so build the namespace from a dict
    return SimpleNamespace(**attrs)


class _FakeCuda:
    """A CUDA runtime stand-in backed by host memory, recording every call."""

    def __init__(self) -> None:
        self.blocks: dict[int, ctypes.Array] = {}
        self.freed: list[tuple[str, int]] = []
        self.copies: list[tuple[str, bool]] = []
        self.synced: list[int] = []
        self.cudart = _namespace(
            {
                "cudaHostAllocDefault": 0,
                "cudaHostAllocMapped": 2,
                "cudaMemcpyKind": _namespace(
                    {
                        "cudaMemcpyHostToDevice": "H2D",
                        "cudaMemcpyDeviceToHost": "D2H",
                        "cudaMemcpyDeviceToDevice": "D2D",
                    }
                ),
                "cudaMalloc": self._alloc,
                "cudaHostAlloc": lambda nbytes, _flags: self._alloc(nbytes),
                "cudaHostGetDevicePointer": lambda ptr, _flags: ptr,
                "cudaFree": lambda ptr: self._free("device", ptr),
                "cudaFreeHost": lambda ptr: self._free("host", ptr),
                "cudaMemcpy": lambda dst, src, nbytes, kind: self._copy(
                    dst, src, nbytes, kind, is_async=False
                ),
                "cudaMemcpyAsync": lambda dst, src, nbytes, kind, _stream: self._copy(
                    dst, src, nbytes, kind, is_async=True
                ),
                "cudaStream_t": lambda handle: handle,
                "cudaStreamSynchronize": self.synced.append,
            }
        )

    def _alloc(self, nbytes: int) -> int:
        # over-allocate so returned addresses are 256-byte aligned like CUDA's
        block = ctypes.create_string_buffer(nbytes + 256)
        ptr = (ctypes.addressof(block) + 255) & ~255
        self.blocks[ptr] = block
        return ptr

    def _free(self, kind: str, ptr: int) -> None:
        self.freed.append((kind, ptr))
        self.blocks.pop(ptr)

    def _copy(self, dst: int, src: int, nbytes: int, kind: str, *, is_async: bool) -> None:
        self.copies.append((kind, is_async))
        ctypes.memmove(dst, src, nbytes)


@pytest.fixture
def fake_cuda(monkeypatch) -> _FakeCuda:
    """Route the CUDA calls made by Buffer allocations and copies through a host-memory fake."""
    fake = _FakeCuda()
    for module in (_buffer, _memory):
        monkeypatch.setattr(module, "cudart", fake.cudart, raising=False)
        monkeypatch.setattr(module, "cuda_call", lambda result: result)
    return fake


if not _CPU_ONLY:
    from tests.core._gpu_fixtures import *  # noqa: F403
