# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Tests for src/trtutils/core/_buffer.py -- the Buffer type.

The ``cpu`` tests swap the CUDA runtime for an in-process fake whose
"device" memory is ordinary host memory, so allocation lifetimes, views,
copy dispatch, and the CUDA Array Interface are exercised without a GPU.
The remaining tests run against the real runtime.
"""

from __future__ import annotations

import ctypes
import gc
from types import SimpleNamespace

import numpy as np
import pytest

from trtutils.core import _bindings, _buffer
from trtutils.core._buffer import Buffer, MemoryLocation, as_buffers


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
    """Route every CUDA call made by the Buffer module through a host-memory fake."""
    fake = _FakeCudart()
    monkeypatch.setattr(_buffer, "cudart", fake, raising=False)
    monkeypatch.setattr(_buffer, "cuda_call", lambda result: result)
    return fake


def _collect() -> None:
    gc.collect()
    gc.collect()


@pytest.mark.cpu
class TestConstruction:
    """Buffer constructors."""

    def test_wrap_ndarray_is_zero_copy(self) -> None:
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        buf = Buffer.wrap(arr)
        assert buf.is_host
        assert not buf.pinned
        assert buf.shape == (3, 4)
        assert buf.dtype == np.float32
        assert buf.array is arr
        assert buf.ptr == arr.ctypes.data

    def test_wrap_buffer_returns_it(self) -> None:
        buf = Buffer.wrap(np.zeros(3))
        assert Buffer.wrap(buf) is buf

    def test_wrap_rejects_strided_array(self) -> None:
        arr = np.zeros((4, 4), dtype=np.float32)[:, ::2]
        with pytest.raises(ValueError, match="C-contiguous"):
            Buffer.wrap(arr)

    def test_wrap_rejects_other_types(self) -> None:
        with pytest.raises(TypeError, match="Cannot wrap"):
            Buffer.wrap([1, 2, 3])

    def test_wrap_readonly_array(self) -> None:
        arr = np.zeros(4, dtype=np.uint8)
        arr.flags.writeable = False
        buf = Buffer.wrap(arr)
        assert buf.readonly
        with pytest.raises(ValueError, match="read-only"):
            buf.copy_from(Buffer.wrap(np.ones(4, dtype=np.uint8)))

    def test_from_array_copies_any_strides(self, fake_cuda) -> None:
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)[:, ::2]
        host = Buffer.from_array(arr)
        assert host.pinned
        np.testing.assert_array_equal(host.array, arr)
        device = Buffer.from_array(arr, MemoryLocation.DEVICE)
        assert device.is_device
        np.testing.assert_array_equal(device.numpy(), arr)

    def test_empty_pageable(self) -> None:
        buf = Buffer.empty((2, 3), np.int32, pinned=False)
        assert buf.is_host
        assert not buf.pinned
        assert buf.array.shape == (2, 3)

    def test_empty_rejects_inferred_dim(self) -> None:
        with pytest.raises(ValueError, match="-1"):
            Buffer.empty((-1, 3), np.float32, pinned=False)

    def test_properties(self) -> None:
        buf = Buffer.wrap(np.zeros((2, 3, 4), dtype=np.float16))
        assert buf.ndim == 3
        assert buf.size == 24
        assert buf.nbytes == 48
        assert len(buf) == 2


@pytest.mark.cpu
class TestViews:
    """Views, reshapes, and leading-axis indexing."""

    def test_view_is_leading_prefix(self) -> None:
        arr = np.arange(24, dtype=np.float32)
        view = Buffer.wrap(arr).view((2, 3))
        assert view.shape == (2, 3)
        assert view.ptr == arr.ctypes.data
        np.testing.assert_array_equal(view.array, arr[:6].reshape(2, 3))

    def test_view_larger_than_buffer_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds"):
            Buffer.wrap(np.zeros(4)).view((5,))

    def test_reshape_infers_dim(self) -> None:
        buf = Buffer.wrap(np.zeros(12, dtype=np.float32)).reshape((-1, 4))
        assert buf.shape == (3, 4)

    def test_reshape_size_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot reshape"):
            Buffer.wrap(np.zeros(12)).reshape((5, 2))

    def test_getitem_index_and_slice(self) -> None:
        arr = np.arange(24, dtype=np.int32).reshape(4, 6)
        buf = Buffer.wrap(arr)
        row = buf[2]
        assert row.shape == (6,)
        assert row.ptr == arr[2].ctypes.data
        np.testing.assert_array_equal(row.array, arr[2])
        rows = buf[1:3]
        assert rows.shape == (2, 6)
        np.testing.assert_array_equal(rows.array, arr[1:3])
        np.testing.assert_array_equal(buf[-1].array, arr[-1])

    def test_getitem_errors(self) -> None:
        buf = Buffer.wrap(np.zeros((2, 2)))
        with pytest.raises(IndexError):
            buf[2]
        with pytest.raises(ValueError, match="step 1"):
            buf[::2]

    def test_device_view_offsets(self, fake_cuda) -> None:
        buf = Buffer.empty((4, 8), np.float32, MemoryLocation.DEVICE)
        assert buf[1].ptr == buf.ptr + 8 * 4
        assert buf[1:3].shape == (2, 8)


@pytest.mark.cpu
class TestLifetime:
    """Memory is only returned to CUDA once nothing references it."""

    def test_pinned_array_outlives_buffer(self, fake_cuda) -> None:
        """The numpy array keeps pinned memory alive: no use-after-free once the Buffer is gone."""
        buf = Buffer.empty((4,), np.float32)
        ptr = buf.ptr
        arr = buf.array
        arr[:] = 3.0
        buf.free()
        del buf
        _collect()
        assert fake_cuda.freed == []
        np.testing.assert_array_equal(arr, 3.0)
        del arr
        _collect()
        assert fake_cuda.freed == [("host", ptr)]

    def test_pinned_view_array_keeps_allocation(self, fake_cuda) -> None:
        buf = Buffer.empty((4, 4), np.float32)
        ptr = buf.ptr
        row = buf[1].array
        del buf
        _collect()
        assert fake_cuda.freed == []
        del row
        _collect()
        assert fake_cuda.freed == [("host", ptr)]

    def test_device_view_keeps_allocation(self, fake_cuda) -> None:
        buf = Buffer.empty((4, 4), np.float32, MemoryLocation.DEVICE)
        ptr = buf.ptr
        view = buf[1:]
        buf.free()
        _collect()
        assert fake_cuda.freed == []
        del view, buf
        _collect()
        assert fake_cuda.freed == [("device", ptr)]

    def test_freed_buffer_raises(self, fake_cuda) -> None:
        buf = Buffer.empty((4,), np.float32, MemoryLocation.DEVICE)
        buf.free()
        with pytest.raises(RuntimeError, match="freed"):
            _ = buf.ptr
        assert not hasattr(buf, "__cuda_array_interface__")

    def test_context_manager_frees(self, fake_cuda) -> None:
        with Buffer.empty((4,), np.float32, MemoryLocation.DEVICE) as buf:
            ptr = buf.ptr
        del buf
        _collect()
        assert fake_cuda.freed == [("device", ptr)]

    def test_from_ptr_keeps_owner(self, fake_cuda) -> None:
        owner = Buffer.empty((8,), np.uint8, MemoryLocation.DEVICE)
        ptr = owner.ptr
        view = Buffer.from_ptr(ptr, (8,), np.uint8, owner=owner)
        del owner
        _collect()
        assert fake_cuda.freed == []
        del view
        _collect()
        assert fake_cuda.freed == [("device", ptr)]


@pytest.mark.cpu
class TestCopies:
    """Copy direction dispatch and validation."""

    @pytest.mark.parametrize(
        ("src_loc", "dst_loc", "kind"),
        [
            pytest.param(MemoryLocation.HOST, MemoryLocation.DEVICE, "H2D", id="h2d"),
            pytest.param(MemoryLocation.DEVICE, MemoryLocation.HOST, "D2H", id="d2h"),
            pytest.param(MemoryLocation.DEVICE, MemoryLocation.DEVICE, "D2D", id="d2d"),
        ],
    )
    @pytest.mark.parametrize("stream", [None, 7], ids=["sync", "async"])
    def test_copy_dispatch(self, fake_cuda, src_loc, dst_loc, kind, stream) -> None:
        data = np.arange(6, dtype=np.float32)
        src = Buffer.from_array(data, src_loc)
        dst = Buffer.empty((2, 3), np.float32, dst_loc)
        fake_cuda.copies.clear()
        dst.copy_from(src, stream)
        assert fake_cuda.copies == [(kind, stream is not None)]
        np.testing.assert_array_equal(dst.numpy().reshape(-1), data)

    def test_host_to_host_uses_numpy(self, fake_cuda) -> None:
        src = Buffer.wrap(np.arange(4, dtype=np.int16))
        dst = Buffer.empty((2, 2), np.int16, pinned=False)
        src.copy_to(dst)
        assert fake_cuda.copies == []
        np.testing.assert_array_equal(dst.array, [[0, 1], [2, 3]])

    def test_copy_mismatch_raises(self) -> None:
        dst = Buffer.wrap(np.zeros(4, dtype=np.float32))
        with pytest.raises(ValueError, match="mismatch"):
            dst.copy_from(Buffer.wrap(np.zeros(4, dtype=np.float64)))
        with pytest.raises(ValueError, match="mismatch"):
            dst.copy_from(Buffer.wrap(np.zeros(5, dtype=np.float32)))

    def test_copy_from_ndarray_raises(self) -> None:
        dst = Buffer.wrap(np.zeros(4, dtype=np.float32))
        with pytest.raises(TypeError, match=r"Buffer\.wrap"):
            dst.copy_from(np.zeros(4, dtype=np.float32))  # ty: ignore[invalid-argument-type]


@pytest.mark.cpu
class TestInterop:
    """Numpy and CUDA Array Interface interop."""

    def test_array_protocol(self) -> None:
        arr = np.arange(4, dtype=np.float32)
        buf = Buffer.wrap(arr)
        assert np.asarray(buf) is arr
        copied = np.array(buf, copy=True)
        assert copied is not arr
        np.testing.assert_array_equal(np.asarray(buf, dtype=np.float64), arr)

    def test_array_protocol_rejects_device(self, fake_cuda) -> None:
        buf = Buffer.empty((4,), np.float32, MemoryLocation.DEVICE)
        with pytest.raises(TypeError, match="numpy"):
            np.asarray(buf)

    def test_cuda_array_interface(self, fake_cuda) -> None:
        buf = Buffer.empty((2, 3), np.float16, MemoryLocation.DEVICE)
        cai = buf.__cuda_array_interface__
        assert cai["shape"] == (2, 3)
        assert cai["typestr"] == np.dtype(np.float16).str
        assert cai["data"] == (buf.ptr, False)
        assert cai["version"] == 3

    def test_host_buffer_has_no_cuda_array_interface(self) -> None:
        assert not hasattr(Buffer.wrap(np.zeros(2)), "__cuda_array_interface__")

    def test_mapped_host_is_device_visible(self, fake_cuda) -> None:
        buf = Buffer.empty((4,), np.float32, pinned=True, mapped=True)
        assert buf.mapped
        assert buf.device_visible
        assert buf.device_ptr == buf.ptr
        assert buf[1].device_ptr == buf.device_ptr + 4
        assert hasattr(buf, "__cuda_array_interface__")

    def test_unmapped_host_has_no_device_ptr(self) -> None:
        with pytest.raises(ValueError, match="not mapped"):
            _ = Buffer.wrap(np.zeros(2)).device_ptr

    def test_wrap_cuda_array_syncs_producer_stream(self, fake_cuda) -> None:
        source = Buffer.empty((2, 2), np.float32, MemoryLocation.DEVICE)
        foreign = SimpleNamespace(
            __cuda_array_interface__={**source.__cuda_array_interface__, "stream": 42}
        )
        wrapped = Buffer.wrap(foreign)
        assert fake_cuda.synced == [42]
        assert wrapped.is_device
        assert wrapped.ptr == source.ptr
        assert wrapped.shape == (2, 2)

    def test_wrap_cuda_array_accepts_size_one_strides(self, fake_cuda) -> None:
        foreign = SimpleNamespace(
            __cuda_array_interface__={
                "shape": (1, 3),
                "typestr": "<f4",
                "data": (1024, True),
                "strides": (999, 4),
                "version": 3,
            }
        )
        wrapped = Buffer.wrap(foreign)
        assert wrapped.readonly

    def test_wrap_cuda_array_rejects_strided(self) -> None:
        foreign = SimpleNamespace(
            __cuda_array_interface__={
                "shape": (2, 3),
                "typestr": "<f4",
                "data": (1024, False),
                "strides": (24, 8),
                "version": 3,
            }
        )
        with pytest.raises(ValueError, match="C-contiguous"):
            Buffer.wrap(foreign)


@pytest.mark.cpu
class TestAsBuffers:
    """Engine input validation."""

    def test_accepts_list_and_tuple(self) -> None:
        buf = Buffer.wrap(np.zeros(2))
        assert as_buffers([buf]) == [buf]
        assert as_buffers((buf,)) == [buf]

    @pytest.mark.parametrize(
        "data",
        [
            pytest.param([np.zeros(2)], id="list-of-arrays"),
            pytest.param(np.zeros(2), id="bare-array"),
        ],
    )
    def test_rejects_non_buffers(self, data) -> None:
        with pytest.raises(TypeError, match="Buffer"):
            as_buffers(data)


@pytest.mark.cpu
class TestBindingOnFakeCuda:
    """Binding staging and fetching over Buffers."""

    @pytest.mark.parametrize("unified", [False, True], ids=["discrete", "unified"])
    def test_stage_and_fetch(self, fake_cuda, unified) -> None:
        binding = _bindings.create_binding(
            np.zeros((4, 3), dtype=np.float32), pagelocked_mem=True, unified_mem=unified
        )
        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        staged = binding.stage(Buffer.wrap(data))
        assert staged.ptr == binding.allocation
        assert staged.shape == (2, 3)
        np.testing.assert_array_equal(binding.fetch((2, 3)).array, data)
        # unified memory stages with a host memcpy and never copies back
        kinds = [kind for kind, _ in fake_cuda.copies]
        assert kinds == ([] if unified else ["H2D", "D2H"])

    def test_unified_binding_is_one_allocation(self, fake_cuda) -> None:
        binding = _bindings.create_binding(
            np.zeros(4, dtype=np.float32), pagelocked_mem=True, unified_mem=True
        )
        assert binding.unified_mem
        assert binding.allocation == binding.host.ptr
        binding.free()
        del binding
        _collect()
        assert [kind for kind, _ in fake_cuda.freed] == ["host"]
