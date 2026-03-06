# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_memory.py -- memory allocation and memcpy."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from trtutils.core import _memory
from trtutils.core._memory import (
    allocate_managed_memory,
    allocate_pinned_memory,
    allocate_to_device,
    cuda_free,
    cuda_host_free,
    cuda_malloc,
    free_device_ptrs,
    get_ptr_pair,
    memcpy_device_to_device,
    memcpy_device_to_device_async,
    memcpy_device_to_host,
    memcpy_device_to_host_async,
    memcpy_host_to_device,
    memcpy_host_to_device_async,
    memcpy_host_to_device_offset,
    memcpy_host_to_device_offset_async,
)
from trtutils.core._stream import stream_synchronize

# ---------------------------------------------------------------------------
# Parametrization helpers
# ---------------------------------------------------------------------------
DTYPES = [
    pytest.param(np.float32, id="float32"),
    pytest.param(np.float16, id="float16"),
    pytest.param(np.int32, id="int32"),
    pytest.param(np.int8, id="int8"),
]

SIZES = [
    pytest.param(1, id="1_element"),
    pytest.param(100, id="100_elements"),
    pytest.param(10000, id="10k_elements"),
]


# ---------------------------------------------------------------------------
# Allocation tests
# ---------------------------------------------------------------------------
class TestCudaMalloc:
    """Tests for cuda_malloc and cuda_free."""

    def test_returns_int(self) -> None:
        """cuda_malloc should return an int pointer."""
        ptr = cuda_malloc(1024)
        assert isinstance(ptr, int)
        assert ptr != 0
        cuda_free(ptr)

    @pytest.mark.parametrize(
        "nbytes",
        [
            pytest.param(1, id="1B"),
            pytest.param(1024, id="1KB"),
            pytest.param(1024 * 1024, id="1MB"),
            pytest.param(16 * 1024 * 1024, id="16MB"),
        ],
    )
    def test_different_sizes(self, nbytes) -> None:
        """cuda_malloc works for various sizes."""
        ptr = cuda_malloc(nbytes)
        assert isinstance(ptr, int)
        assert ptr != 0
        cuda_free(ptr)

    def test_cuda_free(self) -> None:
        """cuda_free on a valid pointer should not raise."""
        ptr = cuda_malloc(1024)
        cuda_free(ptr)  # Should not raise


class TestAllocatePinnedMemory:
    """Tests for allocate_pinned_memory."""

    def test_returns_ndarray(self) -> None:
        """allocate_pinned_memory returns a numpy array."""
        arr = allocate_pinned_memory(1024, np.dtype(np.float32))
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        cuda_host_free(arr)

    def test_size_matches(self) -> None:
        """Allocated array has the correct number of elements."""
        nbytes = 400  # 100 float32 elements
        arr = allocate_pinned_memory(nbytes, np.dtype(np.float32))
        assert arr.size == 100
        cuda_host_free(arr)

    def test_custom_shape(self) -> None:
        """allocate_pinned_memory with explicit shape."""
        nbytes = 2 * 3 * 4  # 2x3 float32 = 24 bytes
        arr = allocate_pinned_memory(nbytes, np.dtype(np.float32), shape=(2, 3))
        assert arr.shape == (2, 3)
        cuda_host_free(arr)

    def test_unified_memory(self) -> None:
        """allocate_pinned_memory with unified_mem=True."""
        arr = allocate_pinned_memory(1024, np.dtype(np.float32), unified_mem=True)
        assert isinstance(arr, np.ndarray)
        cuda_host_free(arr)


class TestCudaHostFree:
    """Tests for cuda_host_free."""

    def test_free_pinned_array(self) -> None:
        """cuda_host_free on a pinned ndarray should not raise."""
        arr = allocate_pinned_memory(1024, np.dtype(np.float32))
        cuda_host_free(arr)  # Should not raise

    def test_free_int_ptr(self) -> None:
        """cuda_host_free can also accept an int host pointer."""
        arr = allocate_pinned_memory(1024, np.dtype(np.float32))
        host_ptr = arr.ctypes.data
        cuda_host_free(host_ptr)


class TestGetPtrPair:
    """Tests for get_ptr_pair."""

    def test_returns_tuple(self) -> None:
        """get_ptr_pair returns (host_ptr, device_ptr) tuple."""
        arr = allocate_pinned_memory(1024, np.dtype(np.float32), unified_mem=True)
        host_ptr, device_ptr = get_ptr_pair(arr)
        assert isinstance(host_ptr, int)
        assert isinstance(device_ptr, int)
        assert host_ptr != 0
        assert device_ptr != 0
        cuda_host_free(arr)


class TestAllocateManagedMemory:
    """Tests for allocate_managed_memory."""

    @pytest.mark.parametrize(
        "stream_arg",
        [
            pytest.param("no_stream", id="without_stream"),
            pytest.param("none", id="explicit_none"),
        ],
    )
    def test_returns_int_ptr(self, stream_arg) -> None:
        """allocate_managed_memory returns an int pointer."""
        kwargs = {} if stream_arg == "no_stream" else {"stream": None}
        ptr = allocate_managed_memory(1024, **kwargs)
        assert isinstance(ptr, int)
        assert ptr != 0
        cuda_free(ptr)

    def test_with_stream(self, cuda_stream) -> None:
        """allocate_managed_memory with a stream attaches memory."""
        ptr = allocate_managed_memory(1024, stream=cuda_stream)
        assert isinstance(ptr, int)
        assert ptr != 0
        cuda_free(ptr)

    def test_calls_cuda_malloc_managed_with_global_flag(self) -> None:
        """Managed allocation uses explicit cudaMemAttachGlobal flag."""
        with patch.object(
            _memory.cudart,
            "cudaMallocManaged",
            return_value=(_memory.cudart.cudaError_t.cudaSuccess, 1234),
        ) as malloc_managed, patch.object(_memory.cudart, "cudaStreamAttachMemAsync") as attach_mem:
            ptr = _memory.allocate_managed_memory(1024)

        assert ptr == 1234
        malloc_managed.assert_called_once_with(1024, _memory.cudart.cudaMemAttachGlobal)
        attach_mem.assert_not_called()

    def test_calls_stream_attach_with_full_argument_set(self) -> None:
        """Stream attach path passes stream, ptr, length, and flags."""
        fake_stream = object()
        with patch.object(
            _memory.cudart,
            "cudaMallocManaged",
            return_value=(_memory.cudart.cudaError_t.cudaSuccess, 5678),
        ) as malloc_managed, patch.object(
            _memory.cudart,
            "cudaStreamAttachMemAsync",
            return_value=(_memory.cudart.cudaError_t.cudaSuccess,),
        ) as attach_mem:
            ptr = _memory.allocate_managed_memory(2048, stream=fake_stream)

        assert ptr == 5678
        malloc_managed.assert_called_once_with(2048, _memory.cudart.cudaMemAttachGlobal)
        attach_mem.assert_called_once_with(
            fake_stream,
            5678,
            0,
            _memory.cudart.cudaMemAttachGlobal,
        )


# ---------------------------------------------------------------------------
# Memcpy roundtrip tests (H2D -> D2H)
# ---------------------------------------------------------------------------
class TestMemcpyRoundtrip:
    """Tests for H2D and D2H synchronous memcpy roundtrips."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("size", SIZES)
    def test_h2d_d2h_sync(self, dtype, size) -> None:
        """Data survives a H2D->D2H roundtrip (sync)."""
        src = np.arange(size, dtype=dtype)
        nbytes = src.size * src.itemsize

        ptr = cuda_malloc(nbytes)
        memcpy_host_to_device(ptr, src)

        dst = np.zeros(size, dtype=dtype)
        memcpy_device_to_host(dst, ptr)

        np.testing.assert_array_equal(src, dst)
        cuda_free(ptr)

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("size", SIZES)
    def test_h2d_d2h_async(self, dtype, size, cuda_stream) -> None:
        """Data survives a H2D->D2H roundtrip (async + sync)."""
        src = np.arange(size, dtype=dtype)
        nbytes = src.size * src.itemsize

        ptr = cuda_malloc(nbytes)
        memcpy_host_to_device_async(ptr, src, cuda_stream)
        stream_synchronize(cuda_stream)

        dst = np.zeros(size, dtype=dtype)
        memcpy_device_to_host_async(dst, ptr, cuda_stream)
        stream_synchronize(cuda_stream)

        np.testing.assert_array_equal(src, dst)
        cuda_free(ptr)


class TestMemcpyD2D:
    """Tests for device-to-device memcpy."""

    @pytest.mark.parametrize(
        "use_async",
        [pytest.param(False, id="sync"), pytest.param(True, id="async")],
    )
    def test_d2d_copy(self, use_async, request) -> None:
        """D2D copy preserves data (sync and async)."""
        src = np.arange(100, dtype=np.float32)
        nbytes = src.nbytes

        ptr1 = cuda_malloc(nbytes)
        ptr2 = cuda_malloc(nbytes)

        memcpy_host_to_device(ptr1, src)

        if use_async:
            stream = request.getfixturevalue("cuda_stream")
            memcpy_device_to_device_async(ptr2, ptr1, nbytes, stream)
            stream_synchronize(stream)
        else:
            memcpy_device_to_device(ptr2, ptr1, nbytes)

        dst = np.zeros(100, dtype=np.float32)
        memcpy_device_to_host(dst, ptr2)

        np.testing.assert_array_equal(src, dst)
        cuda_free(ptr1)
        cuda_free(ptr2)


class TestMemcpyOffset:
    """Tests for H2D offset memcpy."""

    @pytest.mark.parametrize(
        "use_async",
        [pytest.param(False, id="sync"), pytest.param(True, id="async")],
    )
    def test_h2d_offset(self, use_async, request) -> None:
        """H2D with byte offset writes to correct location (sync and async)."""
        # Allocate space for 200 float32 values
        total = 200
        dtype = np.float32
        nbytes = total * np.dtype(dtype).itemsize

        ptr = cuda_malloc(nbytes)

        # Write first 100 elements
        part1 = np.arange(100, dtype=dtype)
        if use_async:
            stream = request.getfixturevalue("cuda_stream")
            memcpy_host_to_device_async(ptr, part1, stream)
            stream_synchronize(stream)
        else:
            memcpy_host_to_device(ptr, part1)

        # Write next 100 elements at offset
        part2 = np.arange(100, 200, dtype=dtype)
        offset = 100 * np.dtype(dtype).itemsize
        if use_async:
            memcpy_host_to_device_offset_async(ptr, part2, offset, stream)
            stream_synchronize(stream)
        else:
            memcpy_host_to_device_offset(ptr, part2, offset)

        # Read all 200 back
        result = np.zeros(total, dtype=dtype)
        memcpy_device_to_host(result, ptr)

        expected = np.arange(200, dtype=dtype)
        np.testing.assert_array_equal(result, expected)
        cuda_free(ptr)


# ---------------------------------------------------------------------------
# allocate_to_device / free_device_ptrs tests
# ---------------------------------------------------------------------------
class TestAllocateToDevice:
    """Tests for allocate_to_device."""

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_allocate_copies_to_device(self, dtype) -> None:
        """allocate_to_device copies arrays to GPU and returns pointers."""
        arr = np.arange(50, dtype=dtype)
        ptrs = allocate_to_device([arr])

        assert len(ptrs) == 1
        assert isinstance(ptrs[0], int)
        assert ptrs[0] != 0

        # Verify data by reading back
        result = np.zeros(50, dtype=dtype)
        memcpy_device_to_host(result, ptrs[0])
        np.testing.assert_array_equal(arr, result)

        free_device_ptrs(ptrs)

    def test_multiple_arrays(self) -> None:
        """allocate_to_device handles multiple arrays."""
        arrays = [
            np.zeros(10, dtype=np.float32),
            np.ones(20, dtype=np.float32),
            np.arange(30, dtype=np.float32),
        ]
        ptrs = allocate_to_device(arrays)
        assert len(ptrs) == 3
        for p in ptrs:
            assert isinstance(p, int)
            assert p != 0
        free_device_ptrs(ptrs)


class TestFreeDevicePtrs:
    """Tests for free_device_ptrs."""

    @pytest.mark.parametrize(
        "count",
        [pytest.param(1, id="single"), pytest.param(5, id="multiple")],
    )
    def test_free_ptrs(self, count) -> None:
        """Free one or more pointers."""
        ptrs = [cuda_malloc(1024) for _ in range(count)]
        free_device_ptrs(ptrs)  # Should not raise

    def test_free_empty_list(self) -> None:
        """Free with an empty list should not raise."""
        free_device_ptrs([])  # Should not raise
