# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_memory.py -- memory allocation and memcpy."""

from __future__ import annotations

import numpy as np
import pytest

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


@pytest.mark.parametrize(
    "nbytes",
    [
        pytest.param(1, id="1B"),
        pytest.param(1024, id="1KB"),
        pytest.param(1024 * 1024, id="1MB"),
        pytest.param(16 * 1024 * 1024, id="16MB"),
    ],
)
def test_cuda_malloc(nbytes: int) -> None:
    """cuda_malloc returns a non-zero int pointer for various sizes."""
    ptr = cuda_malloc(nbytes)
    assert isinstance(ptr, int)
    assert ptr != 0
    cuda_free(ptr)


def test_allocate_pinned_memory() -> None:
    """allocate_pinned_memory returns correct ndarray with shape, dtype, and unified support."""
    # basic: returns ndarray with correct dtype
    arr = allocate_pinned_memory(1024, np.dtype(np.float32))
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    cuda_host_free(arr)
    # size: element count matches
    arr = allocate_pinned_memory(400, np.dtype(np.float32))
    assert arr.size == 100
    cuda_host_free(arr)
    # custom shape
    arr = allocate_pinned_memory(2 * 3 * 4, np.dtype(np.float32), shape=(2, 3))
    assert arr.shape == (2, 3)
    cuda_host_free(arr)
    # unified memory
    arr = allocate_pinned_memory(1024, np.dtype(np.float32), unified_mem=True)
    assert isinstance(arr, np.ndarray)
    cuda_host_free(arr)


def test_cuda_host_free() -> None:
    """cuda_host_free accepts both ndarray and int pointer."""
    # ndarray
    arr = allocate_pinned_memory(1024, np.dtype(np.float32))
    cuda_host_free(arr)
    # int pointer
    arr = allocate_pinned_memory(1024, np.dtype(np.float32))
    host_ptr = arr.ctypes.data
    cuda_host_free(host_ptr)


def test_get_ptr_pair() -> None:
    """get_ptr_pair returns (host_ptr, device_ptr) non-zero int tuple."""
    arr = allocate_pinned_memory(1024, np.dtype(np.float32), unified_mem=True)
    host_ptr, device_ptr = get_ptr_pair(arr)
    assert isinstance(host_ptr, int)
    assert isinstance(device_ptr, int)
    assert host_ptr != 0
    assert device_ptr != 0
    cuda_host_free(arr)


@pytest.mark.parametrize(
    "stream_source",
    [
        pytest.param("none_default", id="no-stream"),
        pytest.param("explicit_none", id="explicit-none"),
        pytest.param("real_stream", id="real-stream"),
    ],
)
def test_allocate_managed_memory(stream_source, request) -> None:
    """allocate_managed_memory returns non-zero pointer with and without stream."""
    if stream_source == "none_default":
        ptr = allocate_managed_memory(1024)
    elif stream_source == "explicit_none":
        ptr = allocate_managed_memory(1024, stream=None)
    else:
        stream = request.getfixturevalue("cuda_stream")
        ptr = allocate_managed_memory(1024, stream=stream)
    assert isinstance(ptr, int)
    assert ptr != 0
    cuda_free(ptr)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("size", SIZES)
def test_memcpy_h2d_d2h_sync(dtype, size) -> None:
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
def test_memcpy_h2d_d2h_async(dtype, size, cuda_stream) -> None:
    """Data survives a H2D->D2H roundtrip (async)."""
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


@pytest.mark.parametrize(
    "use_async",
    [pytest.param(False, id="sync"), pytest.param(True, id="async")],
)
def test_memcpy_d2d(use_async, request) -> None:
    """D2D copy preserves data."""
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


@pytest.mark.parametrize(
    "use_async",
    [pytest.param(False, id="sync"), pytest.param(True, id="async")],
)
def test_memcpy_h2d_offset(use_async, request) -> None:
    """H2D with byte offset writes to correct location."""
    total = 200
    dtype = np.float32
    nbytes = total * np.dtype(dtype).itemsize
    ptr = cuda_malloc(nbytes)
    # write first 100 elements
    part1 = np.arange(100, dtype=dtype)
    if use_async:
        stream = request.getfixturevalue("cuda_stream")
        memcpy_host_to_device_async(ptr, part1, stream)
        stream_synchronize(stream)
    else:
        memcpy_host_to_device(ptr, part1)
    # write next 100 elements at offset
    part2 = np.arange(100, 200, dtype=dtype)
    offset = 100 * np.dtype(dtype).itemsize
    if use_async:
        memcpy_host_to_device_offset_async(ptr, part2, offset, stream)
        stream_synchronize(stream)
    else:
        memcpy_host_to_device_offset(ptr, part2, offset)
    # read all 200 back
    result = np.zeros(total, dtype=dtype)
    memcpy_device_to_host(result, ptr)
    expected = np.arange(200, dtype=dtype)
    np.testing.assert_array_equal(result, expected)
    cuda_free(ptr)


@pytest.mark.parametrize("dtype", DTYPES)
def test_allocate_to_device(dtype) -> None:
    """allocate_to_device copies arrays to GPU, data roundtrips correctly."""
    arr = np.arange(50, dtype=dtype)
    ptrs = allocate_to_device([arr])
    assert len(ptrs) == 1
    assert isinstance(ptrs[0], int)
    assert ptrs[0] != 0
    # verify data by reading back
    result = np.zeros(50, dtype=dtype)
    memcpy_device_to_host(result, ptrs[0])
    np.testing.assert_array_equal(arr, result)
    free_device_ptrs(ptrs)
    # multiple arrays
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


@pytest.mark.parametrize(
    "count",
    [pytest.param(0, id="empty"), pytest.param(1, id="single"), pytest.param(5, id="multiple")],
)
def test_free_device_ptrs(count: int) -> None:
    """free_device_ptrs frees zero or more pointers."""
    ptrs = [cuda_malloc(1024) for _ in range(count)]
    free_device_ptrs(ptrs)
