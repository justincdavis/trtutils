# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for N-D (1D-5D) memory transfers via Buffer and the nd memcpy suite."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.core._buffer import Buffer, MemoryLocation
from trtutils.core._memory import (
    cuda_free,
    cuda_malloc,
    memcpy_nd_device_to_host,
    memcpy_nd_host_to_device,
)
from trtutils.core._stream import create_stream, destroy_stream, stream_synchronize

_SHAPES = [
    pytest.param((32,), id="1d"),
    pytest.param((8, 16), id="2d"),
    pytest.param((4, 8, 16), id="3d"),
    pytest.param((2, 4, 8, 16), id="4d"),
    pytest.param((2, 3, 4, 8, 16), id="5d"),
]


def _views(base: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Build a set of strided views of an array covering common patterns."""
    views: list[tuple[str, np.ndarray]] = [("contiguous", base)]
    if base.ndim >= 2:
        views.append(("transpose", base.T))
        views.append(("step", base[::2]))
        views.append(("reversed", base[::-1]))
    if base.ndim >= 4:
        views.append(("inner_step", base[:, ::2]))
    return views


@pytest.mark.parametrize("shape", _SHAPES)
def test_buffer_contiguous_roundtrip(shape) -> None:
    """Contiguous N-D arrays round-trip through a device Buffer."""
    arr = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    buffer = Buffer.from_array(arr, MemoryLocation.DEVICE)
    assert buffer.shape == shape
    result = buffer.numpy()
    np.testing.assert_array_equal(result, arr)
    buffer.free()


@pytest.mark.parametrize("shape", _SHAPES)
def test_buffer_strided_copy_from(shape) -> None:
    """Non-contiguous N-D views transfer correctly into device Buffers."""
    base = np.random.default_rng(1).standard_normal(shape).astype(np.float32)
    for name, view in _views(base):
        buffer = Buffer.empty(view.shape, np.dtype(np.float32), MemoryLocation.DEVICE)
        buffer.copy_from(view)
        result = buffer.numpy()
        np.testing.assert_array_equal(result, view, err_msg=f"view: {name}")
        buffer.free()


@pytest.mark.parametrize("shape", _SHAPES)
def test_buffer_strided_copy_to(shape) -> None:
    """Device Buffers transfer correctly into non-contiguous N-D views."""
    rng = np.random.default_rng(2)
    for name, pattern in _views(np.zeros(shape, dtype=np.float32)):
        expected = rng.standard_normal(pattern.shape).astype(np.float32)
        buffer = Buffer.from_array(expected, MemoryLocation.DEVICE)
        # carve an identically-shaped strided destination out of a fresh array
        dst_base = np.zeros(shape, dtype=np.float32)
        dst_view = {
            "contiguous": dst_base,
            "transpose": dst_base.T,
            "step": dst_base[::2],
            "reversed": dst_base[::-1],
            "inner_step": dst_base[:, ::2] if dst_base.ndim >= 4 else dst_base,
        }[name]
        buffer.copy_to(dst_view)
        np.testing.assert_array_equal(dst_view, expected, err_msg=f"view: {name}")
        buffer.free()


@pytest.mark.parametrize("shape", _SHAPES)
def test_buffer_strided_async_roundtrip(shape) -> None:
    """Asynchronous strided transfers round-trip after stream synchronize."""
    stream = create_stream()
    base = np.random.default_rng(3).standard_normal(shape).astype(np.float32)
    view = base[::2] if base.ndim >= 2 else base
    buffer = Buffer.empty(view.shape, np.dtype(np.float32), MemoryLocation.DEVICE)
    buffer.copy_from(view, stream)
    out = np.zeros(view.shape, dtype=np.float32)
    buffer.copy_to(out, stream)
    stream_synchronize(stream)
    np.testing.assert_array_equal(out, view)
    buffer.free()
    destroy_stream(stream)


@pytest.mark.parametrize("shape", _SHAPES)
def test_memcpy_nd_functions(shape) -> None:
    """The raw memcpy_nd functions round-trip strided views."""
    base = np.random.default_rng(4).standard_normal(shape).astype(np.float32)
    view = base.T if base.ndim >= 2 else base
    nbytes = view.size * view.itemsize
    ptr = cuda_malloc(nbytes)
    memcpy_nd_host_to_device(ptr, view)
    out_base = np.zeros(shape, dtype=np.float32)
    out_view = out_base.T if out_base.ndim >= 2 else out_base
    memcpy_nd_device_to_host(out_view, ptr)
    np.testing.assert_array_equal(out_view, view)
    cuda_free(ptr)


def test_buffer_getitem_batch_slots() -> None:
    """Leading-axis indexing transfers into individual batch slots of a 4D buffer."""
    batch, shape = 4, (3, 8, 8)
    buffer = Buffer.empty((batch, *shape), np.dtype(np.float32), MemoryLocation.DEVICE)
    rng = np.random.default_rng(5)
    images = [rng.standard_normal(shape).astype(np.float32) for _ in range(batch)]
    for i, image in enumerate(images):
        slot = buffer[i]
        assert slot.shape == shape
        assert slot.owns_memory is False
        slot.copy_from(image)
    result = buffer.numpy()
    for i, image in enumerate(images):
        np.testing.assert_array_equal(result[i], image)
    # slice views select contiguous ranges
    tail = buffer[2:4]
    assert tail.shape == (2, *shape)
    np.testing.assert_array_equal(tail.numpy(), np.stack(images[2:4]))
    buffer.free()


def test_buffer_getitem_host_view() -> None:
    """Host buffer views share memory with the parent."""
    buffer = Buffer.from_array(np.arange(24, dtype=np.float32).reshape(4, 6))
    view = buffer[1]
    assert view.shape == (6,)
    np.testing.assert_array_equal(view.array, np.arange(6, 12, dtype=np.float32))
    view.array[:] = 0.0
    np.testing.assert_array_equal(buffer.array[1], np.zeros(6, dtype=np.float32))
    buffer.free()


def test_buffer_getitem_validation() -> None:
    """Out-of-range indices and stepped slices are rejected."""
    buffer = Buffer.empty((4, 8), np.dtype(np.float32), MemoryLocation.DEVICE)
    with pytest.raises(IndexError, match="out of range"):
        _ = buffer[4]
    with pytest.raises(ValueError, match="step 1"):
        _ = buffer[::2]
    negative = buffer[-1]
    assert negative.ptr == buffer.ptr + 3 * 8 * 4
    buffer.free()


def test_buffer_size_mismatch_raises() -> None:
    """Mismatched transfer sizes raise a ValueError in both directions."""
    buffer = Buffer.empty((4, 4), np.dtype(np.float32), MemoryLocation.DEVICE)
    small = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="size mismatch"):
        buffer.copy_from(small)
    with pytest.raises(ValueError, match="size mismatch"):
        buffer.copy_to(small)
    buffer.free()
