# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_buffer.py -- Buffer and MemoryLocation."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.core._buffer import Buffer, MemoryLocation
from trtutils.core._stream import create_stream, destroy_stream, stream_synchronize

# ----------------------------------------------------------------------
# constructors
# ----------------------------------------------------------------------


def test_empty_device() -> None:
    """Buffer.empty allocates device memory with the requested shape/dtype."""
    buffer = Buffer.empty((4, 4), np.dtype(np.float32), MemoryLocation.DEVICE)
    assert buffer.location == MemoryLocation.DEVICE
    assert buffer.shape == (4, 4)
    assert buffer.dtype == np.float32
    assert buffer.size == 16
    assert buffer.nbytes == 16 * 4
    assert buffer.ptr != 0
    buffer.free()


def test_empty_host_pinned() -> None:
    """Buffer.empty allocates pagelocked host memory by default."""
    buffer = Buffer.empty((8,), np.dtype(np.uint8), MemoryLocation.HOST)
    assert buffer.location == MemoryLocation.HOST
    assert buffer.pinned is True
    assert buffer.mapped is False
    assert isinstance(buffer.array, np.ndarray)
    assert buffer.array.shape == (8,)
    buffer.free()


def test_empty_host_pageable() -> None:
    """Buffer.empty(pinned=False) allocates plain pageable numpy memory."""
    buffer = Buffer.empty((8,), np.dtype(np.uint8), MemoryLocation.HOST, pinned=False)
    assert buffer.pinned is False
    assert buffer.mapped is False
    np.testing.assert_array_equal(buffer.array, np.zeros(8, dtype=np.uint8))
    buffer.free()


def test_empty_host_mapped() -> None:
    """Buffer.empty(mapped=True) exposes a device-visible alias pointer."""
    buffer = Buffer.empty((4,), np.dtype(np.float32), MemoryLocation.HOST, pinned=True, mapped=True)
    assert buffer.pinned is True
    assert buffer.mapped is True
    assert buffer.device_ptr != 0
    buffer.free()


def test_init_host_requires_array() -> None:
    """Constructing a host Buffer without a backing array raises ValueError."""
    with pytest.raises(ValueError, match="backing numpy array"):
        Buffer(MemoryLocation.HOST, np.dtype(np.float32), (4,), 0)


# ----------------------------------------------------------------------
# from_array
# ----------------------------------------------------------------------


def test_from_array_device_roundtrip() -> None:
    """from_array on DEVICE copies data that numpy() reads back correctly."""
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    buffer = Buffer.from_array(arr, MemoryLocation.DEVICE)
    assert buffer.location == MemoryLocation.DEVICE
    assert buffer.shape == arr.shape
    np.testing.assert_array_equal(buffer.numpy(), arr)
    buffer.free()


def test_from_array_host_roundtrip() -> None:
    """from_array on HOST copies data into the backing array."""
    arr = np.arange(12, dtype=np.int32).reshape(3, 4)
    buffer = Buffer.from_array(arr, MemoryLocation.HOST)
    assert buffer.location == MemoryLocation.HOST
    np.testing.assert_array_equal(buffer.array, arr)
    np.testing.assert_array_equal(buffer.numpy(), arr)
    buffer.free()


# ----------------------------------------------------------------------
# from_ptr
# ----------------------------------------------------------------------


def test_from_ptr_device_non_owning() -> None:
    """from_ptr creates a non-owning view that does not free the allocation."""
    owner = Buffer.empty((4,), np.dtype(np.float32), MemoryLocation.DEVICE)
    view = Buffer.from_ptr(owner.ptr, (4,), np.dtype(np.float32), MemoryLocation.DEVICE)
    assert view.owns_memory is False
    assert view.ptr == owner.ptr
    # freeing the view is a no-op; the owner's allocation is unaffected
    view.free()
    np.testing.assert_array_equal(owner.numpy(), np.zeros(4, dtype=np.float32))
    owner.free()


def test_from_ptr_keeps_owner_alive() -> None:
    """from_ptr(owner=...) holds a reference that keeps the owner alive."""
    owner = Buffer.empty((4,), np.dtype(np.float32), MemoryLocation.DEVICE)
    ptr = owner.ptr
    view = Buffer.from_ptr(ptr, (4,), np.dtype(np.float32), MemoryLocation.DEVICE, owner=owner)
    assert view._parent is owner
    del owner
    # the view keeps the owner (and thus the allocation) alive
    view.copy_from(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    np.testing.assert_array_equal(view.numpy(), np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    view.free()


def test_from_ptr_host_view() -> None:
    """from_ptr on HOST wraps existing host memory as a numpy view."""
    arr = np.arange(8, dtype=np.float32)
    view = Buffer.from_ptr(
        arr.ctypes.data, (8,), np.dtype(np.float32), MemoryLocation.HOST, owner=arr
    )
    np.testing.assert_array_equal(view.array, arr)
    view.array[0] = 42.0
    assert arr[0] == 42.0
    view.free()


# ----------------------------------------------------------------------
# __getitem__
# ----------------------------------------------------------------------


def test_getitem_int() -> None:
    """Integer indexing returns a non-owning view of one leading-axis slot."""
    buffer = Buffer.empty((4, 8), np.dtype(np.float32), MemoryLocation.DEVICE)
    view = buffer[1]
    assert view.shape == (8,)
    assert view.owns_memory is False
    assert view.ptr == buffer.ptr + 8 * 4
    buffer.free()


def test_getitem_negative() -> None:
    """Negative indices index from the end of the leading axis."""
    buffer = Buffer.empty((4, 8), np.dtype(np.float32), MemoryLocation.DEVICE)
    assert buffer[-1].ptr == buffer[3].ptr
    buffer.free()


def test_getitem_slice() -> None:
    """Slice indexing (step 1) returns a contiguous multi-element view."""
    buffer = Buffer.empty((4, 8), np.dtype(np.float32), MemoryLocation.DEVICE)
    view = buffer[1:3]
    assert view.shape == (2, 8)
    assert view.ptr == buffer.ptr + 8 * 4
    buffer.free()


def test_getitem_out_of_range() -> None:
    """Out-of-range integer indices raise IndexError."""
    buffer = Buffer.empty((4, 8), np.dtype(np.float32), MemoryLocation.DEVICE)
    with pytest.raises(IndexError, match="out of range"):
        _ = buffer[10]
    buffer.free()


def test_getitem_step_not_one_raises() -> None:
    """Slices with step != 1 raise ValueError."""
    buffer = Buffer.empty((4, 8), np.dtype(np.float32), MemoryLocation.DEVICE)
    with pytest.raises(ValueError, match="step 1"):
        _ = buffer[::2]
    buffer.free()


def test_getitem_zero_dim_raises() -> None:
    """Indexing a 0-dimensional buffer raises IndexError."""
    buffer = Buffer.empty((), np.dtype(np.float32), MemoryLocation.DEVICE)
    with pytest.raises(IndexError, match="0-dimensional"):
        _ = buffer[0]
    buffer.free()


# ----------------------------------------------------------------------
# reshape
# ----------------------------------------------------------------------


def test_reshape() -> None:
    """Reshape returns a non-owning view with the new shape over the same memory."""
    buffer = Buffer.from_array(np.arange(12, dtype=np.float32), MemoryLocation.DEVICE)
    view = buffer.reshape((3, 4))
    assert view.shape == (3, 4)
    assert view.ptr == buffer.ptr
    assert view.owns_memory is False
    np.testing.assert_array_equal(view.numpy(), buffer.numpy().reshape(3, 4))
    buffer.free()


def test_reshape_size_mismatch_raises() -> None:
    """Reshape to a different element count raises ValueError."""
    buffer = Buffer.empty((12,), np.dtype(np.float32), MemoryLocation.DEVICE)
    with pytest.raises(ValueError, match="Cannot reshape"):
        buffer.reshape((5, 5))
    buffer.free()


# ----------------------------------------------------------------------
# copy_to / copy_from
# ----------------------------------------------------------------------


def test_copy_host_to_host() -> None:
    """H2H copy_to transfers between two host buffers."""
    src = Buffer.from_array(np.arange(8, dtype=np.float32), MemoryLocation.HOST)
    dst = Buffer.empty((8,), np.dtype(np.float32), MemoryLocation.HOST)
    src.copy_to(dst)
    np.testing.assert_array_equal(dst.array, src.array)
    src.free()
    dst.free()


def test_copy_host_to_device() -> None:
    """H2D copy_to transfers from a host buffer into a device buffer."""
    src = Buffer.from_array(np.arange(8, dtype=np.float32), MemoryLocation.HOST)
    dst = Buffer.empty((8,), np.dtype(np.float32), MemoryLocation.DEVICE)
    src.copy_to(dst)
    np.testing.assert_array_equal(dst.numpy(), src.array)
    src.free()
    dst.free()


def test_copy_device_to_host() -> None:
    """D2H copy_to transfers from a device buffer into a host buffer."""
    src = Buffer.from_array(np.arange(8, dtype=np.float32), MemoryLocation.DEVICE)
    dst = Buffer.empty((8,), np.dtype(np.float32), MemoryLocation.HOST)
    src.copy_to(dst)
    np.testing.assert_array_equal(dst.array, np.arange(8, dtype=np.float32))
    src.free()
    dst.free()


def test_copy_device_to_device() -> None:
    """D2D copy_to transfers between two device buffers."""
    src = Buffer.from_array(np.arange(8, dtype=np.float32), MemoryLocation.DEVICE)
    dst = Buffer.empty((8,), np.dtype(np.float32), MemoryLocation.DEVICE)
    src.copy_to(dst)
    np.testing.assert_array_equal(dst.numpy(), np.arange(8, dtype=np.float32))
    src.free()
    dst.free()


@pytest.mark.parametrize(
    "location",
    [
        pytest.param(MemoryLocation.HOST, id="host"),
        pytest.param(MemoryLocation.DEVICE, id="device"),
    ],
)
def test_copy_to_numpy_target(location) -> None:
    """copy_to accepts a plain numpy array as the destination."""
    src = Buffer.from_array(np.arange(8, dtype=np.float32), location)
    dst = np.zeros(8, dtype=np.float32)
    src.copy_to(dst)
    np.testing.assert_array_equal(dst, np.arange(8, dtype=np.float32))
    src.free()


@pytest.mark.parametrize(
    "location",
    [
        pytest.param(MemoryLocation.HOST, id="host"),
        pytest.param(MemoryLocation.DEVICE, id="device"),
    ],
)
def test_copy_from_numpy_source(location) -> None:
    """copy_from accepts a plain numpy array as the source."""
    dst = Buffer.empty((8,), np.dtype(np.float32), location)
    src = np.arange(8, dtype=np.float32)
    dst.copy_from(src)
    np.testing.assert_array_equal(dst.numpy(), src)
    dst.free()


def test_copy_to_buffer_size_mismatch_raises() -> None:
    """copy_to between mismatched-size Buffers raises ValueError."""
    src = Buffer.empty((8,), np.dtype(np.float32), MemoryLocation.DEVICE)
    dst = Buffer.empty((4,), np.dtype(np.float32), MemoryLocation.DEVICE)
    with pytest.raises(ValueError, match="size mismatch"):
        src.copy_to(dst)
    src.free()
    dst.free()


def test_copy_from_buffer_size_mismatch_raises() -> None:
    """copy_from between mismatched-size Buffers raises ValueError."""
    dst = Buffer.empty((8,), np.dtype(np.float32), MemoryLocation.DEVICE)
    src = Buffer.empty((4,), np.dtype(np.float32), MemoryLocation.DEVICE)
    with pytest.raises(ValueError, match="size mismatch"):
        dst.copy_from(src)
    dst.free()
    src.free()


def test_copy_to_numpy_size_mismatch_raises() -> None:
    """copy_to a mismatched-size numpy array raises ValueError."""
    buffer = Buffer.empty((8,), np.dtype(np.float32), MemoryLocation.DEVICE)
    with pytest.raises(ValueError, match="size mismatch"):
        buffer.copy_to(np.zeros(4, dtype=np.float32))
    buffer.free()


def test_copy_from_numpy_size_mismatch_raises() -> None:
    """copy_from a mismatched-size numpy array raises ValueError."""
    buffer = Buffer.empty((8,), np.dtype(np.float32), MemoryLocation.DEVICE)
    with pytest.raises(ValueError, match="size mismatch"):
        buffer.copy_from(np.zeros(4, dtype=np.float32))
    buffer.free()


def test_copy_async_roundtrip() -> None:
    """Asynchronous H2D + D2H copies round-trip after stream synchronize."""
    stream = create_stream()
    arr = np.arange(8, dtype=np.float32)
    buffer = Buffer.empty((8,), np.dtype(np.float32), MemoryLocation.DEVICE)
    buffer.copy_from(arr, stream)
    out = np.zeros(8, dtype=np.float32)
    buffer.copy_to(out, stream)
    stream_synchronize(stream)
    np.testing.assert_array_equal(out, arr)
    buffer.free()
    destroy_stream(stream)


def test_to_device_and_to_host() -> None:
    """to_device/to_host allocate a fresh buffer in the target location."""
    host = Buffer.from_array(np.arange(8, dtype=np.float32), MemoryLocation.HOST)
    device = host.to_device()
    assert device.location == MemoryLocation.DEVICE
    np.testing.assert_array_equal(device.numpy(), host.array)
    back = device.to_host()
    assert back.location == MemoryLocation.HOST
    np.testing.assert_array_equal(back.array, host.array)
    host.free()
    device.free()
    back.free()


# ----------------------------------------------------------------------
# numpy()
# ----------------------------------------------------------------------


def test_numpy_host_is_zero_copy() -> None:
    """numpy() on a host buffer returns the backing array itself."""
    buffer = Buffer.from_array(np.arange(4, dtype=np.float32), MemoryLocation.HOST)
    assert buffer.numpy() is buffer.array
    buffer.free()


def test_numpy_device_copies() -> None:
    """numpy() on a device buffer performs a device-to-host copy."""
    arr = np.arange(4, dtype=np.float32)
    buffer = Buffer.from_array(arr, MemoryLocation.DEVICE)
    result = buffer.numpy()
    np.testing.assert_array_equal(result, arr)
    buffer.free()


# ----------------------------------------------------------------------
# free
# ----------------------------------------------------------------------


def test_free_is_idempotent() -> None:
    """Calling free() more than once is safe."""
    buffer = Buffer.empty((4,), np.dtype(np.float32), MemoryLocation.DEVICE)
    buffer.free()
    buffer.free()


def test_free_is_noop_for_views() -> None:
    """free() on a non-owning view never frees the parent's allocation."""
    owner = Buffer.empty((4,), np.dtype(np.float32), MemoryLocation.DEVICE)
    view = owner[0:2]
    view.free()
    assert owner._freed is False
    # the owner's memory is still valid
    owner.copy_from(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    np.testing.assert_array_equal(owner.numpy(), np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    owner.free()


def test_context_manager_frees() -> None:
    """Using a Buffer as a context manager frees it on exit."""
    with Buffer.empty((4,), np.dtype(np.float32), MemoryLocation.DEVICE) as buffer:
        assert buffer._freed is False
    assert buffer._freed is True


# ----------------------------------------------------------------------
# __cuda_array_interface__
# ----------------------------------------------------------------------


def test_cuda_array_interface_device() -> None:
    """Device buffers expose a valid CUDA Array Interface."""
    buffer = Buffer.empty((4, 4), np.dtype(np.float32), MemoryLocation.DEVICE)
    iface = buffer.__cuda_array_interface__
    assert iface["shape"] == (4, 4)
    assert iface["data"] == (buffer.ptr, False)
    assert iface["version"] == 3
    buffer.free()


def test_cuda_array_interface_mapped_host() -> None:
    """Mapped host buffers expose the device alias pointer in the interface."""
    buffer = Buffer.empty((4,), np.dtype(np.float32), MemoryLocation.HOST, pinned=True, mapped=True)
    iface = buffer.__cuda_array_interface__
    assert iface["data"] == (buffer.device_ptr, False)
    buffer.free()


def test_device_ptr_unmapped_host_raises() -> None:
    """device_ptr on an unmapped host buffer raises RuntimeError."""
    buffer = Buffer.empty((4,), np.dtype(np.float32), MemoryLocation.HOST, pinned=True, mapped=False)
    with pytest.raises(RuntimeError, match="not mapped"):
        _ = buffer.device_ptr
    buffer.free()


def test_array_dunder() -> None:
    """__array__ supports implicit numpy conversion for host buffers."""
    buffer = Buffer.from_array(np.arange(4, dtype=np.float32), MemoryLocation.HOST)
    converted = np.asarray(buffer)
    np.testing.assert_array_equal(converted, buffer.array)
    buffer.free()


def test_array_property_device_raises() -> None:
    """The array property on a device buffer raises RuntimeError."""
    buffer = Buffer.empty((4,), np.dtype(np.float32), MemoryLocation.DEVICE)
    with pytest.raises(RuntimeError, match="Device buffers"):
        _ = buffer.array
    buffer.free()
