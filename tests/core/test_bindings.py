# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_bindings.py -- Binding dataclass and allocation."""

from __future__ import annotations

import itertools
from unittest.mock import patch

import numpy as np
import pytest

from trtutils._flags import FLAGS
from trtutils.core import _bindings, _buffer
from trtutils.core._bindings import Binding, allocate_bindings, create_binding
from trtutils.core._buffer import Buffer, MemoryLocation


@pytest.mark.parametrize(
    ("dtype", "shape", "bind_id", "name", "is_input"),
    [
        pytest.param(np.float32, (1, 3, 64, 64), 0, "input_0", True, id="f32_image_input"),
        pytest.param(np.float16, (1, 3, 224, 224), 1, "output_0", False, id="f16_large_output"),
        pytest.param(np.int32, (10,), 2, "counts", True, id="i32_1d_input"),
        pytest.param(np.uint8, (4, 4), 3, "mask", False, id="u8_2d_output"),
    ],
)
def test_create_binding(dtype, shape, bind_id, name, is_input):
    """Common cases: verify all binding fields are populated correctly."""
    arr = np.zeros(shape, dtype=dtype)
    binding = create_binding(arr, bind_id=bind_id, name=name, is_input=is_input)
    assert binding.shape == list(shape)
    assert binding.dtype == dtype
    assert binding.allocation != 0
    assert isinstance(binding.host_allocation, np.ndarray)
    assert binding.host_allocation.shape == shape
    assert binding.index == bind_id
    assert binding.name == name
    assert binding.is_input is is_input
    binding.free()


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        pytest.param(dtype, shape, id=f"{np.dtype(dtype).name}_{'x'.join(map(str, shape))}")
        for dtype, shape in itertools.product(
            [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.float16, np.float32],
            [
                (1,),
                (1, 3),
                (1, 1, 100),
                (1, 3, 64, 64),
                (1, 3, 128, 128),
                (10,),
                (4, 4),
                (8,),
            ],
        )
    ],
)
def test_create_binding_shape_dtype(dtype, shape):
    """Exhaustive dtype/shape coverage for create_binding."""
    arr = np.zeros(shape, dtype=dtype)
    binding = create_binding(arr)
    assert binding.shape == list(shape)
    assert binding.dtype == dtype
    assert binding.allocation != 0
    assert isinstance(binding.host_allocation, np.ndarray)
    assert binding.host_allocation.shape == shape
    binding.free()


@pytest.mark.parametrize("use_array_data", [None, True, False])
@pytest.mark.parametrize("pagelocked_mem", [None, True, False])
@pytest.mark.parametrize("unified_mem", [None, True, False])
def test_create_binding_memory(use_array_data, pagelocked_mem, unified_mem):
    """Verify memory flags and use_array_data host-copy behavior."""
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    binding = create_binding(
        arr, use_array_data=use_array_data, pagelocked_mem=pagelocked_mem, unified_mem=unified_mem
    )
    assert binding.allocation != 0
    assert isinstance(binding.host_allocation, np.ndarray)
    if pagelocked_mem is not None:
        assert binding.pagelocked_mem is pagelocked_mem
    if unified_mem is not None:
        assert binding.unified_mem is unified_mem
    if use_array_data is True:
        np.testing.assert_array_equal(binding.host_allocation, arr)
    elif (use_array_data is None or use_array_data is False) and not binding.pagelocked_mem:
        # pagelocked memory (cudaHostAlloc) is not guaranteed to be zero-initialized,
        # so only check zero contents for regular numpy allocations
        np.testing.assert_array_equal(binding.host_allocation, np.zeros_like(arr))
    binding.free()


def test_create_binding_unified_memory_frees_host_only() -> None:
    """Mapped unified host allocations should not be freed with cudaFree."""
    arr = np.zeros((4,), dtype=np.float32)
    binding = _bindings.create_binding(arr, pagelocked_mem=True, unified_mem=True)

    with patch.object(
        _buffer.cudart, "cudaFree", wraps=_buffer.cudart.cudaFree
    ) as cuda_free, patch.object(
        _buffer.cudart,
        "cudaFreeHost",
        wraps=_buffer.cudart.cudaFreeHost,
    ) as cuda_free_host:
        binding.free()

    cuda_free.assert_not_called()
    cuda_free_host.assert_called_once()


@pytest.mark.parametrize(
    ("pagelocked", "unified"),
    [
        pytest.param(True, False, id="pagelocked"),
        pytest.param(False, False, id="default"),
        pytest.param(True, True, id="unified"),
    ],
)
def test_allocate_returns_io_bindings(simple_engine, pagelocked, unified) -> None:
    """allocate_bindings returns non-empty input, output, and allocation lists."""
    engine, context, _stream = simple_engine
    inputs, outputs, allocations = allocate_bindings(
        engine,
        context,
        pagelocked_mem=pagelocked,
        unified_mem=unified,
    )

    # verify the return types and lengths
    expected = engine.num_io_tensors if FLAGS.TRT_10 else engine.num_bindings
    assert isinstance(inputs, list)
    assert isinstance(outputs, list)
    assert isinstance(allocations, list)
    assert len(inputs) > 0
    assert len(outputs) > 0
    assert len(allocations) > 0
    assert len(inputs) + len(outputs) == expected
    assert len(allocations) == expected

    # verify allocations
    for binding in inputs + outputs:
        assert binding.allocation != 0
        assert isinstance(binding.allocation, int)
    for binding in inputs + outputs:
        assert isinstance(binding.host_allocation, np.ndarray)

    # verify is_input setting
    for binding in inputs:
        assert binding.is_input is True
    for binding in outputs:
        assert binding.is_input is False

    for b in inputs + outputs:
        b.free()


@pytest.mark.parametrize("mapped", [False, True])
def test_binding_from_buffers_derives_device(mapped) -> None:
    """from_buffers derives the device buffer when one is not provided."""
    host = Buffer.empty((4, 4), np.dtype(np.float32), MemoryLocation.HOST, mapped=mapped)
    binding = Binding.from_buffers(host, name="derived", is_input=True)
    assert binding.host is host
    assert binding.device.location == MemoryLocation.DEVICE
    assert binding.device.nbytes == host.nbytes
    assert binding.allocation == binding.device.ptr
    assert binding.host_allocation is host.array
    assert binding.name == "derived"
    assert binding.is_input is True
    if mapped:
        # mapped host memory aliases the device pointer; the view is non-owning
        assert binding.device.owns_memory is False
        assert binding.unified_mem is True
    binding.free()


def test_binding_from_buffers_explicit_pair() -> None:
    """from_buffers accepts an explicit host/device pair."""
    host = Buffer.empty((8,), np.dtype(np.float32), MemoryLocation.HOST)
    device = Buffer.empty((8,), np.dtype(np.float32), MemoryLocation.DEVICE)
    binding = Binding.from_buffers(host, device)
    assert binding.device is device
    assert binding.shape == [8]
    assert binding.dtype == np.float32
    binding.free()


def test_binding_from_buffers_size_mismatch() -> None:
    """Mismatched buffer sizes raise a ValueError."""
    host = Buffer.empty((8,), np.dtype(np.float32), MemoryLocation.HOST)
    device = Buffer.empty((4,), np.dtype(np.float32), MemoryLocation.DEVICE)
    with pytest.raises(ValueError, match="size mismatch"):
        Binding.from_buffers(host, device)
    host.free()
    device.free()


def test_binding_from_buffers_wrong_locations() -> None:
    """Buffers in the wrong memory spaces raise a ValueError."""
    host = Buffer.empty((4,), np.dtype(np.float32), MemoryLocation.HOST)
    device = Buffer.empty((4,), np.dtype(np.float32), MemoryLocation.DEVICE)
    with pytest.raises(ValueError, match="must reside"):
        Binding.from_buffers(device, host)
    host.free()
    device.free()


@pytest.mark.parametrize(
    ("pagelocked", "unified"),
    [
        pytest.param(True, False, id="pagelocked"),
        pytest.param(False, False, id="pageable"),
        pytest.param(True, True, id="unified"),
    ],
)
def test_binding_upload_download_roundtrip(pagelocked, unified) -> None:
    """Upload followed by download recovers the data in all memory modes."""
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    binding = create_binding(arr, pagelocked_mem=pagelocked, unified_mem=unified)
    binding.upload(arr)
    result = binding.download()
    np.testing.assert_array_equal(result, arr)
    binding.free()
