# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_kernels.py -- Kernel class and arg conversion."""

from __future__ import annotations

import ctypes
import gc

import numpy as np
import pytest

from trtutils.core._kernels import Kernel, KernelArgs, create_kernel_args, launch_kernel
from trtutils.core._memory import cuda_free, cuda_malloc, memcpy_device_to_host
from trtutils.core._stream import stream_synchronize

TRIVIAL_KERNEL_CODE = """\
extern "C" __global__ void trivial_kernel(float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (float)idx;
    }
}
"""


def test_create_kernel_args() -> None:
    """create_kernel_args handles int, float, ndarray, multiple args, and verbose."""
    # int arg
    ptrs, intermediates = create_kernel_args(42)
    assert ptrs.dtype == np.uint64
    assert len(ptrs) == 1
    assert intermediates[0].dtype == np.uint64
    # float arg
    ptrs, intermediates = create_kernel_args(3.14)
    assert len(ptrs) == 1
    assert intermediates[0].dtype == np.float32
    # ndarray arg
    arr = np.array([100], dtype=np.uint64)
    ptrs, intermediates = create_kernel_args(arr)
    assert len(ptrs) == 1
    assert intermediates[0] is arr
    # multiple args
    ptrs, intermediates = create_kernel_args(10, 20.0, 30)
    assert len(ptrs) == 3
    assert len(intermediates) == 3
    # verbose
    ptrs, _ = create_kernel_args(42, 3.14, verbose=True)
    assert len(ptrs) == 2


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(np.float32, id="float32"),
        pytest.param(np.float16, id="float16"),
        pytest.param(np.int32, id="int32"),
        pytest.param(np.uint8, id="uint8"),
    ],
)
def test_create_kernel_args_ndarray_dtypes(dtype: np.dtype) -> None:
    """create_kernel_args works with various ndarray dtypes."""
    arr = np.array([1], dtype=dtype)
    ptrs, intermediates = create_kernel_args(arr)
    assert len(ptrs) == 1
    assert intermediates[0] is arr


def test_create_kernel_args_unsupported_raises() -> None:
    """Unsupported arg type raises TypeError."""
    with pytest.raises(TypeError, match="Unrecognized arg type"):
        create_kernel_args("not_a_valid_arg")


@pytest.mark.parametrize(
    "path_type",
    [
        pytest.param(lambda p: p, id="Path"),
        pytest.param(str, id="str"),
    ],
)
@pytest.mark.usefixtures("cuda_context")
def test_kernel_compile(tmp_path, path_type) -> None:
    """Kernel compiles from Path or str, free unloads module."""
    cu_file = tmp_path / "trivial_kernel.cu"
    cu_file.write_text(TRIVIAL_KERNEL_CODE)
    kernel = Kernel(path_type(cu_file), "trivial_kernel")
    assert kernel._kernel is not None
    assert kernel._module is not None
    # free unloads module
    kernel.free()
    assert kernel._module is None
    assert kernel._freed is True


def test_kernel_create_args(trivial_kernel) -> None:
    """Kernel.create_args returns a pointer ndarray with one entry per argument."""
    # returns ndarray with correct dtype and length
    args = trivial_kernel.create_args(42, 10)
    assert isinstance(args, np.ndarray)
    assert args.dtype == np.uint64
    assert len(args) == 2
    # pointer arg works
    ptr = cuda_malloc(1024)
    args = trivial_kernel.create_args(ptr, 256)
    assert len(args) == 2
    cuda_free(ptr)


def test_kernel_create_args_owns_intermediates(trivial_kernel) -> None:
    """The returned KernelArgs keeps the argument buffers its pointers reference."""
    args = trivial_kernel.create_args(1, 2)
    assert isinstance(args, KernelArgs)
    assert len(args._keepalive) == 2
    # views of the argument array keep them too
    assert args[:1]._keepalive is args._keepalive


@pytest.mark.parametrize(
    "use_call",
    [
        pytest.param(False, id="launch_kernel"),
        pytest.param(True, id="__call__"),
    ],
)
def test_kernel_launch(trivial_kernel, cuda_stream, use_call) -> None:
    """Kernel launch produces correct output via launch_kernel and __call__."""
    n = 32
    d_out = cuda_malloc(n * np.dtype(np.float32).itemsize)
    args = trivial_kernel.create_args(d_out, n)

    if use_call:
        trivial_kernel((1, 1, 1), (n, 1, 1), cuda_stream, args)
    else:
        launch_kernel(trivial_kernel._kernel, (1, 1, 1), (n, 1, 1), cuda_stream, args)

    stream_synchronize(cuda_stream)
    result = np.zeros(n, dtype=np.float32)
    memcpy_device_to_host(result, d_out)
    np.testing.assert_array_equal(result, np.arange(n, dtype=np.float32))
    cuda_free(d_out)


def test_kernel_launch_verbose(trivial_kernel, cuda_stream) -> None:
    """Kernel.call with verbose=True does not raise."""
    n = 8
    d_out = cuda_malloc(n * np.dtype(np.float32).itemsize)
    args = trivial_kernel.create_args(d_out, n)
    trivial_kernel.call((1, 1, 1), (n, 1, 1), cuda_stream, args, verbose=True)
    stream_synchronize(cuda_stream)
    cuda_free(d_out)


@pytest.mark.regression
def test_cached_args_survive_intermediate_eviction(trivial_kernel, cuda_stream) -> None:
    """
    A cached argument array stays valid after other create_args calls.

    Kernel argument arrays hold pointers into separately allocated buffers.
    Those buffers used to be retained only by a bounded per-kernel deque, so
    a caller that cached an argument array and reused it later (as the CUDA
    preprocessor does, keyed by batch size) launched against freed memory
    once other batch sizes pushed its buffers out of the deque. Regression
    test for that use-after-free.
    """
    n = 32
    d_out = cuda_malloc(n * np.dtype(np.float32).itemsize)
    d_other = cuda_malloc(n * np.dtype(np.float32).itemsize)
    cached_args = trivial_kernel.create_args(d_out, n)

    # churn argument arrays for a *different* destination, as alternating
    # batch sizes do. this both evicts the cached array's buffers from the
    # bounded deque and refills the freed blocks with a different pointer,
    # so a reclaimed buffer is distinguishable from a retained one.
    for _ in range(512):
        trivial_kernel.create_args(d_other, n)
    gc.collect()

    # the first argument buffer holds the destination device pointer. read it
    # back through the address the args array carries: if the buffer was
    # reclaimed, this reads whatever now occupies that memory instead.
    stored_ptr = ctypes.c_uint64.from_address(int(cached_args[0])).value
    assert stored_ptr == d_out, "argument buffer was reclaimed while still referenced"

    trivial_kernel.call((1, 1, 1), (n, 1, 1), cuda_stream, cached_args)
    stream_synchronize(cuda_stream)

    result = np.zeros(n, dtype=np.float32)
    memcpy_device_to_host(result, d_out)
    np.testing.assert_array_equal(result, np.arange(n, dtype=np.float32))
    cuda_free(d_out)
    cuda_free(d_other)


@pytest.mark.regression
def test_create_args_owns_its_buffers(trivial_kernel) -> None:
    """The returned argument array keeps the buffers its pointers reference."""
    args = trivial_kernel.create_args(1234, 7)
    # the pointers are meaningless without the buffers they point into, so
    # the array must own them rather than rely on the kernel outliving it
    assert getattr(args, "_keepalive", None), "argument array does not retain its buffers"
    assert len(args._keepalive) == len(args)
