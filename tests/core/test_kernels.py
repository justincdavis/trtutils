# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_kernels.py -- Kernel class and arg conversion."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.core._kernels import Kernel, create_kernel_args, launch_kernel
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
    """Kernel.create_args returns ndarray, caching respects max_arg_cache."""
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


@pytest.mark.usefixtures("cuda_context")
def test_kernel_create_args_caching(tmp_path) -> None:
    """Kernel arg cache evicts entries at max_arg_cache."""
    cu_file = tmp_path / "trivial_kernel.cu"
    cu_file.write_text(TRIVIAL_KERNEL_CODE)
    kernel = Kernel(cu_file, "trivial_kernel", max_arg_cache=2)
    kernel.create_args(1, 2)
    assert len(kernel._inter_args) == 1
    kernel.create_args(3, 4)
    assert len(kernel._inter_args) == 2
    kernel.create_args(5, 6)
    assert len(kernel._inter_args) == 2
    kernel.free()


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
