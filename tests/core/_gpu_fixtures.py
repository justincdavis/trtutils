# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Core GPU test fixtures -- CUDA streams, device memory, kernels."""

from __future__ import annotations

import pytest

from trtutils.core import create_stream, cuda_free, cuda_malloc, destroy_stream
from trtutils.core._context import create_context, destroy_context
from trtutils.core._kernels import Kernel
from trtutils.core._nvrtc import find_cuda_include_dir


@pytest.fixture
def cuda_context():
    """Create and push a CUDA context, destroy after test."""
    ctx = create_context()
    yield ctx
    destroy_context(ctx)


@pytest.fixture
def cuda_stream():
    """Create a CUDA stream for the test, destroy after."""
    stream = create_stream()
    yield stream
    destroy_stream(stream)


@pytest.fixture
def device_ptr():
    """Allocate 1KB of device memory, free after."""
    ptr = cuda_malloc(1024)
    yield ptr
    cuda_free(ptr)


@pytest.fixture
def trivial_kernel(tmp_path, cuda_context):
    """Compile and return a trivial CUDA Kernel, freed after test."""
    code = """\
extern "C" __global__ void trivial_kernel(float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (float)idx;
    }
}
"""
    cu_file = tmp_path / "trivial_kernel.cu"
    cu_file.write_text(code)
    kernel = Kernel(cu_file, "trivial_kernel")
    yield kernel
    kernel.free()


@pytest.fixture
def _nvrtc_cache_clear():
    """Clear find_cuda_include_dir lru_cache around nvrtc tests."""
    find_cuda_include_dir.cache_clear()
    yield
    find_cuda_include_dir.cache_clear()
