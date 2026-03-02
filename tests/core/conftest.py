# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Core test fixtures -- CUDA streams, temp engine files, etc."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def simple_onnx_path() -> Path:
    """Path to a minimal ONNX model for core tests."""
    return Path(__file__).parent.parent.parent / "data" / "simple.onnx"


@pytest.fixture(scope="session")
def simple_engine_path(build_test_engine, simple_onnx_path) -> Path:
    """Build and return path to a simple test engine."""
    return build_test_engine(simple_onnx_path)


@pytest.fixture(scope="session")
def _cuda_graph_api_compatible() -> bool:
    """Check if CUDA graph APIs work correctly on this cuda-python version."""
    try:
        from trtutils._flags import FLAGS
        from trtutils.compat._libs import cudart
        from trtutils.core._cuda import cuda_call

        stream = cuda_call(cudart.cudaStreamCreate())
        mode = cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
        cuda_call(cudart.cudaStreamBeginCapture(stream, mode))
        graph = cuda_call(cudart.cudaStreamEndCapture(stream))
        # Use the flag to select the correct API signature
        if FLAGS.CUDA_PYTHON_12:
            graph_exec = cuda_call(cudart.cudaGraphInstantiate(graph, 0))
        else:
            graph_exec = cuda_call(cudart.cudaGraphInstantiate(graph, b"", 0))
        cuda_call(cudart.cudaGraphExecDestroy(graph_exec))
        cuda_call(cudart.cudaGraphDestroy(graph))
        cuda_call(cudart.cudaStreamDestroy(stream))
        return True
    except (TypeError, RuntimeError, Exception):
        return False


@pytest.fixture(autouse=True)
def _skip_cuda_graph_tests(request: pytest.FixtureRequest, _cuda_graph_api_compatible: bool) -> None:
    """Skip CUDA graph tests if the graph API is incompatible."""
    if request.node.get_closest_marker("cuda_graph") and not _cuda_graph_api_compatible:
        pytest.skip("CUDA graph API incompatible with this cuda-python version")
    # Also check by fixture name — graph tests use cuda_stream fixture
    if (
        "cuda_stream" in getattr(request, "fixturenames", [])
        and "CUDAGraph" in (request.node.nodeid or "")
        and not _cuda_graph_api_compatible
    ):
        pytest.skip("CUDA graph API incompatible with this cuda-python version")


@pytest.fixture
def cuda_stream():
    """Create a CUDA stream for the test, destroy after."""
    from trtutils.core import create_stream, destroy_stream

    stream = create_stream()
    yield stream
    destroy_stream(stream)


@pytest.fixture
def device_ptr():
    """Allocate 1KB of device memory, free after."""
    from trtutils.core import cuda_free, cuda_malloc

    ptr = cuda_malloc(1024)
    yield ptr
    cuda_free(ptr)


@pytest.fixture
def simple_engine(simple_engine_path):
    """Load simple test engine, destroy stream after test."""
    from trtutils.core._engine import create_engine
    from trtutils.core._stream import destroy_stream

    engine, context, _logger, stream = create_engine(simple_engine_path)
    yield engine, context, stream
    destroy_stream(stream)


@pytest.fixture
def patched_cache_dir(tmp_path, monkeypatch):
    """Provide a temporary cache directory with get_cache_dir patched."""
    from trtutils.core import cache

    cache_dir = tmp_path / "_engine_cache"
    cache_dir.mkdir()
    monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)
    return cache_dir


@pytest.fixture
def trivial_kernel(tmp_path):
    """Compile and return a trivial CUDA Kernel, freed after test."""
    from trtutils.core._kernels import Kernel

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


@pytest.fixture(autouse=True)
def _nvrtc_cache_clear(request):
    """Clear find_cuda_include_dir lru_cache around nvrtc tests."""
    if "nvrtc" not in request.node.nodeid:
        yield
        return
    from trtutils.core._nvrtc import find_cuda_include_dir

    find_cuda_include_dir.cache_clear()
    yield
    find_cuda_include_dir.cache_clear()
