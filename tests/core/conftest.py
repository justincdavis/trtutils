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
