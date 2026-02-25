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
