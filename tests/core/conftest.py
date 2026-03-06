# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Core test fixtures -- CUDA streams, device memory, etc."""

from __future__ import annotations

import pytest

from trtutils.core import create_stream, cuda_free, cuda_malloc, destroy_stream


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
