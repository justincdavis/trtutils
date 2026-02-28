# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import pytest


@pytest.fixture
def cuda_stream():
    """Create and destroy a CUDA stream for kernel tests."""
    from trtutils.core import create_stream, destroy_stream

    stream = create_stream()
    yield stream
    destroy_stream(stream)
