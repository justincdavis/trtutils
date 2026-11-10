# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_stream.py -- CUDA stream lifecycle."""

from __future__ import annotations

import pytest

from trtutils.compat._libs import cudart
from trtutils.core._cuda import cuda_call
from trtutils.core._stream import create_stream, destroy_stream, stream_synchronize


@pytest.mark.parametrize(
    ("non_blocking", "flags"),
    [
        # cudaStreamDefault, cudaStreamNonBlocking
        pytest.param(False, 0, id="blocking"),
        pytest.param(True, 1, id="non-blocking"),
    ],
)
def test_create_stream_type(non_blocking, flags) -> None:
    """create_stream() returns a cudaStream_t with the requested blocking flag."""
    stream = create_stream(non_blocking=non_blocking)
    assert isinstance(stream, cudart.cudaStream_t)
    assert int(cuda_call(cudart.cudaStreamGetFlags(stream))) == flags
    destroy_stream(stream)


def test_stream_synchronize() -> None:
    """stream_synchronize() completes without error on a valid stream."""
    stream = create_stream()
    stream_synchronize(stream)
    destroy_stream(stream)


def test_multiple_streams() -> None:
    """Multiple streams can be created and destroyed independently."""
    streams = [create_stream() for _ in range(3)]
    for s in streams:
        destroy_stream(s)
