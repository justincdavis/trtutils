# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_stream.py -- CUDA stream lifecycle."""

from __future__ import annotations

import pytest


@pytest.mark.gpu
class TestCreateStream:
    """Tests for create_stream()."""

    def test_create_stream_returns_value(self) -> None:
        """create_stream() returns a non-None stream object."""
        from trtutils.core._stream import create_stream, destroy_stream

        stream = create_stream()
        assert stream is not None
        destroy_stream(stream)

    def test_create_stream_type(self) -> None:
        """create_stream() returns a cudaStream_t."""
        from trtutils.compat._libs import cudart
        from trtutils.core._stream import create_stream, destroy_stream

        stream = create_stream()
        assert isinstance(stream, cudart.cudaStream_t)
        destroy_stream(stream)


@pytest.mark.gpu
class TestDestroyStream:
    """Tests for destroy_stream()."""

    def test_destroy_stream_valid(self) -> None:
        """destroy_stream() on a valid stream should not raise."""
        from trtutils.core._stream import create_stream, destroy_stream

        stream = create_stream()
        destroy_stream(stream)  # Should not raise


@pytest.mark.gpu
class TestStreamSynchronize:
    """Tests for stream_synchronize()."""

    def test_stream_synchronize(self) -> None:
        """stream_synchronize() on a valid stream should not raise."""
        from trtutils.core._stream import (
            create_stream,
            destroy_stream,
            stream_synchronize,
        )

        stream = create_stream()
        stream_synchronize(stream)  # Should not raise
        destroy_stream(stream)


@pytest.mark.gpu
class TestStreamLifecycle:
    """Integration tests for full stream lifecycle."""

    def test_create_sync_destroy(self) -> None:
        """Full lifecycle: create, synchronize, destroy."""
        from trtutils.core._stream import (
            create_stream,
            destroy_stream,
            stream_synchronize,
        )

        stream = create_stream()
        stream_synchronize(stream)
        destroy_stream(stream)

    def test_multiple_streams(self) -> None:
        """Multiple streams can be created and destroyed independently."""
        from trtutils.core._stream import create_stream, destroy_stream

        streams = [create_stream() for _ in range(3)]
        assert len(streams) == 3
        for s in streams:
            assert s is not None
        for s in streams:
            destroy_stream(s)
