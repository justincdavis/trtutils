"""Tests for src/trtutils/core/_context.py -- CUDA context lifecycle."""

from __future__ import annotations

import pytest


@pytest.mark.gpu
class TestCreateContext:
    """Tests for create_context()."""

    def test_create_context_default_device(self):
        """create_context() returns a valid CUcontext for device 0."""
        from trtutils.core._context import create_context, destroy_context

        ctx = create_context()
        assert ctx is not None
        destroy_context(ctx)

    def test_create_context_explicit_device_0(self):
        """create_context(0) returns a valid CUcontext."""
        from trtutils.core._context import create_context, destroy_context

        ctx = create_context(device=0)
        assert ctx is not None
        destroy_context(ctx)

    def test_create_context_returns_context_type(self):
        """create_context() return type should be cuda.CUcontext."""
        from trtutils.compat._libs import cuda
        from trtutils.core._context import create_context, destroy_context

        ctx = create_context()
        assert isinstance(ctx, cuda.CUcontext)
        destroy_context(ctx)


@pytest.mark.gpu
class TestDestroyContext:
    """Tests for destroy_context()."""

    def test_destroy_context_valid(self):
        """destroy_context() on a valid context should not raise."""
        from trtutils.core._context import create_context, destroy_context

        ctx = create_context()
        destroy_context(ctx)  # Should not raise

    def test_create_destroy_lifecycle(self):
        """Full lifecycle: create then destroy without error."""
        from trtutils.core._context import create_context, destroy_context

        ctx = create_context()
        assert ctx is not None
        destroy_context(ctx)
