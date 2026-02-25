"""Tests for src/trtutils/core/_context.py -- CUDA context lifecycle."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.gpu
class TestCreateContext:
    """Tests for create_context()."""

    def test_create_context_default_device(self) -> None:
        """create_context() returns a valid CUcontext for device 0."""
        from trtutils.core._context import create_context, destroy_context

        ctx = create_context()
        assert ctx is not None
        destroy_context(ctx)

    def test_create_context_explicit_device_0(self) -> None:
        """create_context(0) returns a valid CUcontext."""
        from trtutils.core._context import create_context, destroy_context

        ctx = create_context(device=0)
        assert ctx is not None
        destroy_context(ctx)

    def test_create_context_returns_context_type(self) -> None:
        """create_context() return type should be cuda.CUcontext."""
        from trtutils.compat._libs import cuda
        from trtutils.core._context import create_context, destroy_context

        ctx = create_context()
        assert isinstance(ctx, cuda.CUcontext)
        destroy_context(ctx)

    def test_create_context_uses_ctx_create_params_placeholder(self) -> None:
        """create_context() passes None as the first cuCtxCreate argument."""
        from trtutils.compat._libs import cuda
        from trtutils.core import _context

        fake_device = object()
        fake_context = object()
        with patch.object(
            _context.cuda,
            "cuDeviceGet",
            return_value=(cuda.CUresult.CUDA_SUCCESS, fake_device),
        ) as device_get, patch.object(
            _context.cuda,
            "cuCtxCreate",
            return_value=(cuda.CUresult.CUDA_SUCCESS, fake_context),
        ) as ctx_create:
            ctx = _context.create_context()

        assert ctx is fake_context
        device_get.assert_called_once_with(0)
        ctx_create.assert_called_once_with(None, 0, fake_device)


@pytest.mark.gpu
class TestDestroyContext:
    """Tests for destroy_context()."""

    def test_destroy_context_valid(self) -> None:
        """destroy_context() on a valid context should not raise."""
        from trtutils.core._context import create_context, destroy_context

        ctx = create_context()
        destroy_context(ctx)  # Should not raise

    def test_create_destroy_lifecycle(self) -> None:
        """Full lifecycle: create then destroy without error."""
        from trtutils.core._context import create_context, destroy_context

        ctx = create_context()
        assert ctx is not None
        destroy_context(ctx)
