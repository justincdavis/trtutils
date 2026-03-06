# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_context.py -- CUDA context lifecycle."""

from __future__ import annotations

from unittest.mock import ANY, patch

import pytest

from trtutils._flags import FLAGS
from trtutils.compat._libs import cuda
from trtutils.core import _context
from trtutils.core._context import create_context, destroy_context


class TestCreateContext:
    """Tests for create_context()."""

    @pytest.mark.parametrize(
        "device",
        [
            pytest.param(None, id="default_device"),
            pytest.param(0, id="explicit_device_0"),
        ],
    )
    def test_create_context_with_device(self, device: int | None) -> None:
        """create_context() returns a valid CUcontext for the given device arg."""
        ctx = create_context() if device is None else create_context(device=device)
        assert ctx is not None
        destroy_context(ctx)

    def test_create_context_returns_context_type(self) -> None:
        """create_context() return type should be cuda.CUcontext."""
        ctx = create_context()
        assert isinstance(ctx, cuda.CUcontext)
        destroy_context(ctx)

    def test_create_context_calls_cu_ctx_create(self) -> None:
        """create_context() calls cuCtxCreate with the correct arguments."""
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
        if FLAGS.CUDA_PYTHON_13:
            # cuda-python 13+: cuCtxCreate(CUctxCreateParams(), flags, device)
            ctx_create.assert_called_once_with(ANY, 0, fake_device)
        else:
            # cuda-python 11/12: cuCtxCreate(flags, device)
            ctx_create.assert_called_once_with(0, fake_device)


class TestDestroyContext:
    """Tests for destroy_context()."""

    def test_create_destroy_lifecycle(self) -> None:
        """Full lifecycle: create then destroy without error."""
        ctx = create_context()
        assert ctx is not None
        destroy_context(ctx)
