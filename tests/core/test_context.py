# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_context.py -- CUDA context lifecycle."""

from __future__ import annotations

import pytest

from trtutils.compat._libs import cuda
from trtutils.core._context import create_context, destroy_context


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(None, id="default_device"),
        pytest.param(0, id="explicit_device_0"),
    ],
)
def test_create_context_with_device(device: int | None) -> None:
    """create_context() returns a valid CUcontext for the given device arg."""
    ctx = create_context() if device is None else create_context(device=device)
    assert ctx is not None
    destroy_context(ctx)


def test_create_context_returns_context_type() -> None:
    """create_context() return type should be cuda.CUcontext."""
    ctx = create_context()
    assert isinstance(ctx, cuda.CUcontext)
    destroy_context(ctx)
