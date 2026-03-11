# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_cuda.py -- error checking and cuda_call."""

from __future__ import annotations

import pytest

from trtutils.compat._libs import cuda, cudart
from trtutils.core._cuda import check_cuda_err, cuda_call


@pytest.mark.parametrize(
    "err_code",
    [
        pytest.param(cudart.cudaError_t.cudaSuccess, id="cudaError"),
        pytest.param(cuda.CUresult.CUDA_SUCCESS, id="CUresult"),
    ],
)
def test_check_cuda_err_success(err_code: object) -> None:
    """Success codes do not raise."""
    check_cuda_err(err_code)


@pytest.mark.parametrize(
    ("err_code", "match"),
    [
        pytest.param(
            cudart.cudaError_t.cudaErrorMemoryAllocation,
            "Cuda Runtime Error",
            id="cudaError",
        ),
        pytest.param(
            cuda.CUresult.CUDA_ERROR_INVALID_VALUE,
            "Cuda Error",
            id="CUresult",
        ),
    ],
)
def test_check_cuda_err_failure(err_code: object, match: str) -> None:
    """Non-success error codes raise RuntimeError."""
    with pytest.raises(RuntimeError, match=match):
        check_cuda_err(err_code)


def test_check_cuda_err_unknown_type() -> None:
    """Unknown error type raises RuntimeError."""
    with pytest.raises(RuntimeError, match="Unknown error type"):
        check_cuda_err("not_an_enum")


def test_cuda_call() -> None:
    """cuda_call returns values on success, raises on error."""
    # success path
    cuda_call(cuda.cuInit(0))
    count = cuda_call(cuda.cuDeviceGetCount())
    assert isinstance(count, int)
    assert count >= 1
    # single value
    device = cuda_call(cuda.cuDeviceGet(0))
    assert device is not None
    # error path
    with pytest.raises(RuntimeError):
        cuda_call(cuda.cuCtxDestroy(None))
