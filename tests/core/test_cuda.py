# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_cuda.py -- error checking and cuda_call."""

from __future__ import annotations

import pytest

from trtutils.compat._libs import cuda, cudart
from trtutils.core._cuda import check_cuda_err, cuda_call, init_cuda


class TestCheckCudaErr:
    """Tests for check_cuda_err()."""

    @pytest.mark.parametrize(
        "err_code",
        [
            pytest.param(cudart.cudaError_t.cudaSuccess, id="cudaError"),
            pytest.param(cuda.CUresult.CUDA_SUCCESS, id="CUresult"),
        ],
    )
    def test_success(self, err_code: object) -> None:
        """Success codes should not raise."""
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
    def test_failure(self, err_code: object, match: str) -> None:
        """Non-success error codes should raise RuntimeError."""
        with pytest.raises(RuntimeError, match=match):
            check_cuda_err(err_code)

    def test_unknown_type_raises(self) -> None:
        """Unknown error type should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="Unknown error type"):
            check_cuda_err("not_an_enum")


class TestCudaCall:
    """Tests for cuda_call()."""

    def test_tuple_with_success_single_value(self) -> None:
        """cuda_call with (success, value) returns value."""
        cuda_call(cuda.cuInit(0))
        count = cuda_call(cuda.cuDeviceGetCount())
        assert isinstance(count, int)
        assert count >= 1

    def test_tuple_with_error_raises(self) -> None:
        """cuda_call with an error result raises RuntimeError."""
        with pytest.raises(RuntimeError):
            cuda_call(cuda.cuCtxDestroy(None))

    def test_tuple_multiple_values(self) -> None:
        """cuda_call returning multiple values returns them as a tuple."""
        device = cuda_call(cuda.cuDeviceGet(0))
        assert device is not None


class TestInitCuda:
    """Tests for init_cuda()."""

    def test_init_succeeds_and_is_idempotent(self) -> None:
        """init_cuda() should complete without error and be safely callable twice."""
        init_cuda()
        init_cuda()
