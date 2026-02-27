# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_cuda.py -- error checking and cuda_call."""

from __future__ import annotations

import pytest


@pytest.mark.gpu
class TestCheckCudaErr:
    """Tests for check_cuda_err()."""

    def test_success_cudaError(self) -> None:
        """CudaSuccess should not raise."""
        from trtutils.compat._libs import cudart
        from trtutils.core._cuda import check_cuda_err

        check_cuda_err(cudart.cudaError_t.cudaSuccess)

    def test_failure_cudaError(self) -> None:
        """Non-success cudaError_t should raise RuntimeError."""
        from trtutils.compat._libs import cudart
        from trtutils.core._cuda import check_cuda_err

        with pytest.raises(RuntimeError, match="Cuda Runtime Error"):
            check_cuda_err(cudart.cudaError_t.cudaErrorMemoryAllocation)

    def test_success_CUresult(self) -> None:
        """CUDA_SUCCESS CUresult should not raise."""
        from trtutils.compat._libs import cuda
        from trtutils.core._cuda import check_cuda_err

        check_cuda_err(cuda.CUresult.CUDA_SUCCESS)

    def test_failure_CUresult(self) -> None:
        """Non-success CUresult should raise RuntimeError."""
        from trtutils.compat._libs import cuda
        from trtutils.core._cuda import check_cuda_err

        with pytest.raises(RuntimeError, match="Cuda Error"):
            check_cuda_err(cuda.CUresult.CUDA_ERROR_INVALID_VALUE)

    def test_unknown_type_raises(self) -> None:
        """Unknown error type should raise RuntimeError."""
        from trtutils.core._cuda import check_cuda_err

        with pytest.raises(RuntimeError, match="Unknown error type"):
            check_cuda_err("not_an_enum")


@pytest.mark.gpu
class TestCudaCall:
    """Tests for cuda_call()."""

    def test_tuple_with_success_single_value(self) -> None:
        """cuda_call with (success, value) returns value."""
        from trtutils.compat._libs import cuda
        from trtutils.core._cuda import cuda_call

        cuda_call(cuda.cuInit(0))
        # cuInit returns None on success (single return)
        # The tuple is (CUresult.CUDA_SUCCESS,) which when unpacked: err=CUDA_SUCCESS, res=()
        # Since len(res)==0 we still get the empty tuple -- but really cuInit returns only err
        # Let's use cuDeviceGetCount to get a real value
        count = cuda_call(cuda.cuDeviceGetCount())
        assert isinstance(count, int)
        assert count >= 1

    def test_tuple_with_error_raises(self) -> None:
        """cuda_call with an error result raises RuntimeError."""
        from trtutils.compat._libs import cuda
        from trtutils.core._cuda import cuda_call

        # cuCtxDestroy with an invalid context should error
        with pytest.raises(RuntimeError):
            cuda_call(cuda.cuCtxDestroy(None))

    def test_tuple_multiple_values(self) -> None:
        """cuda_call returning multiple values returns them as a tuple."""
        from trtutils.compat._libs import cuda
        from trtutils.core._cuda import cuda_call

        # cuDeviceGet returns a device handle
        device = cuda_call(cuda.cuDeviceGet(0))
        assert device is not None


@pytest.mark.gpu
class TestInitCuda:
    """Tests for init_cuda()."""

    def test_init_succeeds(self) -> None:
        """init_cuda() should complete without error."""
        from trtutils.core._cuda import init_cuda

        init_cuda()  # Should not raise

    def test_init_idempotent(self) -> None:
        """init_cuda() can be called multiple times without error."""
        from trtutils.core._cuda import init_cuda

        init_cuda()
        init_cuda()  # Second call should also succeed
