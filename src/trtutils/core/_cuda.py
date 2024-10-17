# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: TRY004
from __future__ import annotations

import contextlib
from typing import TypeVar

# suppress pycuda import error for docs build
with contextlib.suppress(Exception):
    from cuda import cuda, cudart  # type: ignore[import-untyped, import-not-found]


def check_cuda_err(err: cuda.CUresult | cudart.cudaError_t) -> None:
    """
    Check if a CUDA error occurred and raise an exception if so.

    Parameters
    ----------
    err : cuda.CUresult | cudart.cudaError_t
        The CUDA error to check.

    Raises
    ------
    RuntimeError
        If a CUDA or CUDA Runtime error occurred.

    """
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            err_msg = f"Cuda Error: {err}"
            raise RuntimeError(err_msg)
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            err_msg = f"Cuda Runtime Error: {err}"
            raise RuntimeError(err_msg)
    else:
        err_msg = f"Unknown error type: {err}"
        raise RuntimeError(err_msg)


T = TypeVar("T")


def cuda_call(call: tuple[cuda.CUresult | cudart.cudaError_t, T]) -> T:
    """
    Call a CUDA function and check for errors.

    Parameters
    ----------
    call : tuple[cuda.CUresult | cudart.cudaError_t, T]
        The CUDA function to call and its arguments.

    Returns
    -------
    T
        The result of the CUDA function call.

    """
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        return res[0]
    return res
