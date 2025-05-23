# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib

with contextlib.suppress(Exception):
    try:
        import cuda.bindings.driver as cuda
    except (ImportError, ModuleNotFoundError):
        from cuda import cuda

from ._cuda import cuda_call


def create_context(device: int = 0) -> cuda.CUcontext:
    """
    Create a CUDA context.

    Parameters
    ----------
    device : int
        The device to make a context for. By default 0.

    Returns
    -------
    cuda.CUcontext
        The created CUDA context

    """
    cu_device = cuda_call(cuda.cuDeviceGet(device))
    return cuda_call(cuda.cuCtxCreate(0, cu_device))


def destroy_context(context: cuda.CUcontext) -> None:
    """
    Destory a CUDA context.

    Parameters
    ----------
    context : cuda.CUcontext
        The CUDA context to destroy.

    """
    cuda_call(cuda.cuCtxDestroy(context))
