# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib

with contextlib.suppress(Exception):
    try:
        import cuda.bindings.runtime as cudart
    except (ImportError, ModuleNotFoundError):
        from cuda import cudart

from ._cuda import cuda_call


def create_stream() -> cudart.cudaStream_t:
    """
    Create a CUDA Stream.

    Returns
    -------
    cudart.cudaStream_t
        The CUDA stream.

    """
    return cuda_call(cudart.cudaStreamCreate())


def destroy_stream(stream: cudart.cudaStream_t) -> None:
    """
    Destroy a CUDA Stream.

    Parameters
    ----------
    stream : cudart.cudaStream_t
        The CUDA stream to destroy.

    """
    cuda_call(cudart.cudaStreamDestroy(stream))


def stream_synchronize(stream: cudart.cudaStream_t) -> None:
    """
    Copy a numpy array to a device pointer with error checking.

    Parameters
    ----------
    stream : cudart.cudaStream_t
        The stream to synchronize calls for.

    """
    cuda_call(cudart.cudaStreamSynchronize(stream))
