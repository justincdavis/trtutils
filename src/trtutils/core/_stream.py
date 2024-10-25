# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib

# suppress pycuda import error for docs build
with contextlib.suppress(Exception):
    from cuda import cudart  # type: ignore[import-untyped, import-not-found]

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


def stream_synchronize(stream: cudart.cudaStream_t) -> None:
    """
    Copy a numpy array to a device pointer with error checking.

    Parameters
    ----------
    stream : cudart.cudaStream_t
        The stream to synchronize calls for.

    """
    cuda_call(cudart.cudaStreamSynchronize(stream))
