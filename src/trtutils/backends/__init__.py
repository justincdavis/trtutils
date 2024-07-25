# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Backend implementations for TRTEngine.

This module will automatically assign a backend based on the available
backends. If no backends are available, an ImportError will be raised.
The available backends are:

        - cuda
        - pycuda

The backends will be utilized in the order they are listed above.
As such, if the CUDA backend can be loaded then the TRTEngine
implemented in the cuda submodule will be used. If the CUDA backend
cannot be loaded, then the next backend will be attempted.

Submodules
----------
cuda
    The CUDA backend for TRTEngine.
pycuda
    The PyCUDA backend for TRTEngine.

Classes
-------
TRTEngine
    A class for running inference on a TensorRT engine.
TRTEngineInterface
    An interface for the TRTEngine class.

"""

from __future__ import annotations

import contextlib

from . import cuda, pycuda
from ._interface import TRTEngineInterface

__all__ = ["TRTEngineInterface", "cuda", "pycuda"]
_start_len = len(__all__)

for _backend in [cuda, pycuda]:
    with contextlib.suppress(AttributeError):
        TRTEngine: TRTEngineInterface = _backend.TRTEngine
        __all__ += ["TRTEngine"]
        break

if len(__all__) == _start_len:
    _backends = ["cuda-python", "pycuda"]
    err_msg = (
        f"No backend found. Please install one of the following backends: {_backends}"
    )
    raise ImportError(err_msg)
