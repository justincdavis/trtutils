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

"""

from __future__ import annotations

import contextlib

__all__ = []

_possible_backends = ["cuda-python", "pycuda"]

with contextlib.suppress(ImportError):
    from . import cuda

    __all__ += ["cuda"]

with contextlib.suppress(ImportError):
    from . import pycuda

    __all__ += ["pycuda"]

if len(__all__) == 0:
    err_msg = f"No backend found. Please install one of the following backends: {_possible_backends}"
    raise ImportError(err_msg)

for backend in __all__:
    globals().update(vars()[backend].__dict__)
    __all__ += vars()[backend].__all__
    break
