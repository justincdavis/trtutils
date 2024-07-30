# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Alternative backends for TRTEngine.

Warning:
-------
The alternative implementations for backends may not match the TRTEngineInterface
defined in the core submodule. As such, they may not be directly interchangeable
with the core implementation. Work is ongoing to make the backends more consistent
with the core implementation. Use with caution.

Classes
-------
PyCudaTRTEngine
    A class for creating TensorRT engines from serialized engine files.
    Memory is managed with PyCUDA.

"""

from __future__ import annotations

import contextlib

__all__ = []

with contextlib.suppress(ImportError):
    from ._pycuda import PyCudaTRTEngine

    __all__ += ["PyCudaTRTEngine"]
