# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule containing tools for inspecting TensorRT engines.

Functions
---------
:func:`inspect_engine`
    Inspect a TensorRT engine.

"""

from ._inspect import inspect_engine

__all__ = ["inspect_engine"]
