# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule containing tools for building TensorRT engines.

Functions
---------
:func:`build_engine`
    Build a TensorRT engine from an ONNX file.

"""

from __future__ import annotations

from ._build import build_engine

__all__ = [
    "build_engine",
]
