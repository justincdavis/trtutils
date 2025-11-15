# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule containing tools for inspecting TensorRT engines.

Functions
---------
:func:`get_engine_names`
    Get the input/output names of a TensorRT engine in order.
:func:`inspect_engine`
    Inspect a TensorRT engine.
:func:`inspect_onnx_layers`
    Inspect the layers of an ONNX model.

"""

from ._inspect import inspect_engine
from ._names import get_engine_names
from ._onnx import inspect_onnx_layers

__all__ = ["get_engine_names", "inspect_engine", "inspect_onnx_layers"]
