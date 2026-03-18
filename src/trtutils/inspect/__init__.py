# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule containing tools for inspecting TensorRT engines.

Classes
-------
:class:`LayerInfo`
    Detailed information about a single layer in a TensorRT network.

Functions
---------
:func:`get_engine_names`
    Get the input/output names of a TensorRT engine in order.
:func:`get_tensor_size`
    Calculate the size of a TensorRT tensor in bytes.
:func:`inspect_engine`
    Inspect a TensorRT engine.
:func:`inspect_onnx_layers`
    Inspect the layers of an ONNX model.

"""

from ._inspect import inspect_engine
from ._names import get_engine_names
from ._onnx import inspect_onnx_layers
from ._tensor import get_tensor_size
from ._types import LayerInfo

__all__ = [
    "LayerInfo",
    "get_engine_names",
    "get_tensor_size",
    "inspect_engine",
    "inspect_onnx_layers",
]
