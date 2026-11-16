# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule containing tools for inspecting TensorRT engines.

Classes
-------
:class:`EngineLayer`
    A layer of a built engine, as the TensorRT engine inspector reports it.
:class:`LayerInfo`
    Detailed information about a single layer in a TensorRT network.
:class:`PlacementReport`
    Where the layers of a built engine run, on a DLA or on the GPU.

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
:func:`inspect_placement`
    Find where each layer of a built engine runs: on a DLA or on the GPU.

"""

from ._inspect import inspect_engine
from ._names import get_engine_names
from ._onnx import inspect_onnx_layers
from ._placement import EngineLayer, PlacementReport, inspect_placement
from ._tensor import get_tensor_size
from ._types import LayerInfo

__all__ = [
    "EngineLayer",
    "LayerInfo",
    "PlacementReport",
    "get_engine_names",
    "get_tensor_size",
    "inspect_engine",
    "inspect_onnx_layers",
    "inspect_placement",
]
