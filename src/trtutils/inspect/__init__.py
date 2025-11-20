# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule containing tools for inspecting TensorRT engines.

Classes
-------
:class:`LayerTiming`
    A dataclass to store per-layer profiling statistics.
:class:`ProfilerResult`
    A dataclass to store the complete profiling results.
:class:`LayerProfiler`
    A class that implements TensorRT's IProfiler interface.

Functions
---------
:func:`get_engine_names`
    Get the input/output names of a TensorRT engine in order.
:func:`inspect_engine`
    Inspect a TensorRT engine.
:func:`inspect_onnx_layers`
    Inspect the layers of an ONNX model.
:func:`profile_engine`
    Profile a TensorRT engine layer-by-layer.

"""

from ._inspect import inspect_engine
from ._names import get_engine_names
from ._onnx import inspect_onnx_layers
from ._profiler import LayerTiming, ProfilerResult, LayerProfiler, profile_engine

__all__ = [
    "LayerTiming",
    "ProfilerResult",
    "LayerProfiler",
    "get_engine_names",
    "inspect_engine",
    "inspect_onnx_layers",
    "profile_engine",
]
