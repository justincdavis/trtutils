# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule containing tools for profiling TensorRT engines.

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
:func:`build_fused_layer_map`
    Map individual ONNX layer names to their fused TRT layer.
:func:`identify_quantize_speedups_by_layer`
    Identify which layers benefit most from INT8 quantization.
:func:`profile_engine`
    Profile a TensorRT engine layer-by-layer.
:func:`resolve_fused_layer_value`
    Look up a per-layer metric value, handling TensorRT layer fusion.

"""

from __future__ import annotations

from ._fusion import build_fused_layer_map, resolve_fused_layer_value
from ._optimize import identify_quantize_speedups_by_layer
from ._profiler import LayerProfiler, LayerTiming, ProfilerResult, profile_engine

__all__ = [
    "LayerProfiler",
    "LayerTiming",
    "ProfilerResult",
    "build_fused_layer_map",
    "identify_quantize_speedups_by_layer",
    "profile_engine",
    "resolve_fused_layer_value",
]
