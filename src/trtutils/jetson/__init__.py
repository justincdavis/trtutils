# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
A submodule implementing additional tools for Jetson devices.

Classes
-------
:class:`JetsonBenchmarkResult`
    The results of benchmarking a TRTEngine on a Jetson device.
:class:`JetsonLayerTiming`
    Per-layer timing with power and energy metrics for Jetson profiling.
:class:`JetsonProfilerResult`
    The results of profiling a TRTEngine on a Jetson device.

Functions
---------
:func:`benchmark_engine`
    A mirror of trtutils.benchmark_engine, but also measures energy usage.
:func:`benchmark_engines`
    A mirror of trtutils.benchmark_engines, but also measures energy usage.
:func:`profile_engine`
    A mirror of trtutils.inspect.profile_engine, but also measures per-layer energy usage.

"""

from __future__ import annotations

from ._benchmark import JetsonBenchmarkResult, benchmark_engine, benchmark_engines
from ._profile import JetsonLayerTiming, JetsonProfilerResult, profile_engine

__all__ = [
    "JetsonBenchmarkResult",
    "JetsonLayerTiming",
    "JetsonProfilerResult",
    "benchmark_engine",
    "benchmark_engines",
    "profile_engine",
]
