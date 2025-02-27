# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
A submodule implementing additional tools for Jetson devices.

Classes
-------
:class:`JetsonBenchmarkResult`
    The results of benchmarking a TRTEngine on a Jetson device.

Functions
---------
:func:`benchmark_engine`
    A mirror of trtutils.benchmark_engine, but also measures energy usage.
:func:`benchmark_engines`
    A mirror of trtutils.benchmark_engines, but also measures energy usage.

"""

from __future__ import annotations

from ._benchmark import JetsonBenchmarkResult, benchmark_engine, benchmark_engines

__all__ = [
    "JetsonBenchmarkResult",
    "benchmark_engine",
    "benchmark_engines",
]
