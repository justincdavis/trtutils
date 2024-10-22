# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
A submodule implementing additional tools for Jetson devices.

Classes
-------
JetsonBenchmarkResult
    The results of benchmarking a TRTEngine on a Jetson device.

Functions
---------
benchmark_engine
    A mirror of trtutils.benchmark_engine, but also measures energy usage.

"""

from __future__ import annotations

from ._benchmark import JetsonBenchmarkResult, benchmark_engine

__all__ = [
    "JetsonBenchmarkResult",
    "benchmark_engine",
]
