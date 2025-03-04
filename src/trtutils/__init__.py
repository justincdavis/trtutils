# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: E402, F401
"""
A package for enabling high-level usage of TensorRT in Python.

This package provides a high-level interface for using TensorRT in Python. It
provides a class for creating TensorRT engines from serialized engine files,
a class for running inference on those engines, and a variety of other utilities.

Submodules
----------
:mod:`core`
    A module for the core functionality of the package.
:mod:`jetson`
    A module implementating additional functionality for Jetson devices.
:mod:`impls`
    A module containing implementations for different neural networks.
:mod:`trtexec`
    A module for utilities related to the trtexec tool.

Classes
-------
:class:`BenchmarkResult`
    A dataclass for storing profiling information from benchmarking engines.
:class:`Metric`
    A dataclass storing specific metric information from benchmarking.
:class:`TRTEngine`
    A class for creating TensorRT engines from serialized engine files.
:class:`TRTModel`
    A class for running inference on TensorRT engines.
:class:`ParallelTRTEngines`
    A class for running many TRTEngines in parallel.
:class:`ParallelTRTModels`
    A class for running many TRTModels in parallel.
:class:`QueuedTRTEngine`
    A class for running a TRTEngine in a seperate thread asynchronously.
:class:`QueuedTRTModel`
    A class for running a TRTModel in a seperate thread asynchronously.

Functions
---------
:func:`benchmark_engine`
    Benchmark a TensorRT engine.
:func:`benchmark_engines`
    Benchmark TensorRT engines in parallel or serially.
:func:`find_trtexec`
    Find an instance of the trtexec binary on the system.
:func:`run_trtexec`
    Run a command with trtexec.
:func:`set_log_level`
    Set the log level of the trtutils package.

"""

from __future__ import annotations

__author__ = "Justin Davis"
__version__ = "0.4.1"

import contextlib

from . import core, impls, trtexec
from ._benchmark import BenchmarkResult, Metric, benchmark_engine, benchmark_engines
from ._engine import ParallelTRTEngines, QueuedTRTEngine, TRTEngine
from ._log import set_log_level
from ._model import ParallelTRTModels, QueuedTRTModel, TRTModel
from .trtexec import find_trtexec, run_trtexec

__all__ = [
    "BenchmarkResult",
    "Metric",
    "ParallelTRTEngines",
    "ParallelTRTModels",
    "QueuedTRTEngine",
    "QueuedTRTModel",
    "TRTEngine",
    "TRTModel",
    "benchmark_engine",
    "benchmark_engines",
    "core",
    "find_trtexec",
    "impls",
    "run_trtexec",
    "set_log_level",
    "trtexec",
]

# attempt jetson module import
with contextlib.suppress(ImportError):
    from . import jetson

    __all__ += ["jetson"]
