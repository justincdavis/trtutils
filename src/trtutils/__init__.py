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
:mod:`builder`
    A module for building TensorRT engines.
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
:func:`build_engine`
    Build a TensorRT engine.
:func:`find_trtexec`
    Find an instance of the trtexec binary on the system.
:func:`run_trtexec`
    Run a command with trtexec.
:func:`set_log_level`
    Set the log level of the trtutils package.

Objects
-------
:obj:`FLAGS`
    The flag storage object for trtutils.

"""

from __future__ import annotations

# setup the logger before importing anything else
import logging
import os
import sys

# import the flags object
from ._flags import FLAGS

__author__ = "Justin Davis"
__version__ = "0.4.1"

import contextlib

from . import builder, core, impls, trtexec
from ._benchmark import BenchmarkResult, Metric, benchmark_engine, benchmark_engines
from ._engine import ParallelTRTEngines, QueuedTRTEngine, TRTEngine
from ._log import set_log_level
from ._model import ParallelTRTModels, QueuedTRTModel, TRTModel
from .builder import build_engine
from .trtexec import find_trtexec, run_trtexec

__all__ = [
    "FLAGS",
    "BenchmarkResult",
    "Metric",
    "ParallelTRTEngines",
    "ParallelTRTModels",
    "QueuedTRTEngine",
    "QueuedTRTModel",
    "TRTEngine",
    "TRTModel",
    "benchmark_engine",
    "build_engine",
    "builder",
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

# output available execution api debug
for attr in [a for a in dir(FLAGS) if not a.startswith("_")]:
    _log.debug(f"FLAG {attr}: {getattr(FLAGS, attr)}")


# # start CUDA
# with contextlib.suppress(ImportError):
#     from cuda import cuda  # type: ignore[import-untyped, import-not-found]

#     core.cuda_call(cuda.cuInit(0))

#     device_count = core.cuda_call(cuda.cuDeviceGetCount())
#     _log.info(f"Number of CUDA devices: {device_count}")
