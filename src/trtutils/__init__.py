# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
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
:mod:`inspect`
    A module for inspecting TensorRT engines.
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
:func:`inspect_engine`
    Inspect a TensorRT engine.
:func:`run_trtexec`
    Run a command with trtexec.
:func:`set_log_level`
    Set the log level of the trtutils package.
:func:`enable_jit`
    Enable just-in-time compilation using Numba.
:func:`disable_jit`
    Disable just-in-time compilation using Numba.
:func:`register_jit`
    Decorator for registering functions for potential JIT compilation.

Objects
-------
:obj:`FLAGS`
    The flag storage object for trtutils.
:obj:`LOG`
    The TensorRT compatible logger for trtutils.
:obj:`JIT`
    A context manager for enabling just-in-time compilation using Numba.

"""

from __future__ import annotations

from ._config import CONFIG
from ._flags import FLAGS
from ._jit import JIT, disable_jit, enable_jit, register_jit
from ._log import LOG, set_log_level

__author__ = "Justin Davis"
__version__ = "0.6.0"

import contextlib

from . import builder, core, impls, inspect, trtexec
from ._benchmark import BenchmarkResult, Metric, benchmark_engine, benchmark_engines
from ._engine import ParallelTRTEngines, QueuedTRTEngine, TRTEngine
from ._model import ParallelTRTModels, QueuedTRTModel, TRTModel
from .builder import build_engine
from .inspect import inspect_engine
from .trtexec import find_trtexec, run_trtexec

__all__ = [
    "CONFIG",
    "FLAGS",
    "JIT",
    "LOG",
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
    "build_engine",
    "builder",
    "core",
    "disable_jit",
    "enable_jit",
    "find_trtexec",
    "impls",
    "inspect",
    "inspect_engine",
    "register_jit",
    "run_trtexec",
    "set_log_level",
    "trtexec",
]

# attempt jetson module import
with contextlib.suppress(ImportError):
    from . import jetson

    __all__ += ["jetson"]


# if numba is found, automatically enable the jit
if FLAGS.FOUND_NUMBA:
    LOG.info("Numba found, enabling JIT")
    enable_jit()

# output available execution api debug
for attr in [a for a in dir(FLAGS) if not a.startswith("_")]:
    _flag_str = f"FLAG - {attr}: {getattr(FLAGS, attr)}"
    LOG.debug(_flag_str)
