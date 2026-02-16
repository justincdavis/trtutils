# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: E402
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
:mod:`compat`
    A module for compatibility with other libraries.
:mod:`core`
    A module for the core functionality of the package.
:mod:`download`
    A module for downloading and converting models to ONNX.
:mod:`jetson`
    A module implementating additional functionality for Jetson devices.
:mod:`image`
    A module for image processing with TensorRT.
:mod:`models`
    A module containing implementations of DNN models.
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
:func:`enable_nvtx`
    Enable trtutils NVTX profiling.
:func:`disable_nvtx`
    Disable trtutils NVTX profiling.

Objects
-------
:obj:`CONFIG`
    The config storage object for trtutils.
:obj:`FLAGS`
    The flag storage object for trtutils.
:obj:`LOG`
    The TensorRT compatible logger for trtutils.
:obj:`JIT`
    A context manager for enabling just-in-time compilation using Numba.
:obj:`NVTX`
    A context manager for enabling NVTX profiling.

"""

from __future__ import annotations

import importlib

# before handling anything check if tensorrt and cuda-python are available
not_found_modules = []
try:
    importlib.import_module("tensorrt")
except ModuleNotFoundError:
    not_found_modules.append("tensorrt")

try:
    importlib.import_module("cuda")
except ModuleNotFoundError:
    not_found_modules.append("cuda-python")


if len(not_found_modules) > 0:
    err_msg = "Could not find the following core modules: "
    err_msg += ", ".join(not_found_modules)
    err_msg += ", ensure you installed trtutils with CUDA."
    err_msg += " Available CUDA variants: cu11, cu12, cu13\n"
    err_msg += " Example: pip install trtutils[cu11]\n"
    err_msg += " Example: pip install trtutils[cu12]\n"
    err_msg += " Example: pip install trtutils[cu13]\n"
    raise ImportError(err_msg)


from ._config import CONFIG
from ._flags import FLAGS
from ._jit import JIT, disable_jit, enable_jit, register_jit
from ._log import LOG, set_log_level
from ._nvtx import NVTX, disable_nvtx, enable_nvtx

__author__ = "Justin Davis"
__version__ = "0.7.0"

import contextlib

from . import builder, compat, core, download, image, inspect, models, parallel, trtexec
from ._benchmark import BenchmarkResult, Metric, benchmark_engine, benchmark_engines
from ._engine import TRTEngine
from ._profile import profile_engine
from .builder import build_engine
from .core._device import Device, get_device, set_device
from .inspect import inspect_engine
from .trtexec import find_trtexec, run_trtexec

__all__ = [
    "CONFIG",
    "FLAGS",
    "JIT",
    "LOG",
    "NVTX",
    "BenchmarkResult",
    "Device",
    "Metric",
    "TRTEngine",
    "benchmark_engine",
    "benchmark_engines",
    "build_engine",
    "builder",
    "compat",
    "core",
    "disable_jit",
    "disable_nvtx",
    "download",
    "enable_jit",
    "enable_nvtx",
    "find_trtexec",
    "get_device",
    "image",
    "inspect",
    "inspect_engine",
    "models",
    "parallel",
    "profile_engine",
    "register_jit",
    "run_trtexec",
    "set_device",
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
