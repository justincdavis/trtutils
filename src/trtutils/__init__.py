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
core
    A module for the core functionality of the package.
jetson
    A module implementating additional functionality for Jetson devices.
impls
    A module containing implementations for different neural networks.
trtexec
    A module for utilities related to the trtexec tool.

Classes
-------
BenchmarkResult
    A dataclass for storing profiling information from benchmarking engines.
Metric
    A dataclass storing specific metric information from benchmarking.
TRTEngine
    A class for creating TensorRT engines from serialized engine files.
TRTModel
    A class for running inference on TensorRT engines.
ParallelTRTEngines
    A class for running many TRTEngines in parallel.
ParallelTRTModels
    A class for running many TRTModels in parallel.
QueuedTRTEngine
    A class for running a TRTEngine in a seperate thread asynchronously.
QueuedTRTModel
    A class for running a TRTModel in a seperate thread asynchronously.

Functions
---------
benchmark_engine
    Benchmark a TensorRT engine.
find_trtexec
    Find an instance of the trtexec binary on the system.
run_trtexec
    Run a command with trtexec.
set_log_level
    Set the log level of the trtutils package.

"""

from __future__ import annotations

# setup the logger before importing anything else
import logging
import os
import sys


# Created from answer by Dennis at:
# https://stackoverflow.com/questions/7621897/python-logging-module-globally
def _setup_logger(level: str | None = None) -> None:
    if level is not None:
        level = level.upper()
    level_map: dict[str | None, int] = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        None: logging.WARNING,
    }
    try:
        log_level = level_map[level]
    except KeyError:
        log_level = logging.WARNING

    # create logger
    logger = logging.getLogger(__package__)
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)


def set_log_level(level: str) -> None:
    """
    Set the log level for the trtutils package.

    Parameters
    ----------
    level : str
        The log level to set. One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".

    Raises
    ------
    ValueError
        If the level is not one of the allowed values.

    """
    if level.upper() not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        err_msg = f"Invalid log level: {level}"
        raise ValueError(err_msg)
    _setup_logger(level)


level = os.getenv("TRTUTILS_LOG_LEVEL")
_setup_logger(level)
_log = logging.getLogger(__name__)
if level is not None and level.upper() not in [
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
]:
    _log.warning(f"Invalid log level: {level}. Using default log level: WARNING")

__author__ = "Justin Davis"
__version__ = "0.3.0"

import contextlib

from . import core, impls, trtexec
from ._benchmark import BenchmarkResult, Metric, benchmark_engine
from ._engine import ParallelTRTEngines, QueuedTRTEngine, TRTEngine
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


# start CUDA
with contextlib.suppress(ImportError):
    from cuda import cuda  # type: ignore[import-untyped, import-not-found]

    core.cuda_call(cuda.cuInit(0))

    device_count = core.cuda_call(cuda.cuDeviceGetCount())
    _log.info(f"Number of CUDA devices: {device_count}")
