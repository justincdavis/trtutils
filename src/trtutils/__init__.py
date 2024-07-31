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
backends
    A module providing alternative backends for the TRTEngine class.
core
    A module for the core functionality of the package.
trtexec
    A module for utilities related to the trtexec tool.

Classes
-------
TRTEngine
    A class for creating TensorRT engines from serialized engine files.
TRTModel
    A class for running inference on TensorRT engines.

Functions
---------
find_trtexec
    Find an instance of the trtexec binary on the system.
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
__version__ = "0.1.1"


from . import backends, core, trtexec
from ._engine import TRTEngine
from ._model import TRTModel
from .trtexec import find_trtexec, run_trtexec

__all__ = [
    "TRTEngine",
    "TRTModel",
    "backends",
    "core",
    "find_trtexec",
    "run_trtexec",
    "set_log_level",
    "trtexec",
]
