# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
import logging
import os
import sys
from typing import TYPE_CHECKING

with contextlib.suppress(ImportError):
    import tensorrt as trt  # type: ignore[import-untyped, import-not-found]

if TYPE_CHECKING:
    from typing_extensions import Self


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

    # if not logger.hasHandlers():
    existing_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.level == log_level:
            existing_handler = handler
            break

    if not existing_handler:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(log_level)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    logger.propagate = True


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


# create a TensorRT compatible logger
class TRTLogger(trt.ILogger):
    """
    Logger that implements TensorRT's ILogger interface while using Python's logging system.

    This class bridges TensorRT's logging system with Python's standard logging module,
    allowing TensorRT log messages to be handled by the Python logging framework.
    It also provides convenience methods that match Python's standard logging levels.

    Examples
    --------
    >>> logger = TRTLogger()
    >>> logger.info("Starting TensorRT engine build")
    >>> with trt.Builder(logger) as builder:
    ...     # TensorRT will use the logger for its messages
    ...     pass

    """

    def __init__(self: Self) -> None:
        """
        Initialize the TensorRT logger.

        Creates a logger that implements TensorRT's ILogger interface and
        delegates to a Python logging.Logger instance internally.
        """
        super().__init__()
        self._logger = logging.getLogger("trtutils")

    def log(self: Self, severity: trt.ILogger.Severity, msg: str) -> None:
        """
        Log a message with the specified severity.

        This method implements TensorRT's ILogger.log method and maps TensorRT
        severity levels to Python logging levels.

        Parameters
        ----------
        severity : trt.ILogger.Severity
            TensorRT-specific severity level of the message
        msg : str
            The log message to record

        """
        if severity == trt.ILogger.Severity.INFO:
            self._logger.info(msg)
        elif severity == trt.ILogger.Severity.WARNING:
            self._logger.warning(msg)
        elif severity == trt.ILogger.Severity.ERROR:
            self._logger.error(msg)
        else:
            self._logger.debug(msg)

    def debug(self: Self, msg: str) -> None:
        """
        Log a debug message.

        Parameters
        ----------
        msg : str
            The debug message to log

        """
        self._logger.debug(msg)

    def info(self: Self, msg: str) -> None:
        """
        Log an info message.

        Parameters
        ----------
        msg : str
            The info message to log

        """
        self._logger.info(msg)

    def warning(self: Self, msg: str) -> None:
        """
        Log a warning message.

        Parameters
        ----------
        msg : str
            The warning message to log

        """
        self._logger.warning(msg)

    def error(self: Self, msg: str) -> None:
        """
        Log an error message.

        Parameters
        ----------
        msg : str
            The error message to log

        """
        self._logger.error(msg)

    def critical(self: Self, msg: str) -> None:
        """
        Log a critical message.

        Parameters
        ----------
        msg : str
            The critical message to log

        """
        self._logger.critical(msg)


LOG = TRTLogger()
