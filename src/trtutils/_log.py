# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
import logging
import os
import sys
from typing import TYPE_CHECKING

with contextlib.suppress(ImportError):
    import tensorrt as trt

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import Self


_LEVEL_MAP: dict[str | None, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    None: logging.WARNING,
}


def _setup_logger(level: str | None = None) -> None:
    if level is not None:
        level = level.upper()

    try:
        log_level = _LEVEL_MAP[level]
    except KeyError:
        log_level = logging.WARNING

    # create logger
    logger = logging.getLogger(__package__)
    logger.setLevel(log_level)

    has_handler = False
    if len(logger.handlers) > 0:
        has_handler = True

    if not has_handler:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(log_level)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
    else:
        logger.handlers[0].setLevel(log_level)

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
class TRTLogger(trt.ILogger):  # type: ignore[misc]
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
        self._level = self._logger.getEffectiveLevel()

    @property
    def logger(self: Self) -> logging.Logger:
        """
        Get the internal Python logger.

        Returns
        -------
        logging.Logger
            The internal Python logger instance.

        """
        return self._logger

    @property
    def level(self: Self) -> int:
        """
        Get the current log level.

        Returns
        -------
        int
            The current log level of the logger.

        """
        return self._level

    class _LogLevelContext:
        def __init__(self: Self, logger: TRTLogger, level: str | None) -> None:
            self._logger = logger
            self._level = level
            self._old_level = logger.logger.getEffectiveLevel()

        def __enter__(self: Self) -> TRTLogger:
            if self._level:
                lvl = self._level.upper()
                log_level = _LEVEL_MAP.get(lvl)
                if log_level is not None:
                    self._logger.logger.setLevel(log_level)
            return self._logger

        def __exit__(
            self: Self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            self._logger.logger.setLevel(self._old_level)

    def with_level(self: Self, level: str | None) -> _LogLevelContext:
        """
        Create a context manager to temporarily set the log level.

        Parameters
        ----------
        level : str | None
            The log level to set. One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".

        Returns
        -------
        _LogLevelContext
            A context manager that sets the log level for the duration of the block.

        """
        return self._LogLevelContext(self, level)

    def suppress(self: Self) -> _LogLevelContext:
        """
        Suppress all log messages.

        This method sets the logger's level to CRITICAL, effectively silencing all log messages.

        Returns
        -------
        _LogLevelContext
            A context manager that suppresses all log messages.

        """
        return self._LogLevelContext(self, "CRITICAL")

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
