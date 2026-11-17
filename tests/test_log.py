# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/_log.py -- capturing TensorRT errors and temporary log levels."""

from __future__ import annotations

import logging
import threading

import pytest

from trtutils._log import LOG, set_log_level
from trtutils.compat._libs import trt


@pytest.mark.cpu
def test_capture_errors() -> None:
    """capture_errors collects errors logged on its thread inside the block, even when suppressed."""
    severity = trt.ILogger.Severity
    with LOG.suppress():
        LOG.log(severity.ERROR, "before")
        with LOG.capture_errors() as errors:
            LOG.log(severity.ERROR, "error")
            LOG.log(severity.INTERNAL_ERROR, "internal")
            LOG.log(severity.WARNING, "warning")
            LOG.log(severity.INFO, "info")
            other = threading.Thread(target=LOG.log, args=(severity.ERROR, "other thread"))
            other.start()
            other.join()
        LOG.log(severity.ERROR, "after")
    assert errors == ["error", "internal"]


def _handler_levels() -> list[int]:
    return [handler.level for handler in LOG.logger.handlers]


@pytest.fixture
def _restore_log_state():
    logger = LOG.logger
    level = logger.level
    handlers = [(handler, handler.level) for handler in logger.handlers]
    yield
    logger.setLevel(level)
    for handler, handler_level in handlers:
        handler.setLevel(handler_level)


@pytest.mark.cpu
@pytest.mark.usefixtures("_restore_log_state")
def test_with_level_restores_levels_and_emits() -> None:
    """with_level lowers the logger and handlers, restores both, and level tracks set_log_level."""
    set_log_level("WARNING")
    assert LOG.level == logging.WARNING
    before = _handler_levels()
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    capture = _Capture(level=logging.NOTSET)
    LOG.logger.addHandler(capture)
    try:
        with LOG.with_level("DEBUG"):
            assert LOG.level == logging.DEBUG
            assert all(level == logging.DEBUG for level in _handler_levels())
            LOG.debug("inside")
        LOG.debug("outside")
    finally:
        LOG.logger.removeHandler(capture)

    assert [record.getMessage() for record in records] == ["inside"]
    assert LOG.level == logging.WARNING
    assert _handler_levels() == before

    set_log_level("ERROR")
    assert LOG.level == logging.ERROR
