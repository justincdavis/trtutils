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


@pytest.fixture
def _restore_log_state():
    level = LOG.logger.level
    yield
    LOG.logger.setLevel(level)


@pytest.mark.cpu
@pytest.mark.usefixtures("_restore_log_state")
def test_with_level_restores_level_and_emits() -> None:
    """with_level changes only the logger level, restores it, and leaves handlers alone."""
    set_log_level("WARNING")
    assert LOG.level == logging.WARNING
    before = [(handler, handler.level) for handler in LOG.logger.handlers]
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    capture = _Capture(level=logging.ERROR)
    LOG.logger.addHandler(capture)
    try:
        with LOG.with_level("DEBUG"):
            assert LOG.level == logging.DEBUG
            LOG.error("inside")
        LOG.error("outside")
    finally:
        LOG.logger.removeHandler(capture)

    assert [record.getMessage() for record in records] == ["inside", "outside"]
    assert LOG.level == logging.WARNING
    assert [(handler, handler.level) for handler in LOG.logger.handlers] == before

    set_log_level("ERROR")
    assert LOG.level == logging.ERROR


@pytest.mark.cpu
@pytest.mark.usefixtures("_restore_log_state")
def test_reused_context_nests() -> None:
    """One context instance entered nested restores each prior level in turn."""
    set_log_level("WARNING")
    quiet = LOG.suppress()
    with quiet:
        assert LOG.level == logging.CRITICAL
        with quiet:
            assert LOG.level == logging.CRITICAL
        assert LOG.level == logging.CRITICAL
    assert LOG.level == logging.WARNING
