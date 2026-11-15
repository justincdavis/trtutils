# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/_log.py -- capturing TensorRT errors."""

from __future__ import annotations

import threading

import pytest

from trtutils._log import LOG
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
