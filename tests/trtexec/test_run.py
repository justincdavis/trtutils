# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/trtexec/_run.py -- run_trtexec."""

from __future__ import annotations

import pytest

from trtutils.trtexec import run_trtexec


@pytest.mark.jetson
def test_run_trtexec_help_succeeds() -> None:
    """run_trtexec('--help') exits successfully and returns string streams."""
    success, stdout, stderr = run_trtexec("--help")
    assert success is True
    assert isinstance(stdout, str)
    assert isinstance(stderr, str)


@pytest.mark.jetson
def test_run_trtexec_bad_arg_returns_failure() -> None:
    """run_trtexec returns (False, stdout, stderr) when trtexec rejects an arg."""
    success, stdout, stderr = run_trtexec("--definitely-not-a-valid-flag-xyz")
    assert success is False
    assert isinstance(stdout, str)
    assert isinstance(stderr, str)
