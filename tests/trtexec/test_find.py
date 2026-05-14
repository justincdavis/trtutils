# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/trtexec/_find.py -- find_trtexec."""

from __future__ import annotations

import pytest

from trtutils.trtexec import find_trtexec


@pytest.mark.jetson
def test_find_trtexec_returns_existing_path() -> None:
    """find_trtexec returns an existing trtexec binary on Jetson."""
    path = find_trtexec()
    assert path.exists()
    assert path.name == "trtexec"
