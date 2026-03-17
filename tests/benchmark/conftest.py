# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Benchmark test fixtures -- built engines and iteration constants."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

# Benchmark iteration constants (kept small for fast test runs)
ITERS = 10
WARMUP_ITERS = 2


@pytest.fixture(scope="session")
def engine_path(simple_engine_path) -> Path:
    """Session-scoped built engine for benchmark tests (delegates to global fixture)."""
    return simple_engine_path
