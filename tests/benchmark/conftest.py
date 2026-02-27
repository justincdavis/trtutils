# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Benchmark test fixtures -- built engines and iteration constants."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import Callable

# tests/benchmark/conftest.py -> tests/benchmark -> tests -> project_root -> data/
ONNX_PATH = Path(__file__).parent.parent.parent / "data" / "simple.onnx"

# Benchmark iteration constants (kept small for fast test runs)
ITERS = 10
WARMUP_ITERS = 2


@pytest.fixture(scope="session")
def engine_path(build_test_engine: Callable[..., Path]) -> Path:
    """Session-scoped built engine for benchmark tests."""
    return build_test_engine(ONNX_PATH)
