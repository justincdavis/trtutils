# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Jetson test fixtures -- built engines for Jetson benchmark/profiling tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import Callable


# tests/jetson/conftest.py -> tests/jetson -> tests -> project_root -> data/
ONNX_PATH = Path(__file__).parent.parent.parent / "data" / "simple.onnx"


@pytest.fixture(scope="session")
def engine_path(build_test_engine: Callable[..., Path]) -> Path:
    """Session-scoped built engine for Jetson tests."""
    return build_test_engine(ONNX_PATH)
