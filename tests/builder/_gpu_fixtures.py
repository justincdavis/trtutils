# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""GPU-dependent builder fixtures, loaded by conftest.py when CUDA is available."""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

import pytest

from tests.conftest import DATA_DIR
from trtutils.builder._build import build_engine

if TYPE_CHECKING:
    from pathlib import Path

ONNX_PATH = DATA_DIR / "simple.onnx"


@pytest.fixture(scope="session")
def onnx_path() -> Path:
    """Path to the test ONNX model."""
    if not ONNX_PATH.exists():
        pytest.skip("Test ONNX model not found")
    return ONNX_PATH


@pytest.fixture(scope="session")
def _can_build_engine(onnx_path) -> bool:
    """Check if TRT can build engines on this hardware (session-cached)."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".engine", delete=True) as f:
            build_engine(onnx_path, f.name, optimization_level=1)
            return True
    except RuntimeError:
        return False
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _skip_if_cannot_build(request: pytest.FixtureRequest, _can_build_engine: bool) -> None:
    """Skip builder tests requiring GPU if TRT cannot build engines."""
    if not request.node.get_closest_marker("cpu") and not _can_build_engine:
        pytest.skip("TRT does not support this GPU's compute capability")
