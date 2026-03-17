# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine test fixtures -- built engines, test data, memory mode configs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import tensorrt as trt

from trtutils import TRTEngine
from trtutils.builder import build_engine

if TYPE_CHECKING:
    from typing import Callable, Generator

DATA_DIR = Path(__file__).parent.parent.parent / "data"
ENGINES_DIR = DATA_DIR / "engines"


def _build_test_engine(onnx_name: str) -> Path:
    """Build and cache an engine from an ONNX file in DATA_DIR."""
    onnx_path = DATA_DIR / f"{onnx_name}.onnx"
    engine_path = ENGINES_DIR / f"{onnx_name}_b1_{trt.__version__}.engine"
    if not engine_path.exists():
        ENGINES_DIR.mkdir(parents=True, exist_ok=True)
        build_engine(onnx_path, engine_path, optimization_level=1)
    return engine_path


SIMPLE_ENGINE_PATH = _build_test_engine("simple")

ENGINE_PATHS = [
    pytest.param(SIMPLE_ENGINE_PATH, id="simple"),
]


@pytest.fixture(scope="session")
def engine_path(build_test_engine: Callable[..., Path]) -> Path:
    """Session-scoped built engine for general engine tests."""
    return SIMPLE_ENGINE_PATH


@pytest.fixture
def engine(engine_path) -> Generator:
    """Create a fresh TRTEngine instance per test (no warmup)."""
    eng = TRTEngine(engine_path, warmup=False)
    yield eng
    del eng


@pytest.fixture
def engine_no_pagelocked(engine_path) -> Generator:
    """TRTEngine with pagelocked_mem=False."""
    eng = TRTEngine(engine_path, warmup=False, pagelocked_mem=False)
    yield eng
    del eng


@pytest.fixture
def random_input(engine) -> list:
    """Generate random input matching engine spec."""
    return engine.get_random_input()
