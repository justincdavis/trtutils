# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine test fixtures -- built engines, test data, memory mode configs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import tensorrt as trt

from tests.conftest import DATA_DIR, ENGINES_DIR
from trtutils import TRTEngine
from trtutils.builder import build_engine

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


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


@pytest.fixture
def make_engine(engine_path):
    """Factory fixture that creates TRTEngine instances with automatic cleanup."""
    engines = []

    def _factory(**kwargs):  # noqa: ANN003
        kwargs.setdefault("warmup", False)
        eng = TRTEngine(engine_path, **kwargs)
        engines.append(eng)
        return eng

    yield _factory
    for eng in engines:
        del eng


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
