# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine test fixtures -- built engines, test data, memory mode configs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx
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


def _build_dynamic_test_engine(onnx_name: str, max_batch: int = 4) -> Path:
    """
    Build an engine with a dynamic batch profile from a static ONNX.

    The exported test models are static, so the batch dimension is made
    symbolic here rather than shipping a second ONNX asset. Only trivial
    graphs survive this; models with hardcoded batch constants do not.
    """
    onnx_path = DATA_DIR / f"{onnx_name}.onnx"
    engine_path = ENGINES_DIR / f"{onnx_name}_dyn_b{max_batch}_{trt.__version__}.engine"
    if engine_path.exists():
        return engine_path

    ENGINES_DIR.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(onnx_path))
    for tensor in list(model.graph.input) + list(model.graph.output):
        dim = tensor.type.tensor_type.shape.dim[0]
        dim.ClearField("dim_value")
        dim.dim_param = "batch"
    dyn_onnx = ENGINES_DIR / f"{onnx_name}_dyn.onnx"
    onnx.save(model, str(dyn_onnx))

    shape = tuple(d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim[1:])
    build_engine(
        dyn_onnx,
        engine_path,
        optimization_level=1,
        shapes=[
            (
                model.graph.input[0].name,
                ((1, *shape), (max_batch, *shape), (max_batch, *shape)),
            )
        ],
    )
    return engine_path


SIMPLE_ENGINE_PATH = _build_test_engine("simple")
SIMPLE_DYNAMIC_ENGINE_PATH = _build_dynamic_test_engine("simple")

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
