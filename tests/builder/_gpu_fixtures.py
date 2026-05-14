# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""GPU-dependent builder fixtures, loaded by conftest.py when CUDA is available."""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tests.conftest import DATA_DIR
from trtutils._flags import FLAGS
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
def quantized_onnx_path(tmp_path_factory) -> Path:
    """
    Minimal Q/DQ ONNX: fp32 input -> Quantize -> Dequantize -> Identity -> fp32 output.

    Scale and zero_point are constant initializers. No weights, no dynamic
    shapes. Used for exercising the strongly-typed build path.
    """
    if not FLAGS.STRONGLY_TYPED_SUPPORTED:
        pytest.skip("Installed TensorRT does not support strongly-typed networks")

    out_dir = tmp_path_factory.mktemp("qdq")
    out_path = out_dir / "qdq.onnx"

    shape = [1, 3, 8, 8]
    scale = numpy_helper.from_array(np.array(0.01, dtype=np.float32), name="scale")
    zero_point = numpy_helper.from_array(np.array(0, dtype=np.int8), name="zero_point")

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)

    quantize = helper.make_node(
        "QuantizeLinear",
        inputs=["input", "scale", "zero_point"],
        outputs=["quantized"],
        name="quantize",
    )
    dequantize = helper.make_node(
        "DequantizeLinear",
        inputs=["quantized", "scale", "zero_point"],
        outputs=["dequantized"],
        name="dequantize",
    )
    identity = helper.make_node(
        "Identity",
        inputs=["dequantized"],
        outputs=["output"],
        name="identity",
    )

    graph = helper.make_graph(
        nodes=[quantize, dequantize, identity],
        name="qdq_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[scale, zero_point],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="trtutils-tests",
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    return out_path


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
