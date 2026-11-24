# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/builder/onnx/_shapes.py -- ONNX input/output shape helpers."""

from __future__ import annotations

import onnx
import pytest
from onnx import TensorProto, helper

from trtutils.builder.onnx._shapes import (
    get_onnx_input,
    get_onnx_output,
    get_static_hw,
    get_static_imgsz,
    make_onnx_static,
)

pytestmark = pytest.mark.cpu

# dims: int -> static, str -> dim_param, None -> dim with neither field set
DYNAMIC_SHAPES = [
    pytest.param(["N", 3, 4, 4], (1, 3, 4, 4), id="dim-param-batch"),
    pytest.param([0, 3, 4, 4], (1, 3, 4, 4), id="dim-value-zero"),
    pytest.param([None, 3, 4, 4], (1, 3, 4, 4), id="dim-unset"),
    pytest.param(["N", 3, "H", "W"], (1, 3, 1, 1), id="dim-param-all-spatial"),
]

STATIC_SHAPES = [
    pytest.param([1, 3, 4, 4], id="batch-1"),
    pytest.param([2, 3, 8, 8], id="batch-2"),
]


def _write_relu_model(path, dims) -> None:
    """Write a single-Relu ONNX model with input/output of the given dims."""
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, dims)
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, dims)
    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph([node], "relu", [inp], [out])
    model = helper.make_model(graph)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


@pytest.mark.parametrize(("dims", "expected"), DYNAMIC_SHAPES)
def test_get_onnx_input_output_replace_dynamic(tmp_path, dims, expected) -> None:
    """Input and output name/shape are read, with dynamic dims replaced by 1."""
    path = tmp_path / "model.onnx"
    _write_relu_model(path, dims)
    assert get_onnx_input(path) == ("x", expected)
    assert get_onnx_output(path) == ("y", expected)


@pytest.mark.parametrize("dims", STATIC_SHAPES)
def test_get_onnx_input_output_static(tmp_path, dims) -> None:
    """Static dims are returned unchanged for input and output."""
    path = tmp_path / "model.onnx"
    _write_relu_model(path, dims)
    assert get_onnx_input(path) == ("x", tuple(dims))
    assert get_onnx_output(path) == ("y", tuple(dims))


@pytest.mark.parametrize(("dims", "expected"), DYNAMIC_SHAPES)
def test_make_onnx_static_fixes_dynamic_input(tmp_path, dims, expected) -> None:
    """Dynamic input dims become dim_value 1 on disk, other dims are kept, model stays valid."""
    path = tmp_path / "model.onnx"
    _write_relu_model(path, dims)
    make_onnx_static(path)

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    input_dims = model.graph.input[0].type.tensor_type.shape.dim
    assert tuple(d.dim_value for d in input_dims) == expected
    assert all(d.dim_param == "" for d in input_dims)
    assert get_onnx_input(path) == ("x", expected)


@pytest.mark.parametrize("dims", STATIC_SHAPES)
def test_make_onnx_static_static_unchanged(tmp_path, dims) -> None:
    """A fully static model is not rewritten (file bytes are identical)."""
    path = tmp_path / "model.onnx"
    _write_relu_model(path, dims)
    before = path.read_bytes()
    make_onnx_static(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("dims", "name", "hw", "imgsz"),
    [
        pytest.param([1, 3, 384, 384], "x", (384, 384), 384, id="square"),
        pytest.param(["N", 3, 256, 128], "x", (256, 128), None, id="height-width"),
        pytest.param([1, 3, "h", "w"], "x", None, None, id="symbolic-hw"),
        pytest.param([1, 3, 256, "w"], "x", None, None, id="symbolic-width"),
        pytest.param([1, 384, 384, 3], "x", None, None, id="nhwc"),
        pytest.param([1, 3, 384], "x", None, None, id="rank-3"),
        pytest.param([1, 3, 384, 384], "other", None, None, id="other-name"),
    ],
)
def test_get_static_hw_and_imgsz(tmp_path, dims, name, hw, imgsz) -> None:
    """get_static_hw reads a static NCHW (H, W); get_static_imgsz keeps only square sizes."""
    path = tmp_path / "model.onnx"
    _write_relu_model(path, dims)
    assert get_static_hw(path, name) == hw
    assert get_static_imgsz(path, name) == imgsz
