# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/inspect/ -- engine and ONNX inspection utilities."""

from __future__ import annotations

import pytest

from trtutils.inspect import LayerInfo, get_engine_names, inspect_engine, inspect_onnx_layers


@pytest.mark.parametrize(
    "engine_fixture",
    [
        pytest.param("engine_path", id="simple"),
        pytest.param("yolov10_engine_path", id="yolov10"),
    ],
)
def test_inspect_engine(engine_fixture, request) -> None:
    """inspect_engine returns correct structure with typed fields."""
    result = inspect_engine(request.getfixturevalue(engine_fixture))

    assert isinstance(result, tuple)
    assert len(result) == 4

    mem_size, batch_size, inputs, outputs = result
    assert isinstance(mem_size, int)
    assert mem_size >= 0
    assert isinstance(batch_size, int)
    assert isinstance(inputs, list)
    assert isinstance(outputs, list)

    # both inputs and outputs must have at least one 4-tuple entry
    for tensor_list in (inputs, outputs):
        assert len(tensor_list) > 0
        for entry in tensor_list:
            assert isinstance(entry, tuple)
            assert len(entry) == 4
            name, _shape, _dtype, _fmt = entry
            assert isinstance(name, str)
            assert len(name) > 0


@pytest.mark.parametrize(
    "engine_fixture",
    [
        pytest.param("engine_path", id="simple"),
        pytest.param("yolov10_engine_path", id="yolov10"),
    ],
)
def test_get_engine_names(engine_fixture, request) -> None:
    """get_engine_names returns non-empty lists of input/output name strings."""
    result = get_engine_names(request.getfixturevalue(engine_fixture))

    assert isinstance(result, tuple)
    assert len(result) == 2

    input_names, output_names = result
    assert isinstance(input_names, list)
    assert isinstance(output_names, list)
    assert len(input_names) >= 1
    assert len(output_names) >= 1

    for name in input_names + output_names:
        assert isinstance(name, str)
        assert len(name) > 0


@pytest.mark.parametrize(
    "onnx_fixture",
    [
        pytest.param("simple_onnx_path", id="simple"),
        pytest.param("yolov10_onnx_path", id="yolov10"),
    ],
)
def test_inspect_onnx_layers(onnx_fixture, request) -> None:
    """inspect_onnx_layers returns well-formed layer tuples."""
    onnx_path = request.getfixturevalue(onnx_fixture)
    if not onnx_path.exists():
        pytest.skip(f"{onnx_path.name} not found")

    layers = inspect_onnx_layers(onnx_path)
    assert isinstance(layers, list)
    assert len(layers) > 0
    for layer in layers:
        assert isinstance(layer, LayerInfo)
        assert isinstance(layer.index, int)
        assert isinstance(layer.name, str)
        assert isinstance(layer.layer_type, str)
        assert isinstance(layer.input_tensor_size, int)
        assert isinstance(layer.output_tensor_size, int)
        assert isinstance(layer.dla_compatible, bool)


def test_inspect_engine_nonexistent_raises() -> None:
    """Nonexistent engine file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        inspect_engine("nonexistent_engine_abc123.engine")


def test_inspect_onnx_layers_nonexistent_raises() -> None:
    """Nonexistent ONNX file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        inspect_onnx_layers("nonexistent_model_abc123.onnx")
