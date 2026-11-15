# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for read_onnx() -- validation, parsing, and error branches."""

from __future__ import annotations

from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.builder._onnx import read_onnx
from trtutils.compat._libs import trt


@pytest.mark.parametrize(
    "converter",
    [
        pytest.param(Path, id="path"),
        pytest.param(str, id="str"),
    ],
)
def test_returns_correct_types(onnx_path, converter) -> None:
    """Returned objects are the expected TensorRT types and network has I/O."""
    network, builder, config, parser = read_onnx(converter(onnx_path))
    assert isinstance(network, trt.INetworkDefinition)
    assert isinstance(builder, trt.Builder)
    assert isinstance(config, trt.IBuilderConfig)
    assert isinstance(parser, trt.OnnxParser)
    assert network.num_inputs > 0
    assert network.num_outputs > 0


def test_workspace_memory_set(onnx_path) -> None:
    """Workspace memory is configured on the builder config."""
    _, _, config, _ = read_onnx(onnx_path, workspace=2.0)
    expected_bytes = int(2.0 * (1 << 30))
    if hasattr(config, "max_workspace_size"):
        assert config.max_workspace_size == expected_bytes
    else:
        pool_limit = config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE)
        assert pool_limit == expected_bytes


def test_file_not_found(tmp_path) -> None:
    """FileNotFoundError when ONNX file does not exist."""
    with pytest.raises(FileNotFoundError, match="Could not find ONNX model"):
        read_onnx(tmp_path / "nonexistent.onnx")


def test_is_directory(tmp_path) -> None:
    """IsADirectoryError when path points to a directory."""
    onnx_dir = tmp_path / "model.onnx"
    onnx_dir.mkdir()
    with pytest.raises(IsADirectoryError, match="Path given is a directory"):
        read_onnx(onnx_dir)


def test_wrong_extension(non_onnx_file) -> None:
    """ValueError when file doesn't have .onnx extension."""
    with pytest.raises(ValueError, match=r"File does not have \.onnx extension"):
        read_onnx(non_onnx_file)


def test_invalid_onnx_content(invalid_onnx_file) -> None:
    """Invalid ONNX content raises RuntimeError."""
    with pytest.raises(RuntimeError, match="Cannot parse ONNX file"):
        read_onnx(invalid_onnx_file)


@pytest.mark.parametrize(
    ("node", "shape"),
    [
        pytest.param(
            helper.make_node("LRN", ["r"], ["y"], name="lrn", size=3, bias=0.0),
            [1, 3, 8, 8],
            id="lrn-bias-0",
        ),
        pytest.param(
            helper.make_node("Softmax", ["r"], ["y"], name="softmax", axis=0),
            [5],
            id="softmax-1d",
        ),
    ],
)
def test_parse_failure_reports_parser_errors(tmp_path, node, shape) -> None:
    """A parse failure raises RuntimeError with every parser error and logged TensorRT error."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)
    relu = helper.make_node("Relu", ["x"], ["r"], name="relu")
    graph = helper.make_graph([relu, node], "parse_failure", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "bad.onnx"
    onnx.save(model, str(path))

    # what this tensorrt version reports for the model
    flags = (
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        if FLAGS.EXPLICIT_BATCH_FLAG
        else 0
    )
    network = trt.Builder(LOG).create_network(flags)
    parser = trt.OnnxParser(network, LOG)
    with LOG.capture_errors() as logged:
        assert parser.parse(path.read_bytes()) is False
    expected = [parser.get_error(idx).desc() for idx in range(parser.num_errors)]
    assert len(expected) > 0

    with pytest.raises(RuntimeError, match="Cannot parse ONNX file") as exc_info:
        read_onnx(path)
    message = str(exc_info.value)
    assert str(path.resolve()) in message
    assert "node 1" in message
    for text in expected + logged:
        assert " ".join(text.split()) in message


def test_external_data(onnx_path, tmp_path) -> None:
    """ONNX models with external weight files parse via the model path."""
    ext_path = tmp_path / "external.onnx"
    onnx.save_model(
        onnx.load(str(onnx_path)),
        str(ext_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="external.onnx.data",
        size_threshold=0,
    )
    network, _builder, _config, _parser = read_onnx(ext_path)
    assert network.num_layers > 0
