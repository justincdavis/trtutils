# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for read_onnx() -- validation, parsing, and error branches."""

from __future__ import annotations

import pytest

from trtutils.builder._onnx import read_onnx
from trtutils.compat._libs import trt


def test_valid_onnx(onnx_path) -> None:
    """Returns a 4-tuple (network, builder, config, parser) for valid ONNX."""
    result = read_onnx(onnx_path)
    assert isinstance(result, tuple)
    assert len(result) == 4


def test_returns_correct_types(onnx_path) -> None:
    """Returned objects are the expected TensorRT types."""
    network, builder, config, parser = read_onnx(onnx_path)
    assert isinstance(network, trt.INetworkDefinition)
    assert isinstance(builder, trt.Builder)
    assert isinstance(config, trt.IBuilderConfig)
    assert isinstance(parser, trt.OnnxParser)


def test_network_has_inputs(onnx_path) -> None:
    """Network has at least one input tensor."""
    network, _, _, _ = read_onnx(onnx_path)
    assert network.num_inputs > 0


def test_network_has_outputs(onnx_path) -> None:
    """Network has at least one output tensor."""
    network, _, _, _ = read_onnx(onnx_path)
    assert network.num_outputs > 0


def test_string_path(onnx_path) -> None:
    """Accepts string path in addition to Path object."""
    network, _builder, _config, _parser = read_onnx(str(onnx_path))
    assert network.num_inputs > 0


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
