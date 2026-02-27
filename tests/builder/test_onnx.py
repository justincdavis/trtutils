# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for read_onnx() -- validation, parsing, and error branches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _read_onnx_import():
    """Import read_onnx lazily (only on GPU)."""
    from trtutils.builder._onnx import read_onnx

    return read_onnx


# ===========================================================================
# Successful parsing tests
# ===========================================================================
class TestReadOnnx:
    """Tests for successful read_onnx calls."""

    @pytest.mark.gpu
    def test_valid_onnx(self, onnx_path: Path) -> None:
        """Returns a 4-tuple (network, builder, config, parser) for valid ONNX."""
        read_onnx = _read_onnx_import()
        result = read_onnx(onnx_path)
        assert isinstance(result, tuple)
        assert len(result) == 4

    @pytest.mark.gpu
    def test_returns_correct_types(self, onnx_path: Path) -> None:
        """Returned objects are the expected TensorRT types."""
        from trtutils.compat._libs import trt

        read_onnx = _read_onnx_import()
        network, builder, config, parser = read_onnx(onnx_path)
        assert isinstance(network, trt.INetworkDefinition)
        assert isinstance(builder, trt.Builder)
        assert isinstance(config, trt.IBuilderConfig)
        assert isinstance(parser, trt.OnnxParser)

    @pytest.mark.gpu
    def test_network_has_inputs(self, onnx_path: Path) -> None:
        """Network has at least one input tensor."""
        read_onnx = _read_onnx_import()
        network, _, _, _ = read_onnx(onnx_path)
        assert network.num_inputs > 0

    @pytest.mark.gpu
    def test_network_has_outputs(self, onnx_path: Path) -> None:
        """Network has at least one output tensor."""
        read_onnx = _read_onnx_import()
        network, _, _, _ = read_onnx(onnx_path)
        assert network.num_outputs > 0

    @pytest.mark.gpu
    def test_custom_workspace(self, onnx_path: Path) -> None:
        """Custom workspace size is applied to config."""
        read_onnx = _read_onnx_import()
        # Using a small workspace of 1.0 GiB
        network, _builder, _config, _parser = read_onnx(onnx_path, workspace=1.0)
        assert network is not None

    @pytest.mark.gpu
    def test_string_path(self, onnx_path: Path) -> None:
        """Accepts string path in addition to Path object."""
        read_onnx = _read_onnx_import()
        network, _builder, _config, _parser = read_onnx(str(onnx_path))
        assert network.num_inputs > 0

    @pytest.mark.gpu
    def test_workspace_memory_set(self, onnx_path: Path) -> None:
        """Workspace memory is configured on the builder config."""
        read_onnx = _read_onnx_import()
        from trtutils.compat._libs import trt

        _, _, config, _ = read_onnx(onnx_path, workspace=2.0)
        # The workspace should be set -- check via the appropriate API
        expected_bytes = int(2.0 * (1 << 30))
        if hasattr(config, "max_workspace_size"):
            assert config.max_workspace_size == expected_bytes
        else:
            # TRT 8.4+ uses set_memory_pool_limit; we can verify by querying
            pool_limit = config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE)
            assert pool_limit == expected_bytes


# ===========================================================================
# Error handling tests
# ===========================================================================
class TestReadOnnxErrors:
    """Tests for read_onnx error branches."""

    @pytest.mark.gpu
    def test_file_not_found(self, tmp_path: Path) -> None:
        """FileNotFoundError when ONNX file does not exist."""
        read_onnx = _read_onnx_import()
        with pytest.raises(FileNotFoundError, match="Could not find ONNX model"):
            read_onnx(tmp_path / "nonexistent.onnx")

    @pytest.mark.gpu
    def test_is_directory(self, tmp_path: Path) -> None:
        """IsADirectoryError when path points to a directory."""
        read_onnx = _read_onnx_import()
        onnx_dir = tmp_path / "model.onnx"
        onnx_dir.mkdir()
        with pytest.raises(IsADirectoryError, match="Path given is a directory"):
            read_onnx(onnx_dir)

    @pytest.mark.gpu
    def test_wrong_extension(self, non_onnx_file: Path) -> None:
        """ValueError when file doesn't have .onnx extension."""
        read_onnx = _read_onnx_import()
        with pytest.raises(ValueError, match=r"File does not have \.onnx extension"):
            read_onnx(non_onnx_file)

    @pytest.mark.gpu
    def test_invalid_onnx_content(self, invalid_onnx_file: Path) -> None:
        """Invalid ONNX content raises RuntimeError."""
        read_onnx = _read_onnx_import()
        with pytest.raises(RuntimeError, match="Cannot parse ONNX file"):
            read_onnx(invalid_onnx_file)
