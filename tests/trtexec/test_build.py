# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for trtutils.trtexec._build -- build_engine via trtexec."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import Callable

ONNX_PATH = Path(__file__).parent.parent.parent / "data" / "simple.onnx"


# ============================================================================
# Validation tests (CPU-only, no trtexec needed)
# ============================================================================


@pytest.mark.cpu
class TestTrtexecBuildEngineValidation:
    """Test input validation in trtexec.build_engine."""

    def test_missing_weights_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError raised when weight file does not exist."""
        from trtutils.trtexec._build import build_engine

        fake_weights = tmp_path / "nonexistent.onnx"
        output = tmp_path / "output.engine"

        with pytest.raises(FileNotFoundError, match="Weight file not found"):
            build_engine(fake_weights, output)

    def test_directory_raises(self, tmp_path: Path) -> None:
        """IsADirectoryError raised when weight path is a directory."""
        from trtutils.trtexec._build import build_engine

        weights_dir = tmp_path / "model_dir"
        weights_dir.mkdir()
        output = tmp_path / "output.engine"

        with pytest.raises(IsADirectoryError, match="should not be a directory"):
            build_engine(weights_dir, output)

    def test_invalid_extension_raises(self, tmp_path: Path) -> None:
        """ValueError raised when weight file has an unsupported extension."""
        from trtutils.trtexec._build import build_engine

        bad_file = tmp_path / "model.pt"
        bad_file.write_text("fake weights")
        output = tmp_path / "output.engine"

        with pytest.raises(ValueError, match="invalid extension"):
            build_engine(bad_file, output)

    def test_invalid_dla_core_raises(self, tmp_path: Path) -> None:
        """ValueError raised when dla_core is not 0 or 1."""
        from trtutils.trtexec._build import build_engine

        weights = tmp_path / "model.onnx"
        weights.write_text("fake onnx")
        output = tmp_path / "output.engine"

        with pytest.raises(ValueError, match="DLA core must be either 0 or 1"):
            build_engine(weights, output, use_dla_core=5)

    def test_non_integer_shapes_raises(self, tmp_path: Path) -> None:
        """TypeError raised when shape dimensions are not integers."""
        from trtutils.trtexec import _build as build_module

        weights = tmp_path / "model.onnx"
        weights.write_text("fake onnx")
        output = tmp_path / "output.engine"

        with pytest.raises(TypeError, match="Input shapes must be integers"):
            build_module.build_engine(
                weights,
                output,
                shapes=[("images", (1, 3, "bad", 640))],  # type: ignore[arg-type]
            )


# ============================================================================
# Jetson hardware tests
# ============================================================================


@pytest.mark.jetson
@pytest.mark.gpu
class TestTrtexecBuildEngineOnJetson:
    """Test trtexec.build_engine on actual Jetson hardware."""

    def test_build_from_onnx(self, tmp_path: Path, build_test_engine: Callable[..., Path]) -> None:
        """Building an engine from an ONNX file produces a valid engine file."""
        from trtutils.trtexec._build import build_engine

        output = tmp_path / "test_output.engine"

        success = build_engine(ONNX_PATH, output)

        assert success is True
        assert output.exists()
        assert output.stat().st_size > 0
