# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/trtexec/_build.py -- build_engine via trtexec."""

from __future__ import annotations

from pathlib import Path

import pytest

from trtutils.trtexec._build import build_engine

ONNX_PATH = Path(__file__).parent.parent.parent / "data" / "simple.onnx"


def _make_validation_case(kind: str, tmp_path: Path) -> tuple[Path, dict]:
    if kind == "missing":
        return tmp_path / "nonexistent.onnx", {}
    if kind == "directory":
        d = tmp_path / "model_dir"
        d.mkdir()
        return d, {}
    if kind == "bad-extension":
        f = tmp_path / "model.pt"
        f.write_text("fake weights")
        return f, {}
    if kind == "bad-dla-core":
        f = tmp_path / "model.onnx"
        f.write_text("fake onnx")
        return f, {"use_dla_core": 5}
    if kind == "bad-shape-dim":
        f = tmp_path / "model.onnx"
        f.write_text("fake onnx")
        return f, {"shapes": [("images", (1, 3, "bad", 640))]}
    err_msg = f"Unknown validation case: {kind}"
    raise ValueError(err_msg)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("kind", "exc_type", "match"),
    [
        pytest.param("missing", FileNotFoundError, "Weight file not found", id="missing"),
        pytest.param("directory", IsADirectoryError, "should not be a directory", id="directory"),
        pytest.param("bad-extension", ValueError, "invalid extension", id="bad-extension"),
        pytest.param(
            "bad-dla-core",
            ValueError,
            "DLA core must be either 0 or 1",
            id="bad-dla-core",
        ),
        pytest.param(
            "bad-shape-dim",
            TypeError,
            "Input shapes must be integers",
            id="bad-shape-dim",
        ),
    ],
)
def test_build_engine_validation(
    kind: str,
    exc_type: type[Exception],
    match: str,
    tmp_path: Path,
) -> None:
    """Invalid inputs to build_engine raise the documented exception."""
    weights, kwargs = _make_validation_case(kind, tmp_path)
    output = tmp_path / "output.engine"
    with pytest.raises(exc_type, match=match):
        build_engine(weights, output, **kwargs)


@pytest.mark.jetson
def test_build_from_onnx(tmp_path: Path) -> None:
    """build_engine produces a non-empty engine file from a real ONNX model."""
    output = tmp_path / "test_output.engine"
    success = build_engine(ONNX_PATH, output)
    assert success is True
    assert output.exists()
    assert output.stat().st_size > 0
