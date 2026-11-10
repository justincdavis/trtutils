# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/trtexec/_build.py -- build_engine via trtexec."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from trtutils._flags import FLAGS
from trtutils.trtexec._build import build_engine

if TYPE_CHECKING:
    from pathlib import Path


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
def test_build_from_onnx(tmp_path, simple_onnx_path) -> None:
    """build_engine produces a non-empty engine file from a real ONNX model."""
    output = tmp_path / "test_output.engine"
    success = build_engine(simple_onnx_path, output)
    assert success is True
    assert output.exists()
    assert output.stat().st_size > 0


def _captured_command(monkeypatch, weights: Path, output: Path, **kwargs: object) -> list[str]:
    calls = []

    def fake_run(command, *_args: object, **_kwargs: object):
        calls.append(command)
        return True, "", ""

    monkeypatch.setattr("trtutils.trtexec._build.run_trtexec", fake_run)
    build_engine(weights, output, **kwargs)
    return list(calls[0])


@pytest.mark.cpu
def test_build_engine_spaced_paths_stay_whole(monkeypatch, tmp_path: Path) -> None:
    """Paths containing spaces reach run_trtexec as single arguments."""
    d = tmp_path / "a dir"
    d.mkdir()
    weights = d / "my model.onnx"
    weights.write_text("fake onnx")
    output = d / "my out.engine"
    argv = _captured_command(monkeypatch, weights, output)
    assert f"--onnx={weights.resolve()}" in argv
    assert f"--saveEngine={output.resolve()}" in argv


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("version", "expected"),
    [
        pytest.param((8, 6), "--memPoolSize=workspace:512", id="trt8.6"),
        pytest.param((10, 3), "--memPoolSize=workspace:512", id="trt10"),
        pytest.param((8, 0), "--workspace=512", id="trt8.0"),
    ],
)
def test_build_engine_workspace_flag(monkeypatch, tmp_path: Path, version, expected) -> None:
    """Workspace uses --memPoolSize on TRT >= 8.4 and --workspace before."""
    monkeypatch.setattr(FLAGS, "TRT_VERSION", version)
    weights = tmp_path / "model.onnx"
    weights.write_text("fake onnx")
    argv = _captured_command(monkeypatch, weights, tmp_path / "o.engine", workspace=512)
    assert expected in argv
    assert sum(a.startswith(("--workspace", "--memPoolSize=workspace")) for a in argv) == 1
