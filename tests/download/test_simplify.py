# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/download/_simplify.py -- temp file cleanup."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from trtutils.download import _simplify

pytestmark = pytest.mark.cpu


def _fake_run_cmd(cmd, *, cwd=None, env=None, verbose=None, **_kwargs: object) -> int:
    # like polygraphy: drop a temp file in TMPDIR, then fail or write the output
    tmp_dir = (env if env is not None else os.environ)["TMPDIR"]
    with tempfile.NamedTemporaryFile(
        prefix="tmp_polygraphy_", suffix=".onnx", dir=tmp_dir, delete=False
    ):
        pass
    # like onnxruntime symbolic shape inference: temp model + external data in cwd
    work_dir = Path(cwd) if cwd is not None else Path.cwd()
    (work_dir / "sym_shape_infer_temp.onnx").write_bytes(b"x")
    (work_dir / "00000000-0000-0000-0000-000000000000.data").write_bytes(b"x")
    if "polygraphy" in str(cmd[0]) and "fail" in os.environ.get("SIMPLIFY_TEST", ""):
        raise subprocess.CalledProcessError(1, cmd)
    Path(cmd[-1]).write_bytes(b"simplified")
    return 0


@pytest.mark.parametrize("first_fails", [False, True], ids=["success", "first-tool-fails"])
def test_simplify_leaves_no_temp_files(tmp_path, monkeypatch, first_fails) -> None:
    """Temp files made by the tools are removed on success and on tool failure."""
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmp_dir))
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setenv("SIMPLIFY_TEST", "fail" if first_fails else "")
    monkeypatch.setattr(_simplify, "run_cmd", _fake_run_cmd)
    monkeypatch.setattr(_simplify, "run_uv_pip_install", lambda *_a, **_k: None)

    caller_cwd = tmp_path / "cwd"
    caller_cwd.mkdir()
    monkeypatch.chdir(caller_cwd)

    model = tmp_path / "model.onnx"
    model.write_bytes(b"original")
    bin_path = tmp_path / "venv" / "bin"
    _simplify._run_simplify(model, tmp_path, bin_path, ["polygraphy", "onnxslim"])

    assert model.read_bytes() == b"simplified"
    assert list(tmp_dir.iterdir()) == []
    assert list(caller_cwd.iterdir()) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cwd", "model.onnx", "tmp"]
