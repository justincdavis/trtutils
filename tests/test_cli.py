# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: PLC2701
from __future__ import annotations

from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from trtutils.__main__ import (
    _benchmark,
    _build,
    _build_dla,
    _can_run_on_dla,
    _classify,
    _detect,
    _inspect,
    _main,
)


def test_main_no_args() -> None:
    """Test calling main without arguments."""
    with patch("sys.argv", ["trtutils"]), patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        _main()
        assert "Utilities for TensorRT." in mock_stdout.getvalue()


def test_benchmark_no_args() -> None:
    """Test calling benchmark without required arguments."""
    args = SimpleNamespace()
    # missing required 'engine' attribute
    with pytest.raises(AttributeError):
        _benchmark(args)


def test_build_no_args() -> None:
    """Test calling build without required arguments."""
    args = SimpleNamespace()
    # missing required 'onnx' and 'output' attributes
    with pytest.raises(AttributeError):
        _build(args)


def test_build_dla_no_args() -> None:
    """Test calling build-dla without required arguments."""
    args = SimpleNamespace()
    # missing required 'onnx' and 'output' attributes
    with pytest.raises(AttributeError):
        _build_dla(args)


def test_can_run_on_dla_no_args() -> None:
    """Test calling can-run-on-dla without required arguments."""
    args = SimpleNamespace()
    # missing required 'onnx' attribute
    with pytest.raises(AttributeError):
        _can_run_on_dla(args)


def test_classify_no_args() -> None:
    """Test calling classify without required arguments."""
    args = SimpleNamespace()
    # missing required 'engine' and 'input' attributes
    with pytest.raises(AttributeError):
        _classify(args)


def test_detect_no_args() -> None:
    """Test calling detect without required arguments."""
    args = SimpleNamespace()
    # missing required 'engine' and 'input' attributes
    with pytest.raises(AttributeError):
        _detect(args)


def test_inspect_no_args() -> None:
    """Test calling inspect without required arguments."""
    args = SimpleNamespace()
    # missing required 'engine' attribute
    with pytest.raises(AttributeError):
        _inspect(args)
