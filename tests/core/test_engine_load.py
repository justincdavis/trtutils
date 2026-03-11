# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_engine.py -- Engine loading and name extraction."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from trtutils._flags import FLAGS
from trtutils.core._engine import create_engine, get_engine_names
from trtutils.core._stream import destroy_stream


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("simple_engine_path", id="Path"),
        pytest.param("simple_engine_path_str", id="str"),
    ],
)
def test_create_engine(path, request) -> None:
    """create_engine returns a valid 4-tuple, accepts str and Path."""
    engine_path = request.getfixturevalue(path)
    result = create_engine(engine_path)
    assert isinstance(result, tuple)
    assert len(result) == 4
    engine, context, logger, stream = result
    assert engine is not None
    assert context is not None
    assert logger is not None
    assert stream is not None
    destroy_stream(stream)


def test_create_engine_external_stream(simple_engine_path, cuda_stream) -> None:
    """create_engine reuses the provided stream."""
    _engine, _context, _logger, stream = create_engine(simple_engine_path, stream=cuda_stream)
    assert _engine is not None
    assert stream is cuda_stream


def test_create_engine_invalid_path_raises(tmp_path) -> None:
    """FileNotFoundError for non-existent path."""
    fake_path = tmp_path / "nonexistent.engine"
    with pytest.raises(FileNotFoundError, match="Engine file not found"):
        create_engine(fake_path)


@pytest.mark.parametrize(
    ("no_warn", "dla_core"),
    [
        pytest.param(True, None, id="no_warn"),
        pytest.param(False, 0, id="dla_core"),
    ],
)
def test_create_engine_options(simple_engine_path, no_warn, dla_core) -> None:
    """create_engine with no_warn and dla_core options."""
    kwargs = {"no_warn": no_warn}
    if dla_core is not None:
        kwargs["dla_core"] = dla_core
    engine, _context, _logger, stream = create_engine(simple_engine_path, **kwargs)
    assert engine is not None
    destroy_stream(stream)


def test_get_engine_names(simple_engine) -> None:
    """get_engine_names returns non-empty string lists matching engine tensor count."""
    engine, _context, _stream = simple_engine
    result = get_engine_names(engine)
    assert isinstance(result, tuple)
    assert len(result) == 2
    input_names, output_names = result
    # non-empty lists of strings
    assert isinstance(input_names, list)
    assert isinstance(output_names, list)
    assert len(input_names) > 0
    assert len(output_names) > 0
    for name in input_names + output_names:
        assert isinstance(name, str)
        assert len(name) > 0
    # count matches engine
    expected = engine.num_io_tensors if FLAGS.TRT_10 else engine.num_bindings
    assert len(input_names) + len(output_names) == expected


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("engine_returns_none", "match_msg"),
    [
        pytest.param(True, "Failed to deserialize", id="engine_none"),
        pytest.param(False, "Failed to create execution context", id="context_none"),
    ],
)
def test_create_engine_mocked_none_raises(tmp_path, engine_returns_none, match_msg) -> None:
    """RuntimeError when deserialization or context creation returns None."""
    fake_engine = tmp_path / "bad.engine"
    fake_engine.write_bytes(b"\x00" * 16)

    mock_runtime = MagicMock()
    if engine_returns_none:
        mock_runtime.deserialize_cuda_engine.return_value = None
    else:
        mock_engine = MagicMock()
        mock_engine.create_execution_context.return_value = None
        mock_runtime.deserialize_cuda_engine.return_value = mock_engine

    with patch("trtutils.core._engine.trt.Runtime", return_value=mock_runtime):
        with patch("trtutils.core._engine.Device", MagicMock()):
            with patch("trtutils.core._engine.CONFIG"):
                with pytest.raises(RuntimeError, match=match_msg):
                    create_engine(fake_engine)


@pytest.mark.cpu
def test_get_engine_names_legacy_path() -> None:
    """Legacy TRT < 10 binding-based path returns correct names."""
    mock_engine = MagicMock()
    type(mock_engine).num_bindings = PropertyMock(return_value=3)

    binding_names = ["input_0", "output_0", "output_1"]
    is_input_flags = [True, False, False]

    mock_engine.get_binding_name.side_effect = lambda i: binding_names[i]
    mock_engine.binding_is_input.side_effect = lambda i: is_input_flags[i]

    with patch("trtutils.core._engine.FLAGS") as mock_flags:
        mock_flags.TRT_10 = False
        input_names, output_names = get_engine_names(mock_engine)

    assert input_names == ["input_0"]
    assert output_names == ["output_0", "output_1"]
    assert mock_engine.get_binding_name.call_count == 3
    assert mock_engine.binding_is_input.call_count == 3
    mock_engine.get_tensor_name.assert_not_called()
