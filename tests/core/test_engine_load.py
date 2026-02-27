"""Tests for src/trtutils/core/_engine.py -- Engine loading and name extraction."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# create_engine tests
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestCreateEngine:
    """Tests for create_engine()."""

    def test_create_from_valid_path(self, simple_engine_path) -> None:
        """create_engine returns a 4-tuple (engine, context, logger, stream)."""
        from trtutils.core._engine import create_engine
        from trtutils.core._stream import destroy_stream

        result = create_engine(simple_engine_path)
        assert isinstance(result, tuple)
        assert len(result) == 4

        engine, context, logger, stream = result
        assert engine is not None
        assert context is not None
        assert logger is not None
        assert stream is not None

        destroy_stream(stream)

    def test_create_with_str_path(self, simple_engine_path) -> None:
        """create_engine accepts a string path."""
        from trtutils.core._engine import create_engine
        from trtutils.core._stream import destroy_stream

        engine, _context, _logger, stream = create_engine(str(simple_engine_path))
        assert engine is not None
        destroy_stream(stream)

    def test_create_with_external_stream(self, simple_engine_path, cuda_stream: object) -> None:
        """create_engine reuses the provided stream."""
        from trtutils.core._engine import create_engine

        _engine, _context, _logger, stream = create_engine(simple_engine_path, stream=cuda_stream)
        assert _engine is not None
        # Should return the same stream we passed in
        assert stream is cuda_stream

    def test_create_without_stream(self, simple_engine_path) -> None:
        """create_engine creates a new stream when none is provided."""
        from trtutils.core._engine import create_engine
        from trtutils.core._stream import destroy_stream

        _engine, _context, _logger, stream = create_engine(simple_engine_path)
        assert stream is not None
        destroy_stream(stream)

    def test_invalid_path_raises(self, tmp_path) -> None:
        """create_engine raises FileNotFoundError for a non-existent path."""
        from trtutils.core._engine import create_engine

        fake_path = tmp_path / "nonexistent.engine"
        with pytest.raises(FileNotFoundError, match="Engine file not found"):
            create_engine(fake_path)

    def test_no_warn(self, simple_engine_path) -> None:
        """create_engine with no_warn=True suppresses warnings without error."""
        from trtutils.core._engine import create_engine
        from trtutils.core._stream import destroy_stream

        engine, _context, _logger, stream = create_engine(simple_engine_path, no_warn=True)
        assert engine is not None
        destroy_stream(stream)


# ---------------------------------------------------------------------------
# get_engine_names tests
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestGetEngineNames:
    """Tests for get_engine_names()."""

    def test_returns_input_output_names(self, simple_engine_path) -> None:
        """get_engine_names returns a tuple of (input_names, output_names)."""
        from trtutils.core._engine import create_engine, get_engine_names
        from trtutils.core._stream import destroy_stream

        engine, _context, _logger, stream = create_engine(simple_engine_path)
        try:
            result = get_engine_names(engine)
            assert isinstance(result, tuple)
            assert len(result) == 2

            input_names, output_names = result
            assert isinstance(input_names, list)
            assert isinstance(output_names, list)
        finally:
            destroy_stream(stream)

    def test_names_are_non_empty(self, simple_engine_path) -> None:
        """Engine has at least one input and one output name."""
        from trtutils.core._engine import create_engine, get_engine_names
        from trtutils.core._stream import destroy_stream

        engine, _context, _logger, stream = create_engine(simple_engine_path)
        try:
            input_names, output_names = get_engine_names(engine)
            assert len(input_names) > 0
            assert len(output_names) > 0
        finally:
            destroy_stream(stream)

    def test_names_are_strings(self, simple_engine_path) -> None:
        """All returned names are strings."""
        from trtutils.core._engine import create_engine, get_engine_names
        from trtutils.core._stream import destroy_stream

        engine, _context, _logger, stream = create_engine(simple_engine_path)
        try:
            input_names, output_names = get_engine_names(engine)
            for name in input_names + output_names:
                assert isinstance(name, str)
                assert len(name) > 0
        finally:
            destroy_stream(stream)

    def test_names_match_engine_tensor_count(self, simple_engine_path) -> None:
        """Total names count matches engine's io tensor count."""
        from trtutils._flags import FLAGS
        from trtutils.core._engine import create_engine, get_engine_names
        from trtutils.core._stream import destroy_stream

        engine, _context, _logger, stream = create_engine(simple_engine_path)
        try:
            input_names, output_names = get_engine_names(engine)

            expected = engine.num_io_tensors if FLAGS.TRT_10 else engine.num_bindings

            assert len(input_names) + len(output_names) == expected
        finally:
            destroy_stream(stream)


# ---------------------------------------------------------------------------
# create_engine -- DLA core parameter
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestCreateEngineDLACore:
    """Tests for the dla_core parameter of create_engine()."""

    def test_dla_core_sets_runtime_attribute(self, simple_engine_path) -> None:
        """create_engine with dla_core=0 exercises the DLA_core assignment path."""
        from trtutils.core._engine import create_engine
        from trtutils.core._stream import destroy_stream

        # On non-Jetson hardware DLA_core is accepted but has no effect.
        # The test verifies the code path does not raise.
        engine, _context, _logger, stream = create_engine(simple_engine_path, dla_core=0)
        assert engine is not None
        destroy_stream(stream)


# ---------------------------------------------------------------------------
# create_engine -- mocked failure paths
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestCreateEngineMockedFailures:
    """Tests for defensive checks in create_engine() using mocks."""

    def test_engine_none_raises(self, tmp_path) -> None:
        """RuntimeError raised when deserialize_cuda_engine returns None."""
        from unittest.mock import MagicMock, patch

        from trtutils.core._engine import create_engine

        # Create a fake engine file so the FileNotFoundError check passes.
        fake_engine = tmp_path / "bad.engine"
        fake_engine.write_bytes(b"\x00" * 16)

        # Mock the TRT Runtime so that deserialize_cuda_engine returns None.
        mock_runtime = MagicMock()
        mock_runtime.deserialize_cuda_engine.return_value = None

        with patch("trtutils.core._engine.trt.Runtime", return_value=mock_runtime):
            with patch("trtutils.core._engine.Device", MagicMock()):
                with patch("trtutils.core._engine.CONFIG"):
                    with pytest.raises(RuntimeError, match="Failed to deserialize"):
                        create_engine(fake_engine)

    def test_context_none_raises(self, tmp_path) -> None:
        """RuntimeError raised when create_execution_context returns None."""
        from unittest.mock import MagicMock, patch

        from trtutils.core._engine import create_engine

        # Create a fake engine file.
        fake_engine = tmp_path / "ctx_fail.engine"
        fake_engine.write_bytes(b"\x00" * 16)

        # Mock the TRT Runtime so that deserialization succeeds
        # but create_execution_context returns None.
        mock_engine = MagicMock()
        mock_engine.create_execution_context.return_value = None

        mock_runtime = MagicMock()
        mock_runtime.deserialize_cuda_engine.return_value = mock_engine

        with patch("trtutils.core._engine.trt.Runtime", return_value=mock_runtime):
            with patch("trtutils.core._engine.Device", MagicMock()):
                with patch("trtutils.core._engine.CONFIG"):
                    with pytest.raises(RuntimeError, match="Failed to create execution context"):
                        create_engine(fake_engine)


# ---------------------------------------------------------------------------
# get_engine_names -- legacy TRT < 10 code path
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestGetEngineNamesLegacy:
    """Tests for the TRT < 10 (binding-based) path of get_engine_names()."""

    def test_legacy_binding_path(self) -> None:
        """Exercise get_binding_name / binding_is_input when FLAGS.TRT_10 is False."""
        from unittest.mock import MagicMock, PropertyMock, patch

        from trtutils.core._engine import get_engine_names

        # Build a mock engine that exposes the legacy bindings API.
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

        # Verify the legacy methods were called, not the TRT 10 ones.
        assert mock_engine.get_binding_name.call_count == 3
        assert mock_engine.binding_is_input.call_count == 3
        mock_engine.get_tensor_name.assert_not_called()
