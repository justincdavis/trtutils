"""Tests for src/trtutils/core/_engine.py -- Engine loading and name extraction."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# create_engine tests
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestCreateEngine:
    """Tests for create_engine()."""

    def test_create_from_valid_path(self, simple_engine_path):
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

    def test_create_with_str_path(self, simple_engine_path):
        """create_engine accepts a string path."""
        from trtutils.core._engine import create_engine
        from trtutils.core._stream import destroy_stream

        engine, _context, _logger, stream = create_engine(str(simple_engine_path))
        assert engine is not None
        destroy_stream(stream)

    def test_create_with_external_stream(self, simple_engine_path, cuda_stream):
        """create_engine reuses the provided stream."""
        from trtutils.core._engine import create_engine

        _engine, _context, _logger, stream = create_engine(simple_engine_path, stream=cuda_stream)
        assert _engine is not None
        # Should return the same stream we passed in
        assert stream is cuda_stream

    def test_create_without_stream(self, simple_engine_path):
        """create_engine creates a new stream when none is provided."""
        from trtutils.core._engine import create_engine
        from trtutils.core._stream import destroy_stream

        _engine, _context, _logger, stream = create_engine(simple_engine_path)
        assert stream is not None
        destroy_stream(stream)

    def test_invalid_path_raises(self, tmp_path):
        """create_engine raises FileNotFoundError for a non-existent path."""
        from trtutils.core._engine import create_engine

        fake_path = tmp_path / "nonexistent.engine"
        with pytest.raises(FileNotFoundError, match="Engine file not found"):
            create_engine(fake_path)

    def test_no_warn(self, simple_engine_path):
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

    def test_returns_input_output_names(self, simple_engine_path):
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

    def test_names_are_non_empty(self, simple_engine_path):
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

    def test_names_are_strings(self, simple_engine_path):
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

    def test_names_match_engine_tensor_count(self, simple_engine_path):
        """Total names count matches engine's io tensor count."""
        from trtutils._flags import FLAGS
        from trtutils.core._engine import create_engine, get_engine_names
        from trtutils.core._stream import destroy_stream

        engine, _context, _logger, stream = create_engine(simple_engine_path)
        try:
            input_names, output_names = get_engine_names(engine)

            if FLAGS.TRT_10:
                expected = engine.num_io_tensors
            else:
                expected = engine.num_bindings

            assert len(input_names) + len(output_names) == expected
        finally:
            destroy_stream(stream)
