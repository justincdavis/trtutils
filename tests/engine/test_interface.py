"""Tests for TRTEngineInterface contract tested through TRTEngine."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.gpu
class TestInterfaceInit:
    """Tests for TRTEngineInterface.__init__ via TRTEngine."""

    def test_pagelocked_mem_default(self, engine) -> None:
        """pagelocked_mem=None defaults to True."""
        assert engine.pagelocked_mem is True

    def test_pagelocked_mem_false(self, engine_no_pagelocked) -> None:
        """pagelocked_mem=False is stored correctly."""
        assert engine_no_pagelocked.pagelocked_mem is False

    def test_unified_mem_default(self, engine) -> None:
        """unified_mem=None defaults to FLAGS.IS_JETSON."""
        from trtutils._flags import FLAGS

        assert engine.unified_mem == FLAGS.IS_JETSON

    def test_unified_mem_explicit_true(self, engine_path) -> None:
        """unified_mem=True can be explicitly set."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=False, unified_mem=True)
        assert eng.unified_mem is True
        del eng

    def test_verbose_stored(self, engine_verbose) -> None:
        """verbose=True is stored and accessible."""
        assert engine_verbose._verbose is True

    def test_name_from_path(self, engine, engine_path) -> None:
        """Name property is the stem of the engine path."""
        assert engine.name == engine_path.stem

    def test_backend_invalid_raises(self, engine_path) -> None:
        """Invalid backend raises ValueError."""
        from trtutils import TRTEngine

        with pytest.raises(ValueError, match="Invalid backend"):
            TRTEngine(engine_path, backend="invalid")

    def test_verbose_init_logging(self, engine_path) -> None:
        """verbose=True logs engine info during init."""
        import logging

        from trtutils import TRTEngine

        logger = logging.getLogger("trtutils")
        old_level = logger.level
        # The trtutils logger has propagate=False and level=WARNING,
        # so temporarily lower level and add a handler to capture INFO logs
        logger.setLevel(logging.INFO)

        class _Capture(logging.Handler):
            def __init__(self) -> None:
                super().__init__()
                self.records = []

            def emit(self, record) -> None:
                self.records.append(record)

        handler = _Capture()
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            eng = TRTEngine(engine_path, warmup=False, verbose=True)
            assert eng._verbose is True
            assert any("Loaded engine" in r.getMessage() for r in handler.records)
            del eng
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)


@pytest.mark.gpu
class TestInterfaceProperties:
    """Tests for cached and regular properties on the interface."""

    def test_input_spec(self, engine) -> None:
        """input_spec returns list of (shape, dtype) tuples."""
        spec = engine.input_spec
        assert isinstance(spec, list)
        for shape, dtype in spec:
            assert isinstance(shape, list)
            assert isinstance(dtype, np.dtype)

    def test_input_shapes(self, engine) -> None:
        """input_shapes returns list of tuples."""
        shapes = engine.input_shapes
        assert isinstance(shapes, list)
        assert all(isinstance(s, tuple) for s in shapes)

    def test_output_spec(self, engine) -> None:
        """output_spec returns list of (shape, dtype) tuples."""
        spec = engine.output_spec
        assert isinstance(spec, list)
        for shape, dtype in spec:
            assert isinstance(shape, list)
            assert isinstance(dtype, np.dtype)

    def test_output_shapes(self, engine) -> None:
        """output_shapes returns list of tuples."""
        shapes = engine.output_shapes
        assert isinstance(shapes, list)
        assert all(isinstance(s, tuple) for s in shapes)

    def test_input_dtypes(self, engine) -> None:
        """input_dtypes returns list of numpy dtypes."""
        dtypes = engine.input_dtypes
        assert isinstance(dtypes, list)
        assert all(isinstance(d, np.dtype) for d in dtypes)

    def test_output_dtypes(self, engine) -> None:
        """output_dtypes returns list of numpy dtypes."""
        dtypes = engine.output_dtypes
        assert isinstance(dtypes, list)
        assert all(isinstance(d, np.dtype) for d in dtypes)

    def test_input_names(self, engine) -> None:
        """input_names returns list of strings."""
        names = engine.input_names
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)
        assert len(names) > 0

    def test_output_names(self, engine) -> None:
        """output_names returns list of strings."""
        names = engine.output_names
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)
        assert len(names) > 0

    def test_batch_size(self, engine) -> None:
        """batch_size returns an integer."""
        bs = engine.batch_size
        assert isinstance(bs, int)
        assert bs >= -1

    def test_is_dynamic_batch(self, engine) -> None:
        """is_dynamic_batch returns a bool."""
        assert isinstance(engine.is_dynamic_batch, bool)

    def test_memsize(self, engine) -> None:
        """Memsize returns a positive integer."""
        assert isinstance(engine.memsize, int)
        assert engine.memsize >= 0

    def test_engine_property(self, engine) -> None:
        """Engine property returns the raw TRT engine."""
        assert engine.engine is not None

    def test_context_property(self, engine) -> None:
        """Context property returns the execution context."""
        assert engine.context is not None

    def test_stream_property(self, engine) -> None:
        """Stream property returns the CUDA stream."""
        assert engine.stream is not None

    def test_input_bindings(self, engine) -> None:
        """input_bindings returns list of Binding objects."""
        bindings = engine.input_bindings
        assert isinstance(bindings, list)
        assert len(bindings) > 0

    def test_output_bindings(self, engine) -> None:
        """output_bindings returns list of Binding objects."""
        bindings = engine.output_bindings
        assert isinstance(bindings, list)
        assert len(bindings) > 0

    def test_cached_properties_consistent(self, engine) -> None:
        """Cached properties return same object on repeat access."""
        spec1 = engine.input_spec
        spec2 = engine.input_spec
        assert spec1 is spec2

        shapes1 = engine.output_shapes
        shapes2 = engine.output_shapes
        assert shapes1 is shapes2


@pytest.mark.gpu
class TestInterfaceDelCleanup:
    """Tests for __del__ cleanup."""

    def test_del_frees_bindings(self, engine_path) -> None:
        """__del__ frees bindings without error."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=False)
        del eng  # Should not crash

    def test_del_double_delete(self, engine_path) -> None:
        """Double deletion does not crash (via contextlib.suppress)."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=False)
        eng.__del__()
        del eng  # Second delete should not crash

    def test_del_deletes_context_engine(self, engine_path) -> None:
        """__del__ removes _context and _engine attributes."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=False)
        eng.__del__()
        assert not hasattr(eng, "_context")
        assert not hasattr(eng, "_engine")
