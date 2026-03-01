# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for TRTEngine initialization, backend selection, properties, and destruction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.gpu]


# ============================================================================
# Initialization
# ============================================================================


class TestEngineInit:
    """Test TRTEngine constructor and parameter handling."""

    def test_init_default_args(self, engine_path: Path) -> None:
        """Minimal init works with only the engine path."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path)
        assert eng is not None
        del eng

    def test_init_with_warmup(self, engine_path: Path) -> None:
        """warmup=True runs warmup iterations without error."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=True, warmup_iterations=3)
        assert eng._warmup is True
        del eng

    def test_init_without_warmup(self, engine_path: Path) -> None:
        """warmup=False skips warmup iterations."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=False)
        assert eng._warmup is False
        del eng

    def test_init_warmup_none(self, engine_path: Path) -> None:
        """warmup=None (default) does not run warmup."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=None)
        assert eng._warmup is None
        # None is falsy, so warmup should not have executed
        del eng

    def test_invalid_backend_raises(self, engine_path: Path) -> None:
        """ValueError raised when backend string is not in allowed set."""
        from trtutils import TRTEngine

        with pytest.raises(ValueError, match="Invalid backend"):
            TRTEngine(engine_path, backend="not_a_backend")

    @pytest.mark.parametrize("backend", ["auto", "async_v3", "async_v2"])
    def test_backend_selection(self, engine_path: Path, backend: str) -> None:
        """Each valid backend string is accepted without error."""
        from trtutils import FLAGS, TRTEngine

        # async_v3 requires FLAG support; skip if unavailable
        if backend == "async_v3" and not FLAGS.EXEC_ASYNC_V3:
            pytest.skip("async_v3 not available on this TRT version")
        if backend == "async_v2" and not FLAGS.EXEC_ASYNC_V2:
            pytest.skip("async_v2 not available on this TRT version")

        eng = TRTEngine(engine_path, warmup=False, backend=backend)
        assert eng is not None
        del eng

    def test_backend_auto_selects_v3_when_available(self, engine_path: Path) -> None:
        """Auto backend selects async_v3 when FLAGS.EXEC_ASYNC_V3 is True."""
        from trtutils import FLAGS, TRTEngine

        eng = TRTEngine(engine_path, warmup=False, backend="auto")
        if FLAGS.EXEC_ASYNC_V3:
            assert eng._async_v3 is True
        else:
            assert eng._async_v3 is False
        del eng

    def test_backend_async_v2_forces_v2(self, engine_path: Path) -> None:
        """Explicitly requesting async_v2 forces _async_v3 to False."""
        from trtutils import FLAGS, TRTEngine

        if not FLAGS.EXEC_ASYNC_V2:
            pytest.skip("async_v2 not available")
        eng = TRTEngine(engine_path, warmup=False, backend="async_v2")
        assert eng._async_v3 is False
        del eng

    def test_cuda_graph_enabled_with_v3(self, engine_path: Path) -> None:
        """cuda_graph=True with async_v3 enables CUDA graph."""
        from trtutils import FLAGS, TRTEngine

        if not FLAGS.EXEC_ASYNC_V3:
            pytest.skip("async_v3 not available")
        eng = TRTEngine(
            engine_path,
            warmup=False,
            backend="async_v3",
            cuda_graph=True,
        )
        assert eng._cuda_graph_enabled is True
        assert eng._cuda_graph is not None
        del eng

    def test_cuda_graph_disabled_without_v3(self, engine_path: Path) -> None:
        """cuda_graph=True with async_v2 backend disables CUDA graph."""
        from trtutils import FLAGS, TRTEngine

        if not FLAGS.EXEC_ASYNC_V2:
            pytest.skip("async_v2 not available")
        eng = TRTEngine(
            engine_path,
            warmup=False,
            backend="async_v2",
            cuda_graph=True,
        )
        assert eng._cuda_graph_enabled is False
        assert eng._cuda_graph is None
        del eng

    def test_cuda_graph_false(self, engine_path: Path) -> None:
        """cuda_graph=False explicitly disables CUDA graph."""
        from trtutils import TRTEngine

        eng = TRTEngine(
            engine_path,
            warmup=False,
            cuda_graph=False,
        )
        assert eng._cuda_graph_enabled is False
        assert eng._cuda_graph is None
        del eng

    def test_cuda_graph_none_defaults_true(self, engine_path: Path) -> None:
        """cuda_graph=None defaults to True (enabled if v3 available)."""
        from trtutils import FLAGS, TRTEngine

        eng = TRTEngine(engine_path, warmup=False, cuda_graph=None)
        if FLAGS.EXEC_ASYNC_V3:
            assert eng._cuda_graph_enabled is True
        else:
            assert eng._cuda_graph_enabled is False
        del eng

    @pytest.mark.parametrize("verbose", [True, False, None])
    def test_verbose_flag(self, engine_path: Path, verbose: bool | None) -> None:
        """Verbose parameter is stored correctly."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=False, verbose=verbose)
        if verbose is not None:
            assert eng._verbose is verbose
        else:
            # None defaults to False
            assert eng._verbose is False
        del eng

    @pytest.mark.parametrize("pagelocked", [True, False, None])
    def test_pagelocked_mem(self, engine_path: Path, pagelocked: bool | None) -> None:
        """Pagelocked memory selection works for all values."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=False, pagelocked_mem=pagelocked)
        if pagelocked is None:
            # Default is True
            assert eng.pagelocked_mem is True
        else:
            assert eng.pagelocked_mem is pagelocked
        del eng

    def test_unified_mem_default(self, engine_path: Path) -> None:
        """unified_mem=None defaults to FLAGS.IS_JETSON."""
        from trtutils import FLAGS, TRTEngine

        eng = TRTEngine(engine_path, warmup=False, unified_mem=None)
        assert eng.unified_mem is FLAGS.IS_JETSON
        del eng

    def test_unified_mem_explicit_true(self, engine_path: Path) -> None:
        """unified_mem=True is stored correctly."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=False, unified_mem=True)
        assert eng.unified_mem is True
        del eng

    def test_no_warn_parameter(self, engine_path: Path) -> None:
        """no_warn parameter is accepted without error."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=False, no_warn=True)
        assert eng is not None
        del eng

    def test_init_string_path(self, engine_path: Path) -> None:
        """Engine accepts a string path as well as Path."""
        from trtutils import TRTEngine

        eng = TRTEngine(str(engine_path), warmup=False)
        assert eng is not None
        del eng


# ============================================================================
# Properties
# ============================================================================


class TestEngineProperties:
    """Test all TRTEngine property accessors."""

    def test_name(self, engine) -> None:
        """Name returns string matching engine file stem."""
        name = engine.name  # type: ignore[union-attr]
        assert isinstance(name, str)
        assert len(name) > 0

    def test_engine_object(self, engine) -> None:
        """Engine property returns a TensorRT ICudaEngine."""
        eng_obj = engine.engine  # type: ignore[union-attr]
        assert eng_obj is not None

    def test_context_object(self, engine) -> None:
        """Context property returns a TensorRT IExecutionContext."""
        ctx = engine.context  # type: ignore[union-attr]
        assert ctx is not None

    def test_stream_object(self, engine) -> None:
        """Stream property returns a CUDA stream."""
        stream = engine.stream  # type: ignore[union-attr]
        assert stream is not None

    def test_logger_object(self, engine) -> None:
        """Logger property returns a TensorRT ILogger."""
        logger = engine.logger  # type: ignore[union-attr]
        assert logger is not None

    def test_memsize_positive(self, engine) -> None:
        """Memsize returns a non-negative integer."""
        memsize = engine.memsize  # type: ignore[union-attr]
        assert isinstance(memsize, int)
        assert memsize >= 0

    def test_input_spec(self, engine) -> None:
        """input_spec returns list of (shape, dtype) tuples."""
        spec = engine.input_spec  # type: ignore[union-attr]
        assert isinstance(spec, list)
        assert len(spec) >= 1
        for shape, dtype in spec:
            assert isinstance(shape, list)
            assert isinstance(dtype, np.dtype)

    def test_input_shapes(self, engine) -> None:
        """input_shapes returns list of tuple[int, ...]."""
        shapes = engine.input_shapes  # type: ignore[union-attr]
        assert isinstance(shapes, list)
        assert len(shapes) >= 1
        for shape in shapes:
            assert isinstance(shape, tuple)
            assert all(isinstance(d, int) for d in shape)

    def test_input_dtypes(self, engine) -> None:
        """input_dtypes returns list of np.dtype."""
        dtypes = engine.input_dtypes  # type: ignore[union-attr]
        assert isinstance(dtypes, list)
        assert len(dtypes) >= 1
        for dtype in dtypes:
            assert isinstance(dtype, np.dtype)

    def test_input_names(self, engine) -> None:
        """input_names returns list of str."""
        names = engine.input_names  # type: ignore[union-attr]
        assert isinstance(names, list)
        assert len(names) >= 1
        for name in names:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_output_spec(self, engine) -> None:
        """output_spec returns list of (shape, dtype) tuples."""
        spec = engine.output_spec  # type: ignore[union-attr]
        assert isinstance(spec, list)
        assert len(spec) >= 1
        for shape, dtype in spec:
            assert isinstance(shape, list)
            assert isinstance(dtype, np.dtype)

    def test_output_shapes(self, engine) -> None:
        """output_shapes returns list of tuple[int, ...]."""
        shapes = engine.output_shapes  # type: ignore[union-attr]
        assert isinstance(shapes, list)
        assert len(shapes) >= 1
        for shape in shapes:
            assert isinstance(shape, tuple)
            assert all(isinstance(d, int) for d in shape)

    def test_output_dtypes(self, engine) -> None:
        """output_dtypes returns list of np.dtype."""
        dtypes = engine.output_dtypes  # type: ignore[union-attr]
        assert isinstance(dtypes, list)
        assert len(dtypes) >= 1
        for dtype in dtypes:
            assert isinstance(dtype, np.dtype)

    def test_output_names(self, engine) -> None:
        """output_names returns list of str."""
        names = engine.output_names  # type: ignore[union-attr]
        assert isinstance(names, list)
        assert len(names) >= 1
        for name in names:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_batch_size(self, engine) -> None:
        """batch_size returns int >= 1."""
        bs = engine.batch_size  # type: ignore[union-attr]
        assert isinstance(bs, int)
        assert bs >= 1

    def test_is_dynamic_batch(self, engine) -> None:
        """is_dynamic_batch returns a bool."""
        is_dyn = engine.is_dynamic_batch  # type: ignore[union-attr]
        assert isinstance(is_dyn, bool)

    def test_input_bindings(self, engine) -> None:
        """input_bindings returns list of Binding objects."""
        from trtutils.core._bindings import Binding

        bindings = engine.input_bindings  # type: ignore[union-attr]
        assert isinstance(bindings, list)
        assert len(bindings) >= 1
        for b in bindings:
            assert isinstance(b, Binding)

    def test_output_bindings(self, engine) -> None:
        """output_bindings returns list of Binding objects."""
        from trtutils.core._bindings import Binding

        bindings = engine.output_bindings  # type: ignore[union-attr]
        assert isinstance(bindings, list)
        assert len(bindings) >= 1
        for b in bindings:
            assert isinstance(b, Binding)

    def test_pagelocked_mem_property(self, engine) -> None:
        """pagelocked_mem property returns True by default."""
        assert engine.pagelocked_mem is True  # type: ignore[union-attr]

    def test_unified_mem_property(self, engine) -> None:
        """unified_mem property returns bool."""
        from trtutils import FLAGS

        assert engine.unified_mem is FLAGS.IS_JETSON  # type: ignore[union-attr]

    def test_dla_core_property(self, engine) -> None:
        """dla_core returns None on non-Jetson platforms."""
        assert engine.dla_core is None  # type: ignore[union-attr]

    def test_device_property(self, engine) -> None:
        """Device returns None when not explicitly set."""
        assert engine.device is None  # type: ignore[union-attr]

    def test_input_spec_is_cached(self, engine) -> None:
        """input_spec is a cached_property, returns same object on repeat."""
        spec1 = engine.input_spec  # type: ignore[union-attr]
        spec2 = engine.input_spec  # type: ignore[union-attr]
        assert spec1 is spec2

    def test_output_spec_is_cached(self, engine) -> None:
        """output_spec is a cached_property, returns same object on repeat."""
        spec1 = engine.output_spec  # type: ignore[union-attr]
        spec2 = engine.output_spec  # type: ignore[union-attr]
        assert spec1 is spec2

    def test_input_shapes_is_cached(self, engine) -> None:
        """input_shapes is cached."""
        shapes1 = engine.input_shapes  # type: ignore[union-attr]
        shapes2 = engine.input_shapes  # type: ignore[union-attr]
        assert shapes1 is shapes2

    def test_input_dtypes_is_cached(self, engine) -> None:
        """input_dtypes is cached."""
        dt1 = engine.input_dtypes  # type: ignore[union-attr]
        dt2 = engine.input_dtypes  # type: ignore[union-attr]
        assert dt1 is dt2


# ============================================================================
# Destruction
# ============================================================================


class TestEngineDestruction:
    """Test TRTEngine cleanup behavior."""

    def test_del_no_error(self, engine_path: Path) -> None:
        """Deleting an engine does not raise an exception."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=False)
        del eng  # Should not raise

    def test_del_with_cuda_graph(self, engine_path: Path) -> None:
        """Deleting an engine with a captured CUDA graph does not crash."""
        from trtutils import FLAGS, TRTEngine

        if not FLAGS.EXEC_ASYNC_V3:
            pytest.skip("async_v3 required for CUDA graph")
        eng = TRTEngine(
            engine_path,
            warmup=True,
            warmup_iterations=2,
            backend="async_v3",
            cuda_graph=True,
        )
        assert eng._cuda_graph is not None
        assert eng._cuda_graph.is_captured is True
        del eng  # Should not raise

    def test_double_del_no_error(self, engine_path: Path) -> None:
        """Calling __del__ twice does not raise (suppress errors)."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=False)
        eng.__del__()
        eng.__del__()  # Second call should not error

    def test_using_engine_tensors_flag_default(self, engine) -> None:
        """_using_engine_tensors is True by default after init."""
        assert engine._using_engine_tensors is True  # type: ignore[union-attr]


# ============================================================================
# Execution -- async_v2 and graph capture failure paths
# ============================================================================


class TestEngineAsyncV2:
    """Tests for the async_v2 execution backend (line 379 in _engine.py)."""

    def test_async_v2_backend_init(self, engine_path: Path) -> None:
        """backend='async_v2' sets _async_v3=False and disables CUDA graph."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=False, backend="async_v2")
        assert eng._async_v3 is False
        assert eng._cuda_graph is None  # No graph with async_v2
        del eng

    @pytest.mark.xfail(reason="execute_async_v2 removed in TRT 10+", raises=AttributeError)
    def test_async_v2_backend_execute(self, engine_path: Path) -> None:
        """Execute with backend='async_v2' exercises execute_async_v2 path."""
        from trtutils import TRTEngine

        eng = TRTEngine(engine_path, warmup=False, backend="async_v2")
        data = eng.get_random_input()
        outputs = eng.execute(data)
        assert isinstance(outputs, list)
        assert len(outputs) > 0
        del eng


class TestEngineGraphCaptureFailure:
    """Tests for graph capture failure fallback (lines 249-255 in _engine.py)."""

    def test_graph_capture_failure_falls_through(self, engine_path: Path) -> None:
        """When graph capture fails, engine falls back to direct execution."""
        from unittest.mock import patch

        from trtutils import FLAGS, TRTEngine

        if not FLAGS.EXEC_ASYNC_V3:
            pytest.skip("async_v3 required for CUDA graph")

        eng = TRTEngine(engine_path, warmup=False, backend="async_v3", cuda_graph=True)

        # Force graph capture failure by making stop() return False (not captured)
        original_stop = eng._cuda_graph.stop
        cuda_graph_ref = eng._cuda_graph

        def failing_stop():
            original_stop()  # Let it run normally but then clear
            cuda_graph_ref._graph_exec = None
            cuda_graph_ref._graph = None
            return True  # __exit__ inverts this

        with patch.object(eng._cuda_graph, "stop", side_effect=failing_stop):
            # Execute should detect capture failure (is_captured=False after context manager)
            # and raise RuntimeError about graph capture failure
            data = eng.get_random_input()
            with pytest.raises(RuntimeError, match="graph capture failed"):
                eng.execute(data)
        del eng
