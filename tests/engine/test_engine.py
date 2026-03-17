# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for TRTEngine initialization, backend selection, properties, and destruction."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from trtutils import TRTEngine
from trtutils._flags import FLAGS
from trtutils.core._bindings import Binding

from .conftest import ENGINE_PATHS


def test_init_default_args(engine_path) -> None:
    """Minimal init works with only the engine path."""
    eng = TRTEngine(engine_path)
    assert eng is not None
    del eng


@pytest.mark.parametrize(
    ("warmup", "expected"),
    [
        pytest.param(True, True, id="warmup-true"),
        pytest.param(False, False, id="warmup-false"),
        pytest.param(None, None, id="warmup-none"),
    ],
)
def test_init_warmup(engine_path, warmup, expected) -> None:
    """_warmup attribute matches the constructor argument."""
    eng = TRTEngine(engine_path, warmup=warmup, warmup_iterations=3)
    assert eng._warmup is expected
    del eng


@pytest.mark.parametrize(
    ("backend", "cuda_graph", "expect_enabled"),
    [
        pytest.param("async_v3", True, True, id="v3-graph-true"),
        pytest.param("async_v2", True, False, id="v2-graph-true"),
        pytest.param("auto", False, False, id="auto-graph-false"),
        pytest.param("auto", None, None, id="auto-graph-none"),
    ],
)
def test_cuda_graph_init(engine_path, backend, cuda_graph, expect_enabled) -> None:
    """cuda_graph enablement depends on backend and explicit setting."""
    if backend == "async_v3" and not FLAGS.EXEC_ASYNC_V3:
        pytest.skip("async_v3 not available")
    if backend == "async_v2" and not FLAGS.EXEC_ASYNC_V2:
        pytest.skip("async_v2 not available")
    eng = TRTEngine(engine_path, warmup=False, backend=backend, cuda_graph=cuda_graph)
    if expect_enabled is None:
        if FLAGS.EXEC_ASYNC_V3:
            assert eng._cuda_graph_enabled is True
        else:
            assert eng._cuda_graph_enabled is False
    elif expect_enabled:
        assert eng._cuda_graph_enabled is True
        assert eng._cuda_graph is not None
    else:
        assert eng._cuda_graph_enabled is False
    del eng


@pytest.mark.parametrize(
    ("unified_mem", "expected"),
    [
        pytest.param(None, None, id="unified-default"),
        pytest.param(True, True, id="unified-true"),
    ],
)
def test_unified_mem(engine_path, unified_mem, expected) -> None:
    """unified_mem parameter is stored correctly."""
    eng = TRTEngine(engine_path, warmup=False, unified_mem=unified_mem)
    if expected is None:
        assert eng.unified_mem is FLAGS.IS_JETSON
    else:
        assert eng.unified_mem is expected
    del eng


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("auto", id="backend-auto"),
        pytest.param("async_v3", id="backend-v3"),
        pytest.param("async_v2", id="backend-v2"),
    ],
)
def test_backend_selection(engine_path, backend) -> None:
    """Engine initializes with a given backend string."""
    if backend == "async_v3" and not FLAGS.EXEC_ASYNC_V3:
        pytest.skip("async_v3 not available")
    if backend == "async_v2" and not FLAGS.EXEC_ASYNC_V2:
        pytest.skip("async_v2 not available")
    eng = TRTEngine(engine_path, warmup=False, backend=backend)
    assert eng is not None
    del eng


@pytest.mark.parametrize(
    ("pagelocked", "expected"),
    [
        pytest.param(True, True, id="pagelocked-true"),
        pytest.param(False, False, id="pagelocked-false"),
        pytest.param(None, True, id="pagelocked-default"),
    ],
)
def test_pagelocked_mem(engine_path, pagelocked, expected) -> None:
    """Pagelocked memory setting is stored correctly."""
    eng = TRTEngine(engine_path, warmup=False, pagelocked_mem=pagelocked)
    assert eng.pagelocked_mem is expected
    del eng


def test_memsize(engine_path) -> None:
    """Memsize returns a non-negative integer."""
    eng = TRTEngine(engine_path, warmup=False)
    assert isinstance(eng.memsize, int)
    assert eng.memsize >= 0
    del eng


def test_del_no_error(engine_path) -> None:
    """Deleting an engine does not raise an exception."""
    eng = TRTEngine(engine_path, warmup=False)
    del eng


def test_del_with_cuda_graph(engine_path) -> None:
    """Deleting an engine with a captured CUDA graph does not crash."""
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
    del eng


def test_double_del_no_error(engine_path) -> None:
    """Calling __del__ twice does not raise."""
    eng = TRTEngine(engine_path, warmup=False)
    eng.__del__()
    eng.__del__()


def test_using_engine_tensors_flag_default(engine) -> None:
    """_using_engine_tensors is True by default after init."""
    assert engine._using_engine_tensors is True


def test_async_v2_backend_init(engine_path) -> None:
    """backend='async_v2' sets _async_v3=False and disables CUDA graph."""
    eng = TRTEngine(engine_path, warmup=False, backend="async_v2")
    assert eng._async_v3 is False
    assert eng._cuda_graph is None
    del eng


def test_async_v2_backend_execute(engine_path) -> None:
    """Execute with backend='async_v2' exercises execute_async_v2 path."""
    if not FLAGS.EXEC_ASYNC_V2:
        pytest.skip("async_v2 not available")
    eng = TRTEngine(engine_path, warmup=False, backend="async_v2")
    data = eng.get_random_input()
    outputs = eng.execute(data)
    assert isinstance(outputs, list)
    assert len(outputs) > 0
    del eng


@pytest.mark.cuda_graph
def test_graph_capture_failure_falls_through(engine_path) -> None:
    """When graph capture fails, engine falls back to direct execution."""
    if not FLAGS.EXEC_ASYNC_V3:
        pytest.skip("async_v3 required for CUDA graph")

    eng = TRTEngine(engine_path, warmup=False, backend="async_v3", cuda_graph=True)

    original_stop = eng._cuda_graph.stop
    cuda_graph_ref = eng._cuda_graph

    def failing_stop():
        original_stop()
        cuda_graph_ref._graph_exec = None
        cuda_graph_ref._graph = None
        return True

    with patch.object(eng._cuda_graph, "stop", side_effect=failing_stop):
        data = eng.get_random_input()
        with pytest.raises(RuntimeError, match="graph capture failed"):
            eng.execute(data)
    del eng


@pytest.mark.parametrize("engine_path", ENGINE_PATHS)
class TestEngine:
    """Property and accessor tests parametrized on engine path."""

    @pytest.fixture(autouse=True)
    def _setup_engine(self, engine_path) -> None:
        self.engine = TRTEngine(engine_path, warmup=False)
        yield
        del self.engine

    def test_spec_properties(self) -> None:
        """Spec properties return correctly typed lists of (shape, dtype) tuples."""
        for spec in (self.engine.input_spec, self.engine.output_spec):
            assert isinstance(spec, list)
            assert len(spec) >= 1
            for shape, dtype in spec:
                assert isinstance(shape, list)
                assert isinstance(dtype, np.dtype)

        for shapes in (self.engine.input_shapes, self.engine.output_shapes):
            assert isinstance(shapes, list)
            assert len(shapes) >= 1
            for shape in shapes:
                assert isinstance(shape, tuple)
                assert all(isinstance(d, int) for d in shape)

        for dtypes in (self.engine.input_dtypes, self.engine.output_dtypes):
            assert isinstance(dtypes, list)
            assert len(dtypes) >= 1
            for dtype in dtypes:
                assert isinstance(dtype, np.dtype)

    def test_name_properties(self) -> None:
        """Name properties return non-empty string lists."""
        for names in (self.engine.input_names, self.engine.output_names):
            assert isinstance(names, list)
            assert len(names) >= 1
            for name in names:
                assert isinstance(name, str)
                assert len(name) > 0

    def test_binding_properties(self) -> None:
        """Binding properties return lists of Binding objects."""
        for bindings in (self.engine.input_bindings, self.engine.output_bindings):
            assert isinstance(bindings, list)
            assert len(bindings) >= 1
            for b in bindings:
                assert isinstance(b, Binding)

    def test_scalar_properties(self) -> None:
        """Scalar properties return expected types and values."""
        assert isinstance(self.engine.name, str)
        assert len(self.engine.name) > 0

        assert isinstance(self.engine.batch_size, int)
        assert self.engine.batch_size >= 1

        assert isinstance(self.engine.is_dynamic_batch, bool)

        assert isinstance(self.engine.memsize, int)
        assert self.engine.memsize >= 0

        assert self.engine.unified_mem is FLAGS.IS_JETSON
        assert self.engine.dla_core is None
        assert self.engine.device is None

    def test_cached_properties(self) -> None:
        """Cached properties return the same object on repeated access."""
        assert self.engine.input_spec is self.engine.input_spec
        assert self.engine.output_spec is self.engine.output_spec
        assert self.engine.input_shapes is self.engine.input_shapes
        assert self.engine.input_dtypes is self.engine.input_dtypes
