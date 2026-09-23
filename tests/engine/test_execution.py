# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for TRTEngine execution methods -- execute, raw_exec, graph_exec, mock_execute."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils._flags import FLAGS
from trtutils.core import Buffer, MemoryLocation, stream_synchronize


def _to_device(buffers: list[Buffer]) -> list[Buffer]:
    """Copy host Buffers into new device Buffers."""
    return [Buffer.from_array(b.array, MemoryLocation.DEVICE) for b in buffers]


class TestExecute:
    """Tests for TRTEngine.execute()."""

    def test_execute_output_matches_spec(self, engine, random_input) -> None:
        """execute() output types, shapes, and dtypes match engine output_spec."""
        outputs = engine.execute(random_input)
        assert isinstance(outputs, list)
        assert len(outputs) == len(engine.output_spec)
        for out, (shape, dtype) in zip(outputs, engine.output_spec):
            assert isinstance(out, np.ndarray)
            assert list(out.shape) == shape
            assert out.dtype == dtype

    def test_execute_deterministic(self, engine, random_input) -> None:
        """Same input produces same output across two runs."""
        out1 = engine.execute(random_input)
        out2 = engine.execute(random_input)
        for o1, o2 in zip(out1, out2):
            np.testing.assert_array_equal(o1, o2)

    @pytest.mark.parametrize(
        "no_copy",
        [
            pytest.param(True, id="no-copy-true"),
            pytest.param(False, id="no-copy-false"),
            pytest.param(None, id="no-copy-none"),
        ],
    )
    def test_execute_no_copy(self, engine, random_input, no_copy) -> None:
        """execute() works with all no_copy values."""
        outputs = engine.execute(random_input, no_copy=no_copy)
        assert isinstance(outputs, list)
        assert len(outputs) >= 1

    @pytest.mark.parametrize(
        "debug",
        [
            pytest.param(True, id="debug-true"),
            pytest.param(False, id="debug-false"),
            pytest.param(None, id="debug-none"),
        ],
    )
    def test_execute_debug(self, engine, random_input, debug) -> None:
        """execute() completes without error for all debug settings."""
        outputs = engine.execute(random_input, debug=debug)
        assert outputs is not None

    def test_execute_pagelocked(self, engine, random_input) -> None:
        """Pagelocked memory (default) produces correct outputs."""
        assert engine.pagelocked_mem is True
        outputs = engine.execute(random_input)
        assert outputs is not None
        assert len(outputs) >= 1

    def test_execute_no_pagelocked(self, engine_no_pagelocked) -> None:
        """Non-pagelocked memory path produces correct outputs."""
        assert engine_no_pagelocked.pagelocked_mem is False
        rand_input = engine_no_pagelocked.get_random_input()
        outputs = engine_no_pagelocked.execute(rand_input)
        assert outputs is not None
        assert len(outputs) >= 1

    def test_execute_unified_mem(self, make_engine) -> None:
        """Unified memory mode produces correct outputs."""
        eng = make_engine(pagelocked_mem=True, unified_mem=True)
        rand_input = eng.get_random_input()
        outputs = eng.execute(rand_input)
        assert outputs is not None
        assert len(outputs) >= 1

    def test_execute_parity_pagelocked_vs_not(self, make_engine) -> None:
        """Pagelocked vs non-pagelocked produce equivalent outputs."""
        eng_pl = make_engine(pagelocked_mem=True)
        eng_no_pl = make_engine(pagelocked_mem=False)
        rand_input = eng_pl.get_random_input()
        out_pl = eng_pl.execute(rand_input)
        out_no_pl = eng_no_pl.execute(rand_input)
        for o1, o2 in zip(out_pl, out_no_pl):
            assert np.allclose(o1, o2)

    def test_execute_parity_unified_vs_default(self, make_engine) -> None:
        """Unified memory vs default produce equivalent outputs."""
        eng_default = make_engine()
        eng_unified = make_engine(pagelocked_mem=True, unified_mem=True)
        rand_input = eng_default.get_random_input()
        out_default = eng_default.execute(rand_input)
        out_unified = eng_unified.execute(rand_input)
        for o1, o2 in zip(out_default, out_unified):
            assert np.allclose(o1, o2)

    def test_execute_async_v2_backend(self, make_engine) -> None:
        """Execute succeeds with explicit async_v2 backend."""
        if not FLAGS.EXEC_ASYNC_V2:
            pytest.skip("async_v2 not available in this TensorRT version")
        eng = make_engine(backend="async_v2")
        assert eng._async_v3 is False
        rand_input = eng.get_random_input()
        outputs = eng.execute(rand_input)
        assert isinstance(outputs, list)
        assert len(outputs) >= 1

    def test_execute_async_v2_parity(self, make_engine) -> None:
        """async_v2 and auto backends produce equivalent outputs."""
        if not FLAGS.EXEC_ASYNC_V2:
            pytest.skip("async_v2 not available in this TensorRT version")
        eng_auto = make_engine(backend="auto", cuda_graph=False)
        eng_v2 = make_engine(backend="async_v2")
        rand_input = eng_auto.get_random_input()
        out_auto = eng_auto.execute(rand_input)
        out_v2 = eng_v2.execute(rand_input)
        for o1, o2 in zip(out_auto, out_v2):
            assert np.allclose(o1, o2)

    def test_no_copy_true_returns_same_buffer(self, engine, random_input) -> None:
        """no_copy=True returns the internal host allocation buffers."""
        outputs = engine.execute(random_input, no_copy=True)
        for out, binding in zip(outputs, engine.output_bindings):
            assert out is binding.host_allocation

    def test_no_copy_false_returns_copy(self, engine, random_input) -> None:
        """no_copy=False returns independent copies."""
        outputs = engine.execute(random_input, no_copy=False)
        for out, binding in zip(outputs, engine.output_bindings):
            assert out is not binding.host_allocation
            assert np.array_equal(out, binding.host_allocation)

    def test_no_copy_none_defaults_to_copy(self, engine, random_input) -> None:
        """Default (None) behavior is to copy."""
        outputs = engine.execute(random_input)
        for out, binding in zip(outputs, engine.output_bindings):
            assert out is not binding.host_allocation

    def test_no_copy_true_overwritten_by_next_exec(self, engine, random_input) -> None:
        """no_copy=True buffers are overwritten by next execute call."""
        outputs_first = engine.execute(random_input, no_copy=True)
        [o.copy() for o in outputs_first]

        new_input = engine.get_random_input(new=True)
        engine.execute(new_input, no_copy=True)

        for out_ref, binding in zip(outputs_first, engine.output_bindings):
            assert out_ref is binding.host_allocation


class TestDeviceInputs:
    """Tests for TRTEngine.execute() with device-resident Buffers."""

    def test_device_inputs_match_host_inputs(self, make_engine) -> None:
        """Device and host Buffers holding the same data produce the same outputs."""
        eng = make_engine()
        host = eng.get_random_input()
        device = _to_device(host)
        out_device = eng.execute(device)
        out_host = eng.execute(host)
        for od, oh in zip(out_device, out_host):
            np.testing.assert_array_equal(od, oh)

    def test_device_inputs_bound_in_place(self, make_engine) -> None:
        """Aligned device Buffers are read by TensorRT directly, without a copy."""
        if not FLAGS.EXEC_ASYNC_V3:
            pytest.skip("Only relevant for async_v3 backend")
        eng = make_engine(backend="async_v3")
        device = _to_device(eng.get_random_input())
        eng.execute(device)
        assert eng._input_addresses == [b.ptr for b in device]

    def test_host_after_device_rebinds_engine_memory(self, make_engine) -> None:
        """A host call after a device call points the context back at the engine bindings."""
        eng = make_engine()
        host = eng.get_random_input()
        eng.execute(_to_device(host))
        eng.execute(host)
        assert eng._input_addresses == [b.allocation for b in eng.input_bindings]

    @pytest.mark.parametrize(
        ("pagelocked_mem", "unified_mem"),
        [
            pytest.param(True, False, id="pagelocked"),
            pytest.param(False, False, id="no-pagelocked"),
            pytest.param(True, True, id="unified"),
        ],
    )
    def test_device_inputs_memory_mode(self, make_engine, pagelocked_mem, unified_mem) -> None:
        """Device inputs work for every host memory mode."""
        eng = make_engine(pagelocked_mem=pagelocked_mem, unified_mem=unified_mem)
        host = eng.get_random_input()
        out_device = eng.execute(_to_device(host))
        out_host = eng.execute(host)
        for od, oh in zip(out_device, out_host):
            np.testing.assert_allclose(od, oh)

    def test_misaligned_device_input_is_staged(self, make_engine) -> None:
        """A device view TensorRT cannot address directly is copied into the binding."""
        eng = make_engine()
        host = eng.get_random_input()
        misaligned: list[Buffer] = []
        for buffer in host:
            backing = Buffer.empty((buffer.size + 1,), buffer.dtype, MemoryLocation.DEVICE)
            view = backing[1:].reshape(buffer.shape)
            view.copy_from(buffer)
            misaligned.append(view)
        out_misaligned = eng.execute(misaligned)
        assert eng._input_addresses == [b.allocation for b in eng.input_bindings]
        for om, oh in zip(out_misaligned, eng.execute(host)):
            np.testing.assert_array_equal(om, oh)

    def test_cuda_array_interface_input(self, make_engine) -> None:
        """Objects exposing __cuda_array_interface__ can be wrapped as inputs."""

        class _Foreign:
            def __init__(self, buffer: Buffer) -> None:
                self._buffer = buffer
                self.__cuda_array_interface__ = buffer.__cuda_array_interface__

        eng = make_engine()
        host = eng.get_random_input()
        wrapped = [Buffer.wrap(_Foreign(b)) for b in _to_device(host)]
        for ow, oh in zip(eng.execute(wrapped), eng.execute(host)):
            np.testing.assert_array_equal(ow, oh)


class TestInputValidation:
    """Tests for the checks execute() applies to its inputs."""

    def test_ndarray_input_raises(self, engine, random_input) -> None:
        """Plain numpy arrays are rejected with a pointer to Buffer.wrap."""
        with pytest.raises(TypeError, match=r"Buffer\.wrap"):
            engine.execute([b.array for b in random_input])

    def test_wrong_input_count_raises(self, engine, random_input) -> None:
        """A different number of inputs than the engine has raises ValueError."""
        with pytest.raises(ValueError, match="expects"):
            engine.execute([*random_input, *random_input])

    def test_wrong_dtype_raises(self, engine, random_input) -> None:
        """An input with the wrong dtype raises ValueError."""
        bad = [Buffer.wrap(b.array.astype(np.float64)) for b in random_input]
        with pytest.raises(ValueError, match="dtype"):
            engine.execute(bad)

    def test_wrong_shape_raises_and_leaves_engine_usable(self, engine, random_input) -> None:
        """A static engine rejects other shapes without changing its state."""
        bad = [Buffer.wrap(np.zeros((2, *b.shape), dtype=b.dtype)) for b in random_input]
        with pytest.raises(ValueError, match="shape"):
            engine.execute(bad)
        assert engine.active_input_shapes == [tuple(b.shape) for b in engine.input_bindings]
        assert engine.execute(random_input) is not None


class TestRawExec:
    """Tests for TRTEngine.raw_exec()."""

    def test_raw_exec_returns_device_output_views(self, engine, random_input) -> None:
        """raw_exec returns device Buffers over the output bindings."""
        outputs = engine.raw_exec(random_input)
        stream_synchronize(engine.stream)
        assert len(outputs) == len(engine.output_bindings)
        for out, binding in zip(outputs, engine.output_bindings):
            assert isinstance(out, Buffer)
            assert out.is_device
            assert out.ptr == binding.allocation
            assert list(out.shape) == binding.shape

    def test_raw_exec_matches_execute(self, engine, random_input) -> None:
        """raw_exec outputs hold the same values execute returns."""
        outputs = engine.raw_exec(_to_device(random_input))
        stream_synchronize(engine.stream)
        raw = [o.numpy() for o in outputs]
        for r, e in zip(raw, engine.execute(random_input)):
            np.testing.assert_array_equal(r, e)

    def test_execute_after_raw_exec_rebinds(self, make_engine) -> None:
        """execute() with host inputs after raw_exec() with device inputs rebinds the engine memory."""
        eng = make_engine()
        host = eng.get_random_input()
        eng.raw_exec(_to_device(host))
        stream_synchronize(eng.stream)
        eng.execute(host)
        assert eng._input_addresses == [b.allocation for b in eng.input_bindings]


class TestMockExecute:
    """Tests for mock_execute(), warmup(), get_random_input(), and __call__."""

    def test_get_random_input_matches_spec(self, engine) -> None:
        """get_random_input returns host Buffers matching engine input spec shapes and dtypes."""
        data = engine.get_random_input()
        assert isinstance(data, list)
        assert len(data) == len(engine.input_spec)
        for buffer, (shape, dtype) in zip(data, engine.input_spec):
            assert isinstance(buffer, Buffer)
            assert buffer.is_host
            assert list(buffer.shape) == shape
            assert buffer.dtype == dtype

    def test_cached_returns_same(self, engine) -> None:
        """Default (new=None) returns cached data on second call."""
        data1 = engine.get_random_input()
        data2 = engine.get_random_input()
        for a, b in zip(data1, data2):
            np.testing.assert_array_equal(a, b)

    def test_new_true_generates_fresh(self, engine) -> None:
        """new=True generates fresh random data."""
        engine.get_random_input()
        data2 = engine.get_random_input(new=True)
        assert isinstance(data2, list)

    def test_mock_execute_no_data(self, engine) -> None:
        """mock_execute(data=None) generates random input internally."""
        result = engine.mock_execute()
        assert isinstance(result, list)
        assert all(isinstance(a, np.ndarray) for a in result)

    def test_mock_execute_with_data(self, engine, random_input) -> None:
        """mock_execute(data=...) uses provided data."""
        result = engine.mock_execute(data=random_input)
        assert isinstance(result, list)

    def test_mock_execute_returns_no_copy(self, engine) -> None:
        """mock_execute uses no_copy=True internally."""
        result = engine.mock_execute()
        assert isinstance(result, list)

    @pytest.mark.parametrize(
        "iterations",
        [
            pytest.param(1, id="single-iteration"),
            pytest.param(3, id="multiple-iterations"),
        ],
    )
    def test_warmup_runs(self, engine, iterations) -> None:
        """warmup(iterations=N) runs N mock executions."""
        engine.warmup(iterations)

    def test_call_delegates_to_execute(self, engine, random_input) -> None:
        """__call__ delegates to execute()."""
        result = engine(random_input)
        assert isinstance(result, list)
        assert all(isinstance(a, np.ndarray) for a in result)

    def test_call_no_copy(self, engine, random_input) -> None:
        """__call__ with no_copy=True."""
        result = engine(random_input, no_copy=True)
        assert isinstance(result, list)


@pytest.mark.cuda_graph
class TestGraphExec:
    """Tests for CUDA graph capture, replay, edge cases, and bypass."""

    def test_first_execute_captures_graph(self, make_engine) -> None:
        """cuda_graph=True + first execute() triggers graph capture."""
        eng = make_engine(cuda_graph=True)
        data = eng.get_random_input()
        eng.execute(data)
        if eng._cuda_graph is not None:
            assert eng._cuda_graph.is_captured

    def test_graph_exec_after_capture(self, make_engine) -> None:
        """After warmup with cuda_graph=True, graph_exec succeeds."""
        eng = make_engine(warmup=True, warmup_iterations=2)
        if eng._cuda_graph is not None and eng._cuda_graph.is_captured:
            eng.graph_exec()

    def test_graph_exec_without_capture_raises(self, make_engine) -> None:
        """graph_exec raises RuntimeError when no graph is captured."""
        eng = make_engine(cuda_graph=False)
        with pytest.raises(RuntimeError, match="No CUDA graph captured"):
            eng.graph_exec()

    def test_graph_exec_deterministic(self, make_engine) -> None:
        """Graph replay produces consistent output."""
        eng = make_engine(warmup=True, warmup_iterations=2)
        if eng._cuda_graph is None or not eng._cuda_graph.is_captured:
            pytest.skip("CUDA graph not captured")
        data = eng.get_random_input()
        out1 = eng.execute(data)
        out2 = eng.execute(data)
        for o1, o2 in zip(out1, out2):
            np.testing.assert_array_equal(o1, o2)

    def test_graph_invalidate_and_recapture(self, make_engine) -> None:
        """Invalidating graph allows recapture on next execute."""
        eng = make_engine(warmup=True, warmup_iterations=2)
        if eng._cuda_graph is None or not eng._cuda_graph.is_captured:
            pytest.skip("CUDA graph not captured")
        eng._cuda_graph.invalidate()
        assert not eng._cuda_graph.is_captured
        data = eng.get_random_input()
        result = eng.execute(data)
        assert isinstance(result, list)
        assert eng._cuda_graph.is_captured

    def test_graph_exec_debug(self, make_engine) -> None:
        """graph_exec(debug=True) synchronizes the stream."""
        eng = make_engine(warmup=True, warmup_iterations=2)
        if eng._cuda_graph is not None and eng._cuda_graph.is_captured:
            eng.graph_exec(debug=True)

    def test_device_inputs_bypass_graph(self, make_engine) -> None:
        """Device inputs run without the graph, which stays valid for host calls."""
        eng = make_engine(warmup=True, warmup_iterations=2)
        if eng._cuda_graph is None or not eng._cuda_graph.is_captured:
            pytest.skip("CUDA graph not captured")
        host = eng.get_random_input()
        expected = eng.execute(host)
        out_device = eng.execute(_to_device(host))
        assert eng._cuda_graph.is_captured
        for od, oe in zip(out_device, expected):
            np.testing.assert_array_equal(od, oe)
        for oh, oe in zip(eng.execute(host), expected):
            np.testing.assert_array_equal(oh, oe)

    def test_execute_with_cuda_graph_full_flow(self, make_engine) -> None:
        """Full flow: init -> execute (capture) -> execute (replay)."""
        eng = make_engine(cuda_graph=True)
        data = eng.get_random_input()
        out1 = eng.execute(data)
        assert isinstance(out1, list)
        out2 = eng.execute(data)
        assert isinstance(out2, list)
        for o1, o2 in zip(out1, out2):
            np.testing.assert_array_equal(o1, o2)

    def test_execute_second_call_replays_graph(self, make_engine) -> None:
        """Second execute() replays the captured graph."""
        eng = make_engine(warmup=True, warmup_iterations=2)
        data = eng.get_random_input()
        result = eng.execute(data)
        assert isinstance(result, list)

    def test_execute_no_copy_with_graph(self, make_engine) -> None:
        """no_copy=True works correctly with CUDA graph path."""
        eng = make_engine(warmup=True, warmup_iterations=2)
        if eng._cuda_graph is None or not eng._cuda_graph.is_captured:
            pytest.skip("CUDA graph not captured")
        data = eng.get_random_input()
        outputs = eng.execute(data, no_copy=True)
        for out, binding in zip(outputs, eng.output_bindings):
            assert out is binding.host_allocation

    def test_graph_exec_no_default_sync(self, make_engine) -> None:
        """graph_exec() without debug does not add extra sync."""
        eng = make_engine(warmup=True, warmup_iterations=2)
        if eng._cuda_graph is not None and eng._cuda_graph.is_captured:
            eng.graph_exec()

    def test_cuda_graph_disabled(self, make_engine) -> None:
        """cuda_graph=False disables graph capture entirely."""
        eng = make_engine(cuda_graph=False)
        assert eng._cuda_graph is None
        data = eng.get_random_input()
        result = eng.execute(data)
        assert isinstance(result, list)

    def test_cuda_graph_with_async_v2(self, make_engine) -> None:
        """cuda_graph=True + async_v2 backend disables graph."""
        eng = make_engine(backend="async_v2", cuda_graph=True)
        assert not eng._cuda_graph_enabled
        assert eng._cuda_graph is None

    def test_capture_cuda_graph_none_raises(self, make_engine) -> None:
        """_capture_cuda_graph raises RuntimeError when _cuda_graph is None."""
        eng = make_engine(cuda_graph=True)
        saved = eng._cuda_graph
        eng._cuda_graph = None
        with pytest.raises(RuntimeError, match="CUDA graph is not enabled"):
            eng._capture_cuda_graph()
        eng._cuda_graph = saved

    def test_raw_exec_bypasses_graph_capture(self, make_engine) -> None:
        """raw_exec() does not trigger CUDA graph capture."""
        eng = make_engine(cuda_graph=True)
        if eng._cuda_graph is None:
            pytest.skip("CUDA graph not enabled")
        assert eng._cuda_graph.is_captured is False

        outputs = eng.raw_exec(eng.get_random_input())
        stream_synchronize(eng.stream)
        assert outputs is not None
        assert eng._cuda_graph.is_captured is False

        eng.execute(eng.get_random_input())
        assert eng._cuda_graph.is_captured is True
