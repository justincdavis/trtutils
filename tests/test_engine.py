# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: SLF001
from __future__ import annotations

import time
from threading import Thread
from unittest.mock import patch

import numpy as np
import pytest

import trtutils
from trtutils.core import (
    allocate_to_device,
    free_device_ptrs,
    memcpy_device_to_host,
    stream_synchronize,
)

from .common import build_engine

pytestmark = [pytest.mark.gpu]

NUM_ENGINES = 4
NUM_ITERS = 1_000


# ============================================================================
# Basic Engine Execution
# ============================================================================


def test_engine_run() -> None:
    """Test basic engine execution with mock data."""
    engine_path = build_engine()

    engine = trtutils.TRTEngine(
        engine_path,
        warmup=False,
    )

    outputs = engine.mock_execute()

    assert outputs is not None


def test_engine_run_backends() -> None:
    """Test each backend of the engine."""
    engine_path = build_engine()

    backends: list[str] = []
    if trtutils.FLAGS.EXEC_ASYNC_V3:
        backends.append("async_v3")
    if trtutils.FLAGS.EXEC_ASYNC_V2:
        backends.append("async_v2")

    # test the supported backends
    for backend in backends:
        engine = trtutils.TRTEngine(
            engine_path,
            warmup=False,
            backend=backend,
        )

        outputs = engine.mock_execute()

        assert outputs is not None


def test_multiple_engines_run() -> None:
    """Test running multiple engines simultaneously."""
    engine_path = build_engine()

    engines = [trtutils.TRTEngine(engine_path, warmup=False) for _ in range(NUM_ENGINES)]

    outputs = [engine.mock_execute() for engine in engines]

    for o in outputs:
        assert o is not None


def test_engine_run_in_thread() -> None:
    """Test engine execution in a separate thread."""
    result = [False]

    def run(result: list[bool]) -> None:
        engine_path = build_engine()

        engine = trtutils.TRTEngine(
            engine_path,
            warmup=False,
        )

        outputs = engine.mock_execute()

        assert outputs is not None

        result[0] = True

    thread = Thread(target=run, args=(result,), daemon=True)
    thread.start()

    thread.join()

    assert result[0]


def test_multiple_engines_run_in_threads() -> None:
    """Test running multiple engines in separate threads with multiple iterations."""
    result = [0] * NUM_ENGINES

    def run(threadid: int, result: list[int], iters: int) -> None:
        engine_path = build_engine()

        engine = trtutils.TRTEngine(
            engine_path,
            warmup=False,
        )

        outputs = None
        succeses = 0
        for _ in range(iters):
            outputs = engine.mock_execute()
            if outputs is not None:
                succeses += 1
        assert outputs is not None
        result[threadid] = succeses
        del engine

    threads = [
        Thread(target=run, args=(threadid, result, NUM_ITERS), daemon=True)
        for threadid in range(NUM_ENGINES)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for r in result:
        assert r == NUM_ITERS


# ============================================================================
# Pagelocked Memory
# ============================================================================


def test_engine_run_no_pagelocked() -> None:
    """Test the engine runs when pagelocked memory is disabled."""
    engine_path = build_engine()

    engine = trtutils.TRTEngine(
        engine_path,
        warmup=False,
        pagelocked_mem=False,
    )

    outputs = engine.mock_execute()

    assert outputs is not None


def test_engine_parity_non_pagelocked() -> None:
    """Test that the same engine gets same results with/without pagelocked memory."""
    engine_path = build_engine()

    engine = trtutils.TRTEngine(
        engine_path,
        warmup=False,
        pagelocked_mem=True,
    )
    engine_no_pagelocked = trtutils.TRTEngine(
        engine_path,
        warmup=False,
        pagelocked_mem=False,
    )

    rand_input = engine.get_random_input()

    outputs = engine.execute(rand_input)
    outputs_no_pagelocked = engine_no_pagelocked.execute(rand_input)

    for out, out_no_page in zip(outputs, outputs_no_pagelocked):
        assert np.allclose(out, out_no_page)


@pytest.mark.performance
def test_engine_pagelocked_performance() -> None:
    """Test that the engine runs faster with pagelocked memory."""
    engine_path = build_engine()

    engine = trtutils.TRTEngine(
        engine_path,
        warmup=True,
        warmup_iterations=10,
        pagelocked_mem=True,
    )

    engine_no_pagelocked = trtutils.TRTEngine(
        engine_path,
        warmup=True,
        warmup_iterations=10,
        pagelocked_mem=False,
    )

    rand_input = engine.get_random_input()

    pagelocked_times = []
    non_pagelocked_times = []

    for _ in range(NUM_ITERS * 5):
        t0 = time.time()
        engine.execute(rand_input)
        t1 = time.time()
        pagelocked_times.append(t1 - t0)

        t00 = time.time()
        engine_no_pagelocked.execute(rand_input)
        t11 = time.time()
        non_pagelocked_times.append(t11 - t00)

    pagelock_mean = np.mean(pagelocked_times)
    non_pagelock_mean = np.mean(non_pagelocked_times)
    speedup = non_pagelock_mean / pagelock_mean
    assert speedup > 1.0

    print(
        f"Pagelocked mean: {pagelock_mean}, Non-pagelocked mean: {non_pagelock_mean}, Speedup: {speedup}"
    )


# ============================================================================
# Direct Exec / Raw Exec
# ============================================================================


def test_engine_direct_exec_pointer_reset() -> None:
    """Ensure direct_exec followed by execute resets tensor pointers and maintains output parity."""
    engine_path = build_engine()

    engine = trtutils.TRTEngine(
        engine_path,
        warmup=False,
    )
    context = engine._context
    input_ptrs = engine._input_allocations
    input_names = [i.name for i in engine._inputs]

    rand_input = engine.get_random_input()

    device_ptrs = allocate_to_device(rand_input)
    for custom_ptr, native_ptr in zip(device_ptrs, input_ptrs):
        assert custom_ptr != native_ptr

    # direct_exec pass
    _ = engine.direct_exec(device_ptrs, no_warn=True)

    # ensure different tensor pointers setup
    for i, name in enumerate(input_names):
        assert context.get_tensor_address(name) != input_ptrs[i]
        assert context.get_tensor_address(name) == device_ptrs[i]
    assert engine._using_engine_tensors is False

    # normal execute pass
    _ = engine.execute(rand_input)

    # ensure the pointers are reset
    for i, name in enumerate(input_names):
        assert context.get_tensor_address(name) == input_ptrs[i]
    assert engine._using_engine_tensors is True

    # free
    free_device_ptrs(device_ptrs)
    del engine


def test_engine_direct_exec_parity() -> None:
    """Ensure direct_exec has the same output parity as execute."""
    engine_path = build_engine()

    engine = trtutils.TRTEngine(
        engine_path,
        warmup=False,
    )

    rand_input = engine.get_random_input()

    device_ptrs = allocate_to_device(rand_input)

    # direct_exec pass
    outputs_direct = engine.direct_exec(device_ptrs, no_warn=True)
    outputs_direct_copy = [o.copy() for o in outputs_direct]

    # normal execute pass
    outputs_execute = engine.execute(rand_input)

    # compare parity
    for out_direct, out_exec in zip(outputs_direct_copy, outputs_execute):
        assert np.array_equal(out_direct, out_exec)

    # free
    free_device_ptrs(device_ptrs)
    del engine


def test_engine_raw_exec_pointer_reset() -> None:
    """Ensure raw_exec followed by execute resets tensor pointers correctly."""
    engine_path = build_engine()

    engine = trtutils.TRTEngine(
        engine_path,
        warmup=False,
    )

    context = engine._context
    input_ptrs = engine._input_allocations
    input_names = [i.name for i in engine._inputs]

    rand_input = engine.get_random_input()

    device_ptrs = allocate_to_device(rand_input)

    for custom_ptr, native_ptr in zip(device_ptrs, input_ptrs):
        assert custom_ptr != native_ptr

    # raw_exec pass
    _ = engine.raw_exec(device_ptrs, no_warn=True)

    # ensure different tensor pointers setup
    for i, name in enumerate(input_names):
        assert context.get_tensor_address(name) != input_ptrs[i]
        assert context.get_tensor_address(name) == device_ptrs[i]
    assert engine._using_engine_tensors is False

    # normal execute pass
    _ = engine.execute(rand_input)

    # ensure the pointers are reset
    for i, name in enumerate(input_names):
        assert context.get_tensor_address(name) == input_ptrs[i]
    assert engine._using_engine_tensors is True

    # free
    free_device_ptrs(device_ptrs)
    del engine


def test_engine_raw_exec_parity() -> None:
    """Ensure raw_exec produces the same results as execute."""
    engine_path = build_engine()

    engine = trtutils.TRTEngine(
        engine_path,
        warmup=False,
    )

    rand_input = engine.get_random_input()

    device_ptrs = allocate_to_device(rand_input)

    # raw_exec pass
    output_ptrs = engine.raw_exec(device_ptrs, no_warn=True)

    # make sure computation is finished before copying results
    stream_synchronize(engine._stream)

    # copy outputs from device to host
    outputs_raw = []
    for idx, out_ptr in enumerate(output_ptrs):
        shape, dtype = engine.output_spec[idx]
        host_arr = np.empty(shape, dtype=dtype)
        memcpy_device_to_host(host_arr, out_ptr)
        outputs_raw.append(host_arr)

    # execute pass
    outputs_execute = engine.execute(rand_input)

    # compare parity
    for out_raw, out_exec in zip(outputs_raw, outputs_execute):
        assert np.array_equal(out_raw, out_exec)

    # free
    free_device_ptrs(device_ptrs)
    del engine


# ============================================================================
# Constructor Tests
# ============================================================================


def test_engine_invalid_backend_raises() -> None:
    """Test that an invalid backend raises ValueError."""
    engine_path = build_engine()

    with pytest.raises(ValueError, match="Invalid backend"):
        trtutils.TRTEngine(engine_path, backend="not_a_backend")


def test_engine_invalid_path_raises() -> None:
    """Test that a non-existent engine path raises an exception."""
    with pytest.raises((FileNotFoundError, RuntimeError)):
        trtutils.TRTEngine("/nonexistent/path/to/engine.engine")


def test_engine_default_pagelocked() -> None:
    """Test that pagelocked_mem defaults to True."""
    engine_path = build_engine()

    engine = trtutils.TRTEngine(engine_path, warmup=False)
    assert engine.pagelocked_mem is True
    del engine


# ============================================================================
# Property Tests
# ============================================================================


def test_engine_name_is_path_stem() -> None:
    """Test that engine.name is the stem of the engine file path."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(engine_path, warmup=False)
    assert engine.name == engine_path.stem
    del engine


def test_engine_input_spec_shape_dtype() -> None:
    """Test that input_spec returns list of (shape, dtype) tuples."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(engine_path, warmup=False)

    spec = engine.input_spec
    assert isinstance(spec, list)
    assert len(spec) >= 1
    for shape, dtype in spec:
        assert isinstance(shape, list)
        assert isinstance(dtype, np.dtype)
    del engine


def test_engine_output_spec_shape_dtype() -> None:
    """Test that output_spec returns list of (shape, dtype) tuples."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(engine_path, warmup=False)

    spec = engine.output_spec
    assert isinstance(spec, list)
    assert len(spec) >= 1
    for shape, dtype in spec:
        assert isinstance(shape, list)
        assert isinstance(dtype, np.dtype)
    del engine


def test_engine_input_shapes() -> None:
    """Test that input_shapes returns list of tuples."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(engine_path, warmup=False)

    shapes = engine.input_shapes
    assert isinstance(shapes, list)
    assert len(shapes) >= 1
    for shape in shapes:
        assert isinstance(shape, tuple)
        assert all(isinstance(d, int) for d in shape)
    del engine


def test_engine_output_shapes() -> None:
    """Test that output_shapes returns list of tuples."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(engine_path, warmup=False)

    shapes = engine.output_shapes
    assert isinstance(shapes, list)
    assert len(shapes) >= 1
    for shape in shapes:
        assert isinstance(shape, tuple)
        assert all(isinstance(d, int) for d in shape)
    del engine


def test_engine_input_dtypes() -> None:
    """Test that input_dtypes returns list of numpy dtypes."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(engine_path, warmup=False)

    dtypes = engine.input_dtypes
    assert isinstance(dtypes, list)
    assert len(dtypes) >= 1
    for dtype in dtypes:
        assert isinstance(dtype, np.dtype)
    del engine


def test_engine_output_dtypes() -> None:
    """Test that output_dtypes returns list of numpy dtypes."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(engine_path, warmup=False)

    dtypes = engine.output_dtypes
    assert isinstance(dtypes, list)
    assert len(dtypes) >= 1
    for dtype in dtypes:
        assert isinstance(dtype, np.dtype)
    del engine


def test_engine_input_names() -> None:
    """Test that input_names returns list of strings."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(engine_path, warmup=False)

    names = engine.input_names
    assert isinstance(names, list)
    assert len(names) >= 1
    for name in names:
        assert isinstance(name, str)
        assert len(name) > 0
    del engine


def test_engine_output_names() -> None:
    """Test that output_names returns list of strings."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(engine_path, warmup=False)

    names = engine.output_names
    assert isinstance(names, list)
    assert len(names) >= 1
    for name in names:
        assert isinstance(name, str)
        assert len(name) > 0
    del engine


def test_engine_batch_size() -> None:
    """Test that batch_size returns an integer."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(engine_path, warmup=False)

    bs = engine.batch_size
    assert isinstance(bs, int)
    assert bs >= 1
    del engine


def test_engine_memsize_positive() -> None:
    """Test that memsize returns a non-negative integer."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(engine_path, warmup=False)

    assert isinstance(engine.memsize, int)
    assert engine.memsize >= 0
    del engine


# ============================================================================
# Execute Variants
# ============================================================================


def test_engine_execute_no_copy() -> None:
    """Test that execute with no_copy returns host allocations without copying."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(engine_path, warmup=False)

    rand_input = engine.get_random_input()
    outputs = engine.execute(rand_input, no_copy=True)

    assert outputs is not None
    assert len(outputs) >= 1
    # no_copy returns the internal host allocation buffers directly
    for out, binding in zip(outputs, engine._outputs):
        assert out is binding.host_allocation
    del engine


def test_engine_execute_verbose() -> None:
    """Smoke test: execute with verbose=True doesn't crash."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(engine_path, warmup=False, verbose=True)

    outputs = engine.mock_execute(verbose=True)
    assert outputs is not None
    del engine


def test_engine_execute_debug() -> None:
    """Smoke test: execute with debug=True doesn't crash."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(engine_path, warmup=False)

    rand_input = engine.get_random_input()
    outputs = engine.execute(rand_input, debug=True)
    assert outputs is not None
    del engine


def test_engine_call_is_execute() -> None:
    """Test that __call__ delegates to execute."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(engine_path, warmup=False)

    rand_input = engine.get_random_input()
    outputs_call = engine(rand_input)
    outputs_execute = engine.execute(rand_input)

    for o_call, o_exec in zip(outputs_call, outputs_execute):
        assert np.array_equal(o_call, o_exec)
    del engine


def test_engine_mock_execute_returns_outputs() -> None:
    """Test that mock_execute returns valid outputs."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(engine_path, warmup=False)

    outputs = engine.mock_execute()
    assert outputs is not None
    assert isinstance(outputs, list)
    assert len(outputs) >= 1
    for out in outputs:
        assert isinstance(out, np.ndarray)
    del engine


def test_engine_mock_execute_with_custom_data() -> None:
    """Test that mock_execute accepts custom input data."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(engine_path, warmup=False)

    custom_data = engine.get_random_input(new=True)
    outputs = engine.mock_execute(data=custom_data)
    assert outputs is not None
    del engine


# ============================================================================
# Warmup
# ============================================================================


def test_engine_warmup_runs_iterations() -> None:
    """Test that warmup=True runs without error."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(
        engine_path,
        warmup=True,
        warmup_iterations=3,
    )

    # Should still be able to execute after warmup
    outputs = engine.mock_execute()
    assert outputs is not None
    del engine


def test_engine_warmup_false_skips() -> None:
    """Test that warmup=False skips warmup."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(
        engine_path,
        warmup=False,
    )

    # _warmup should be False/None
    assert not engine._warmup
    outputs = engine.mock_execute()
    assert outputs is not None
    del engine


# ============================================================================
# get_random_input
# ============================================================================


def test_get_random_input_shape_matches_spec() -> None:
    """Test that get_random_input returns arrays matching input_spec."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(engine_path, warmup=False)

    rand_input = engine.get_random_input()
    assert len(rand_input) == len(engine.input_spec)

    for arr, (shape, dtype) in zip(rand_input, engine.input_spec):
        assert list(arr.shape) == shape
        assert arr.dtype == dtype
    del engine


def test_get_random_input_cached() -> None:
    """Test that get_random_input returns the same cached data by default."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(engine_path, warmup=False)

    r1 = engine.get_random_input()
    r2 = engine.get_random_input()

    for a, b in zip(r1, r2):
        assert a is b  # same object (cached)
    del engine


def test_get_random_input_new_generates_fresh() -> None:
    """Test that get_random_input(new=True) generates fresh data."""
    engine_path = build_engine()
    engine = trtutils.TRTEngine(engine_path, warmup=False)

    r1 = engine.get_random_input()
    r1_copy = [a.copy() for a in r1]
    r2 = engine.get_random_input(new=True)

    # New should replace the cache, so data should differ (probabilistically)
    any_different = False
    for a, b in zip(r1_copy, r2):
        if not np.array_equal(a, b):
            any_different = True
            break
    assert any_different
    del engine


# ============================================================================
# CUDA Graph Tests
# ============================================================================


@pytest.mark.cuda_graph
@pytest.mark.skipif(
    not trtutils.FLAGS.EXEC_ASYNC_V3,
    reason="CUDA graph requires async_v3 backend",
)
class TestCUDAGraph:
    """Tests for CUDA graph capture, replay, and lifecycle."""

    # ---- Category 1: Graph Capture Verification ----

    def test_cuda_graph_lazy_capture(self) -> None:
        """Test that CUDA graph is captured lazily on first execution."""
        engine_path = build_engine()

        engine = trtutils.TRTEngine(
            engine_path,
            warmup=False,
            backend="async_v3",
            cuda_graph=True,
        )

        # Graph object should exist but not yet captured
        cuda_graph = engine._cuda_graph
        assert cuda_graph is not None
        assert cuda_graph.is_captured is False

        # Execute once to trigger capture
        outputs = engine.mock_execute()
        assert outputs is not None

        # Now graph should be captured
        assert cuda_graph.is_captured is True

    def test_cuda_graph_capture_with_warmup(self) -> None:
        """Test that CUDA graph is captured during warmup."""
        engine_path = build_engine()

        engine = trtutils.TRTEngine(
            engine_path,
            warmup=True,
            warmup_iterations=5,
            backend="async_v3",
            cuda_graph=True,
        )

        # Graph should be captured during warmup
        cuda_graph = engine._cuda_graph
        assert cuda_graph is not None
        assert cuda_graph.is_captured is True

        # Execution should still work
        outputs = engine.mock_execute()
        assert outputs is not None

    # ---- Category 2: Graph Replay ----

    def test_cuda_graph_replay_verification(self) -> None:
        """Test that captured graph is reused across multiple executions."""
        engine_path = build_engine()

        engine = trtutils.TRTEngine(
            engine_path,
            warmup=True,
            backend="async_v3",
            cuda_graph=True,
        )

        cuda_graph = engine._cuda_graph
        assert cuda_graph is not None
        # Store the graph executable ID
        assert cuda_graph.is_captured is True
        graph_exec_id = id(cuda_graph._graph_exec)

        # Run 100 iterations
        for _ in range(100):
            outputs = engine.mock_execute()
            assert outputs is not None

            # Graph executable should not change (no recapture)
            assert id(cuda_graph._graph_exec) == graph_exec_id

    # ---- Category 3: Performance Impact ----

    @pytest.mark.performance
    def test_cuda_graph_performance_improvement(self) -> None:
        """Test that CUDA graph improves execution performance."""
        engine_path = build_engine()

        # Create two engines: with and without CUDA graph
        engine_with_graph = trtutils.TRTEngine(
            engine_path,
            warmup=True,
            warmup_iterations=10,
            backend="async_v3",
            cuda_graph=True,
        )

        engine_without_graph = trtutils.TRTEngine(
            engine_path,
            warmup=True,
            warmup_iterations=10,
            backend="async_v3",
            cuda_graph=False,
        )

        # Use same input for both
        rand_input = engine_with_graph.get_random_input()

        # Measure with graph
        with_graph_times = []
        for _ in range(NUM_ITERS):
            t0 = time.perf_counter()
            engine_with_graph.execute(rand_input)
            t1 = time.perf_counter()
            with_graph_times.append(t1 - t0)

        # Measure without graph
        without_graph_times = []
        for _ in range(NUM_ITERS):
            t0 = time.perf_counter()
            engine_without_graph.execute(rand_input)
            t1 = time.perf_counter()
            without_graph_times.append(t1 - t0)

        # Calculate statistics
        mean_with = np.mean(with_graph_times)
        mean_without = np.mean(without_graph_times)
        speedup = mean_without / mean_with

        # Print for visibility
        print(f"\nWith graph: {mean_with * 1000:.3f}ms")
        print(f"Without graph: {mean_without * 1000:.3f}ms")
        print(f"Speedup: {speedup:.2f}x")

        # Assert no regression (CUDA graphs may not show improvement on small models)
        assert speedup >= 0.95, f"Expected no regression, got {speedup:.3f}"

        # Verify results are identical
        out_with = engine_with_graph.execute(rand_input)
        out_without = engine_without_graph.execute(rand_input)
        for o1, o2 in zip(out_with, out_without):
            assert np.allclose(o1, o2)

    @pytest.mark.performance
    def test_cuda_graph_performance_consistency(self) -> None:
        """Test that CUDA graph provides consistent performance."""
        engine_path = build_engine()

        engine = trtutils.TRTEngine(
            engine_path,
            warmup=True,
            warmup_iterations=10,
            backend="async_v3",
            cuda_graph=True,
        )

        rand_input = engine.get_random_input()

        # Measure latencies
        latencies = []
        for _ in range(NUM_ITERS):
            t0 = time.perf_counter()
            engine.execute(rand_input)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

        mean = np.mean(latencies)
        std = np.std(latencies)
        cv = std / mean  # Coefficient of variation
        min_lat = np.min(latencies)
        max_lat = np.max(latencies)

        print(f"\nMean: {mean * 1000:.3f}ms, Std: {std * 1000:.3f}ms, CV: {cv:.3f}")
        print(f"Min: {min_lat * 1000:.3f}ms, Max: {max_lat * 1000:.3f}ms")

        # Assert reasonable variance (allow for system jitter)
        assert cv < 0.35, f"High variation: CV={cv:.3f}"
        assert min_lat / max_lat > 0.3, "Large outliers detected"

    # ---- Category 4: Invalidation Scenarios ----

    def test_cuda_graph_invalidation_on_input_binding_change(self) -> None:
        """Test that graph is invalidated when input bindings change."""
        engine_path = build_engine()

        engine = trtutils.TRTEngine(
            engine_path,
            warmup=True,
            backend="async_v3",
            cuda_graph=True,
        )

        cuda_graph = engine._cuda_graph
        assert cuda_graph is not None
        assert cuda_graph.is_captured is True

        # Trigger invalidation by changing input bindings
        engine._set_input_bindings()

        # Graph should now be invalidated
        assert cuda_graph.is_captured is False

        # Execute again to recapture
        outputs = engine.mock_execute()
        assert outputs is not None

        # Graph should be recaptured
        assert cuda_graph.is_captured is True

    def test_cuda_graph_invalidation_on_output_binding_change(self) -> None:
        """Test that graph is invalidated when output bindings change."""
        engine_path = build_engine()

        engine = trtutils.TRTEngine(
            engine_path,
            warmup=True,
            backend="async_v3",
            cuda_graph=True,
        )

        cuda_graph = engine._cuda_graph
        assert cuda_graph is not None
        assert cuda_graph.is_captured is True

        # Trigger invalidation by changing output bindings
        engine._set_output_bindings()

        # Graph should now be invalidated
        assert cuda_graph.is_captured is False

        # Execute again to recapture
        outputs = engine.mock_execute()
        assert outputs is not None

        # Graph should be recaptured
        assert cuda_graph.is_captured is True

    def test_cuda_graph_recapture_after_invalidation(self) -> None:
        """Test that graph recapture produces correct results after invalidation."""
        engine_path = build_engine()

        engine = trtutils.TRTEngine(
            engine_path,
            warmup=True,
            backend="async_v3",
            cuda_graph=True,
        )

        cuda_graph = engine._cuda_graph
        assert cuda_graph is not None
        rand_input = engine.get_random_input()
        output_before = engine.execute(rand_input)

        # Manually invalidate graph
        cuda_graph.invalidate()
        assert cuda_graph.is_captured is False

        # Execute with same input (will recapture)
        output_after = engine.execute(rand_input)

        # Verify graph is recaptured
        assert cuda_graph.is_captured is True

        # Outputs should be identical
        for o1, o2 in zip(output_before, output_after):
            assert np.allclose(o1, o2)

    # ---- Category 5: Backend Compatibility ----

    def test_cuda_graph_only_with_async_v3(self) -> None:
        """Test that CUDA graph only works with async_v3 backend."""
        engine_path = build_engine()

        engine = trtutils.TRTEngine(
            engine_path,
            backend="async_v3",
            cuda_graph=True,
        )

        assert engine._cuda_graph_enabled is True
        assert engine._cuda_graph is not None
        assert engine._async_v3 is True

    @pytest.mark.skipif(
        not trtutils.FLAGS.EXEC_ASYNC_V2,
        reason="async_v2 backend not available",
    )
    def test_cuda_graph_disabled_with_async_v2(self) -> None:
        """Test that CUDA graph is disabled with async_v2 backend."""
        engine_path = build_engine()

        engine = trtutils.TRTEngine(
            engine_path,
            backend="async_v2",
            cuda_graph=True,
        )

        assert engine._cuda_graph_enabled is False
        assert engine._cuda_graph is None

        outputs = engine.mock_execute()
        assert outputs is not None

    # ---- Category 6: Explicit Enable/Disable ----

    def test_cuda_graph_explicit_disable(self) -> None:
        """Test that cuda_graph=False disables CUDA graph."""
        engine_path = build_engine()

        engine = trtutils.TRTEngine(
            engine_path,
            backend="async_v3",
            cuda_graph=False,
        )

        assert engine._cuda_graph_enabled is False
        assert engine._cuda_graph is None

        outputs = engine.mock_execute()
        assert outputs is not None

    def test_cuda_graph_explicit_enable(self) -> None:
        """Test that cuda_graph=True enables CUDA graph with async_v3."""
        engine_path = build_engine()

        engine = trtutils.TRTEngine(
            engine_path,
            warmup=False,
            backend="async_v3",
            cuda_graph=True,
        )

        assert engine._cuda_graph_enabled is True
        cuda_graph = engine._cuda_graph
        assert cuda_graph is not None

        # Not captured yet (no warmup)
        assert cuda_graph.is_captured is False

        # Execute to capture
        outputs = engine.mock_execute()
        assert outputs is not None

        # Now captured
        assert cuda_graph.is_captured is True

    # ---- Category 7: Capture Failures ----

    def test_cuda_graph_graceful_failure_handling(self) -> None:
        """Test that CUDA graph capture failures raise a clear error."""
        engine_path = build_engine()

        with patch("trtutils.core._graph.CUDAGraph.is_captured", property(lambda _self: False)):
            engine = trtutils.TRTEngine(
                engine_path,
                warmup=False,
                backend="async_v3",
                cuda_graph=True,
            )

            with pytest.raises(RuntimeError) as exc_info:
                engine.mock_execute()

            assert "CUDA graph capture failed" in str(exc_info.value)
            assert "cuda_graph=False" in str(exc_info.value)

    # ---- Category 8: Thread Safety ----

    def test_cuda_graph_thread_safety(self) -> None:
        """Test that each engine has independent CUDA graphs in separate threads."""
        results: dict[int, dict] = {}

        def run(threadid: int, result: dict, iters: int) -> None:
            engine_path = build_engine()
            engine = trtutils.TRTEngine(
                engine_path,
                warmup=True,
                backend="async_v3",
                cuda_graph=True,
            )

            cuda_graph = engine._cuda_graph
            assert cuda_graph is not None
            graph_exec_id = id(cuda_graph._graph_exec)

            for _ in range(iters):
                engine.mock_execute()
                assert id(cuda_graph._graph_exec) == graph_exec_id

            result[threadid] = {"graph_exec_id": graph_exec_id, "success": True}

        threads = [
            Thread(target=run, args=(i, results, NUM_ITERS), daemon=True) for i in range(NUM_ENGINES)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(results) == NUM_ENGINES
        for threadid in range(NUM_ENGINES):
            assert results[threadid]["success"] is True

        graph_ids = [results[i]["graph_exec_id"] for i in range(NUM_ENGINES)]
        assert len(set(graph_ids)) == NUM_ENGINES, "Threads should have independent graphs"

    # ---- Category 9: Alternative Execution Paths ----

    def test_cuda_graph_bypass_with_direct_exec(self) -> None:
        """Test that direct_exec bypasses CUDA graph capture."""
        engine_path = build_engine()

        engine = trtutils.TRTEngine(
            engine_path,
            warmup=False,
            backend="async_v3",
            cuda_graph=True,
        )

        cuda_graph = engine._cuda_graph
        assert cuda_graph is not None
        assert cuda_graph.is_captured is False

        rand_input = engine.get_random_input()
        device_ptrs = allocate_to_device(rand_input)

        engine.direct_exec(device_ptrs, no_warn=True)

        # Graph should still not be captured (direct_exec doesn't capture)
        assert cuda_graph.is_captured is False

        free_device_ptrs(device_ptrs)

        # Now call regular execute
        outputs = engine.execute(rand_input)
        assert outputs is not None

        # Now graph should be captured
        assert cuda_graph.is_captured is True

    def test_cuda_graph_bypass_with_raw_exec(self) -> None:
        """Test that raw_exec bypasses CUDA graph capture."""
        engine_path = build_engine()

        engine = trtutils.TRTEngine(
            engine_path,
            warmup=False,
            backend="async_v3",
            cuda_graph=True,
        )

        cuda_graph = engine._cuda_graph
        assert cuda_graph is not None
        assert cuda_graph.is_captured is False

        rand_input = engine.get_random_input()
        device_ptrs = allocate_to_device(rand_input)

        output_ptrs = engine.raw_exec(device_ptrs, no_warn=True)
        assert output_ptrs is not None

        # Graph should still not be captured (raw_exec doesn't capture)
        assert cuda_graph.is_captured is False

        free_device_ptrs(device_ptrs)

        # Now call regular execute
        outputs = engine.execute(rand_input)
        assert outputs is not None

        # Now graph should be captured
        assert cuda_graph.is_captured is True

    # ---- Category 10: graph_exec method ----

    def test_graph_exec_raises_without_capture(self) -> None:
        """Test that graph_exec raises RuntimeError when no graph is captured."""
        engine_path = build_engine()

        engine = trtutils.TRTEngine(
            engine_path,
            warmup=False,
            backend="async_v3",
            cuda_graph=True,
        )

        with pytest.raises(RuntimeError, match="No CUDA graph captured"):
            engine.graph_exec()

    def test_graph_exec_after_warmup(self) -> None:
        """Test that graph_exec works after warmup captures the graph."""
        engine_path = build_engine()

        engine = trtutils.TRTEngine(
            engine_path,
            warmup=True,
            warmup_iterations=3,
            backend="async_v3",
            cuda_graph=True,
        )

        assert engine._cuda_graph is not None
        assert engine._cuda_graph.is_captured is True

        # graph_exec should not raise
        engine.graph_exec()

    def test_graph_exec_debug_sync(self) -> None:
        """Test that graph_exec with debug=True synchronizes the stream."""
        engine_path = build_engine()

        engine = trtutils.TRTEngine(
            engine_path,
            warmup=True,
            warmup_iterations=3,
            backend="async_v3",
            cuda_graph=True,
        )

        # Should not raise; debug forces stream synchronization
        engine.graph_exec(debug=True)
