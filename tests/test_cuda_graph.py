# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
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
from trtutils.core import allocate_to_device, free_device_ptrs

from .common import build_engine

NUM_ENGINES = 4
NUM_ITERS = 1_000

# Skip entire file if async_v3 not available
pytestmark = pytest.mark.skipif(
    not trtutils.FLAGS.EXEC_ASYNC_V3,
    reason="CUDA graph requires async_v3 backend",
)


# ============================================================================
# Category 1: Graph Capture Verification
# ============================================================================


def test_cuda_graph_lazy_capture() -> None:
    """Test that CUDA graph is captured lazily on first execution."""
    engine_path = build_engine()

    engine = trtutils.TRTEngine(
        engine_path,
        warmup=False,
        backend="async_v3",
        cuda_graph=True,
    )

    # Graph object should exist but not yet captured
    assert engine._cuda_graph is not None
    assert engine._cuda_graph.is_captured is False

    # Execute once to trigger capture
    outputs = engine.mock_execute()
    assert outputs is not None

    # Now graph should be captured
    assert engine._cuda_graph.is_captured is True


def test_cuda_graph_capture_with_warmup() -> None:
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
    assert engine._cuda_graph is not None
    assert engine._cuda_graph.is_captured is True

    # Execution should still work
    outputs = engine.mock_execute()
    assert outputs is not None


# ============================================================================
# Category 2: Graph Replay
# ============================================================================


def test_cuda_graph_replay_verification() -> None:
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


# ============================================================================
# Category 3: Performance Impact
# ============================================================================


@pytest.mark.performance
def test_cuda_graph_performance_improvement() -> None:
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

    # Assert improvement (may be small for simple models)
    assert speedup > 1.0, f"Expected speedup > 1.0, got {speedup:.3f}"

    # Verify results are identical
    out_with = engine_with_graph.execute(rand_input)
    out_without = engine_without_graph.execute(rand_input)
    for o1, o2 in zip(out_with, out_without):
        assert np.allclose(o1, o2)


@pytest.mark.performance
def test_cuda_graph_performance_consistency() -> None:
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

    # Assert low variance
    assert cv < 0.2, f"High variation: CV={cv:.3f}"
    assert min_lat / max_lat > 0.5, "Large outliers detected"


# ============================================================================
# Category 4: Invalidation Scenarios
# ============================================================================


def test_cuda_graph_invalidation_on_input_binding_change() -> None:
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
    # Graph should be captured
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


def test_cuda_graph_invalidation_on_output_binding_change() -> None:
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
    # Graph should be captured
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


def test_cuda_graph_recapture_after_invalidation() -> None:
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
    # Get random input and execute
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


# ============================================================================
# Category 5: Backend Compatibility
# ============================================================================


def test_cuda_graph_only_with_async_v3() -> None:
    """Test that CUDA graph only works with async_v3 backend."""
    engine_path = build_engine()

    engine = trtutils.TRTEngine(
        engine_path,
        backend="async_v3",
        cuda_graph=True,
    )

    # CUDA graph should be enabled
    assert engine._cuda_graph_enabled is True
    assert engine._cuda_graph is not None
    assert engine._async_v3 is True


@pytest.mark.skipif(
    not trtutils.FLAGS.EXEC_ASYNC_V2,
    reason="async_v2 backend not available",
)
def test_cuda_graph_disabled_with_async_v2() -> None:
    """Test that CUDA graph is disabled with async_v2 backend."""
    engine_path = build_engine()

    engine = trtutils.TRTEngine(
        engine_path,
        backend="async_v2",
        cuda_graph=True,
    )

    # CUDA graph should be disabled
    assert engine._cuda_graph_enabled is False
    assert engine._cuda_graph is None

    # Execution should still work
    outputs = engine.mock_execute()
    assert outputs is not None


# ============================================================================
# Category 6: Explicit Enable/Disable
# ============================================================================


def test_cuda_graph_explicit_disable() -> None:
    """Test that cuda_graph=False disables CUDA graph."""
    engine_path = build_engine()

    engine = trtutils.TRTEngine(
        engine_path,
        backend="async_v3",
        cuda_graph=False,
    )

    # CUDA graph should be disabled
    assert engine._cuda_graph_enabled is False
    assert engine._cuda_graph is None

    # Execution should still work
    outputs = engine.mock_execute()
    assert outputs is not None


def test_cuda_graph_explicit_enable() -> None:
    """Test that cuda_graph=True enables CUDA graph with async_v3."""
    engine_path = build_engine()

    engine = trtutils.TRTEngine(
        engine_path,
        warmup=False,
        backend="async_v3",
        cuda_graph=True,
    )

    # CUDA graph should be enabled
    assert engine._cuda_graph_enabled is True
    assert engine._cuda_graph is not None

    # Not captured yet (no warmup)
    assert engine._cuda_graph.is_captured is False

    # Execute to capture
    outputs = engine.mock_execute()
    assert outputs is not None

    # Now captured
    assert engine._cuda_graph.is_captured is True


# ============================================================================
# Category 7: Capture Failures
# ============================================================================


def test_cuda_graph_graceful_failure_handling() -> None:
    """Test that CUDA graph capture failures are handled gracefully."""
    engine_path = build_engine()

    # Mock stop() to return False (simulating capture failure)
    with patch("trtutils.core._graph.CUDAGraph.stop", return_value=False):
        engine = trtutils.TRTEngine(
            engine_path,
            warmup=False,
            backend="async_v3",
            cuda_graph=True,
        )

        # Should not raise exception despite capture "failure"
        outputs = engine.mock_execute()
        assert outputs is not None

        cuda_graph = engine._cuda_graph
        assert cuda_graph is not None
        # Graph should not be captured
        assert cuda_graph.is_captured is False


# ============================================================================
# Category 8: Thread Safety
# ============================================================================


def test_cuda_graph_thread_safety() -> None:
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
        # Store initial graph exec id
        graph_exec_id = id(cuda_graph._graph_exec)

        # Run iterations
        for _ in range(iters):
            engine.mock_execute()

            # Verify graph exec id never changes
            assert id(cuda_graph._graph_exec) == graph_exec_id

        result[threadid] = {"graph_exec_id": graph_exec_id, "success": True}

    # Create and start threads
    threads = [
        Thread(target=run, args=(i, results, NUM_ITERS), daemon=True) for i in range(NUM_ENGINES)
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Verify all threads succeeded
    assert len(results) == NUM_ENGINES
    for threadid in range(NUM_ENGINES):
        assert results[threadid]["success"] is True

    # Verify all graph exec ids are unique (each thread has its own graph)
    graph_ids = [results[i]["graph_exec_id"] for i in range(NUM_ENGINES)]
    assert len(set(graph_ids)) == NUM_ENGINES, "Threads should have independent graphs"


# ============================================================================
# Category 9: Alternative Execution Paths
# ============================================================================


def test_cuda_graph_bypass_with_direct_exec() -> None:
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
    # Graph not captured yet
    assert cuda_graph.is_captured is False

    # Allocate random input to device
    rand_input = engine.get_random_input()
    device_ptrs = allocate_to_device(rand_input)

    # Call direct_exec
    engine.direct_exec(device_ptrs, no_warn=True)

    # Graph should still not be captured (direct_exec doesn't capture)
    assert cuda_graph.is_captured is False

    # Free device memory
    free_device_ptrs(device_ptrs)

    # Now call regular execute
    outputs = engine.execute(rand_input)
    assert outputs is not None

    # Now graph should be captured
    assert cuda_graph.is_captured is True


def test_cuda_graph_bypass_with_raw_exec() -> None:
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
    # Graph not captured yet
    assert cuda_graph.is_captured is False

    # Allocate random input to device
    rand_input = engine.get_random_input()
    device_ptrs = allocate_to_device(rand_input)

    # Call raw_exec
    output_ptrs = engine.raw_exec(device_ptrs, no_warn=True)
    assert output_ptrs is not None

    # Graph should still not be captured (raw_exec doesn't capture)
    assert cuda_graph.is_captured is False

    # Free device memory
    free_device_ptrs(device_ptrs)
    free_device_ptrs(output_ptrs)

    # Now call regular execute
    outputs = engine.execute(rand_input)
    assert outputs is not None

    # Now graph should be captured
    assert cuda_graph.is_captured is True
