# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for benchmark_engines() in trtutils._benchmark."""

from __future__ import annotations

from trtutils import BenchmarkResult, benchmark_engines

from .conftest import ITERS, WARMUP_ITERS


def test_benchmark_engines_sequential(engine_path) -> None:
    """Sequential mode returns correct count of BenchmarkResults with proper raw length."""
    results = benchmark_engines(
        [engine_path, engine_path],
        iterations=ITERS,
        warmup_iterations=WARMUP_ITERS,
    )
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, BenchmarkResult) for r in results)
    for result in results:
        assert len(result.latency.raw) == ITERS


def test_benchmark_engines_single(engine_path) -> None:
    """benchmark_engines works with a single engine in the list."""
    results = benchmark_engines(
        [engine_path],
        iterations=ITERS,
        warmup_iterations=WARMUP_ITERS,
    )
    assert len(results) == 1
    assert isinstance(results[0], BenchmarkResult)


def test_benchmark_engines_parallel(engine_path) -> None:
    """Parallel mode returns a single BenchmarkResult with positive latencies."""
    results = benchmark_engines(
        [engine_path],
        iterations=ITERS,
        warmup_iterations=WARMUP_ITERS,
        parallel=True,
    )
    assert len(results) == 1
    assert isinstance(results[0], BenchmarkResult)
    assert len(results[0].latency.raw) == ITERS
    assert all(v > 0 for v in results[0].latency.raw)


def test_benchmark_engines_tuple_input(engine_path) -> None:
    """benchmark_engines accepts (path, dla_core) tuple entries with Path and str."""
    results = benchmark_engines(
        [(engine_path, None)],
        iterations=ITERS,
        warmup_iterations=WARMUP_ITERS,
    )
    assert len(results) == 1
    assert isinstance(results[0], BenchmarkResult)

    results_str = benchmark_engines(
        [(str(engine_path), None)],
        iterations=ITERS,
        warmup_iterations=WARMUP_ITERS,
    )
    assert len(results_str) == 1
    assert isinstance(results_str[0], BenchmarkResult)
