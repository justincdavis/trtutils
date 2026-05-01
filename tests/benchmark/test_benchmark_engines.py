# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for benchmark_engines() in trtutils._benchmark."""

from __future__ import annotations

import pytest

from trtutils import BenchmarkResult, benchmark_engines

from .conftest import ITERS, WARMUP_ITERS


@pytest.mark.parametrize(
    "count",
    [pytest.param(1, id="single"), pytest.param(2, id="multiple")],
)
def test_benchmark_engines_sequential(engine_path, count) -> None:
    """Sequential benchmark_engines returns correct count of valid BenchmarkResults."""
    results = benchmark_engines(
        [engine_path] * count,
        iterations=ITERS,
        warmup_iterations=WARMUP_ITERS,
    )
    assert isinstance(results, list)
    assert len(results) == count
    assert all(isinstance(r, BenchmarkResult) for r in results)
    for result in results:
        assert len(result.latency.raw) == ITERS


def test_benchmark_engines_parallel(engine_path) -> None:
    """Parallel mode returns valid BenchmarkResults with positive latencies."""
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


@pytest.mark.parametrize(
    "entry_factory",
    [
        pytest.param(lambda p: (p, None), id="path-tuple"),
        pytest.param(lambda p: (str(p), None), id="string-tuple"),
    ],
)
def test_benchmark_engines_tuple_input(engine_path, entry_factory) -> None:
    """benchmark_engines accepts (path, dla_core) tuple entries."""
    results = benchmark_engines(
        [entry_factory(engine_path)],
        iterations=ITERS,
        warmup_iterations=WARMUP_ITERS,
    )
    assert len(results) == 1
    assert isinstance(results[0], BenchmarkResult)
