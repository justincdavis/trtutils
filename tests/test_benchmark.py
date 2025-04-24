# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from trtutils import (
    BenchmarkResult,
    Metric,
    benchmark_engine,
    benchmark_engines,
)

from .common import build_engine

ITERS = 10
WARMUP_ITERS = 2


def test_metric_class() -> None:
    """Test the Metric class."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    metric = Metric(data)

    assert metric.raw
    assert metric.mean == 3.0
    assert metric.median == 3.0
    assert metric.min == 1.0
    assert metric.max == 5.0

    # Test string representation
    str_repr = str(metric)
    assert "mean=3.000" in str_repr
    assert "median=3.000" in str_repr
    assert "min=1.000" in str_repr
    assert "max=5.000" in str_repr

    # Test repr representation
    repr_str = repr(metric)
    assert "mean=3.0" in repr_str
    assert "median=3.0" in repr_str

    # Ensure raw data is not in string representation
    assert "[1.0, 2.0, 3.0, 4.0, 5.0]" not in str_repr
    assert "[1.0, 2.0, 3.0, 4.0, 5.0]" not in repr_str


def test_benchmark_result_class() -> None:
    """Test the BenchmarkResult class."""
    data = [0.1, 0.2, 0.3, 0.4, 0.5]
    latency = Metric(data)
    result = BenchmarkResult(latency)

    assert result.latency.mean == 0.3

    # Test string representation
    str_repr = str(result)
    assert "BenchmarkResult(latency=" in str_repr

    # Test repr representation
    repr_str = repr(result)
    assert "BenchmarkResult(latency=" in repr_str


def test_benchmark_engine() -> None:
    """Test benchmarking a single engine."""
    engine_path = build_engine()

    result = benchmark_engine(
        engine_path, iterations=ITERS, warmup_iterations=WARMUP_ITERS
    )

    assert isinstance(result, BenchmarkResult)
    assert isinstance(result.latency, Metric)
    assert len(result.latency.raw) == ITERS


def test_benchmark_engines() -> None:
    """Test benchmarking multiple engines sequentially."""
    engine_path = build_engine()

    num_engines = 2
    results = benchmark_engines(
        [engine_path] * num_engines, iterations=ITERS, warmup_iterations=WARMUP_ITERS
    )

    assert len(results) == num_engines
    assert all(isinstance(r, BenchmarkResult) for r in results)
    for result in results:
        assert len(result.latency.raw) == ITERS


def test_benchmark_engines_parallel() -> None:
    """Test benchmarking multiple engines in parallel."""
    engine_path = build_engine()

    results = benchmark_engines(
        [engine_path, engine_path],
        iterations=ITERS,
        warmup_iterations=WARMUP_ITERS,
        parallel=True,
    )

    assert len(results) == 1
    assert isinstance(results[0], BenchmarkResult)
    assert len(results[0].latency.raw) == ITERS
