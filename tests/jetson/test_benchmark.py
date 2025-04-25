# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from tests.common import build_engine
from trtutils import FLAGS, Metric
from trtutils.jetson import (
    JetsonBenchmarkResult,
    benchmark_engine,
    benchmark_engines,
)

ITERS = 10
WARMUP_ITERS = 2


def test_benchmark_result_class() -> None:
    """Test the BenchmarkResult class."""
    if not FLAGS.IS_JETSON:
        return

    data = [0.1, 0.2, 0.3, 0.4, 0.5]
    latency = Metric(data)
    power_draw = Metric(data)
    energy = Metric(data)
    result = JetsonBenchmarkResult(latency, power_draw, energy)

    assert result.latency.mean == 0.3
    assert result.power_draw.mean == 0.3
    assert result.energy.mean == 0.3

    # Test string representation
    str_repr = str(result)
    assert "JetsonBenchmarkResult(latency=" in str_repr

    # Test repr representation
    repr_str = repr(result)
    assert "JetsonBenchmarkResult(latency=" in repr_str


def test_benchmark_engine() -> None:
    """Test benchmarking a single engine."""
    if not FLAGS.IS_JETSON:
        return

    engine_path = build_engine()

    result = benchmark_engine(
        engine_path, iterations=ITERS, warmup_iterations=WARMUP_ITERS
    )

    assert isinstance(result, JetsonBenchmarkResult)
    assert isinstance(result.latency, Metric)
    assert isinstance(result.power_draw, Metric)
    assert isinstance(result.energy, Metric)
    assert len(result.latency.raw) == ITERS
    assert len(result.power_draw.raw) == ITERS
    assert len(result.energy.raw) == ITERS


def test_benchmark_engines() -> None:
    """Test benchmarking multiple engines sequentially."""
    if not FLAGS.IS_JETSON:
        return

    engine_path = build_engine()

    num_engines = 2
    results = benchmark_engines(
        [engine_path] * num_engines, iterations=ITERS, warmup_iterations=WARMUP_ITERS
    )

    assert len(results) == num_engines
    assert all(isinstance(r, JetsonBenchmarkResult) for r in results)
    for result in results:
        assert len(result.latency.raw) == ITERS
        assert len(result.power_draw.raw) == ITERS
        assert len(result.energy.raw) == ITERS


def test_benchmark_engines_parallel() -> None:
    """Test benchmarking multiple engines in parallel."""
    if not FLAGS.IS_JETSON:
        return

    engine_path = build_engine()

    results = benchmark_engines(
        [engine_path, engine_path],
        iterations=ITERS,
        warmup_iterations=WARMUP_ITERS,
        parallel=True,
    )

    assert len(results) == 1
    assert isinstance(results[0], JetsonBenchmarkResult)
    assert len(results[0].latency.raw) == ITERS
    assert len(results[0].power_draw.raw) == ITERS
    assert len(results[0].energy.raw) == ITERS
