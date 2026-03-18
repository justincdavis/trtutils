# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for benchmark_engine() in trtutils._benchmark."""

from __future__ import annotations

import pytest

from trtutils import BenchmarkResult, Metric, TRTEngine, benchmark_engine

from .conftest import ITERS, WARMUP_ITERS


def test_benchmark_engine_path(engine_path) -> None:
    """benchmark_engine with a Path returns valid BenchmarkResult with correct stats."""
    result = benchmark_engine(
        engine_path,
        iterations=ITERS,
        warmup_iterations=WARMUP_ITERS,
    )
    assert isinstance(result, BenchmarkResult)
    assert isinstance(result.latency, Metric)
    assert len(result.latency.raw) == ITERS
    assert all(v > 0 for v in result.latency.raw)
    assert result.latency.mean > 0
    assert result.latency.median > 0
    assert result.latency.min > 0
    assert result.latency.max > 0


def test_benchmark_engine_string_path(engine_path) -> None:
    """benchmark_engine accepts a string path."""
    result = benchmark_engine(
        str(engine_path),
        iterations=ITERS,
        warmup_iterations=WARMUP_ITERS,
    )
    assert isinstance(result, BenchmarkResult)


def test_benchmark_engine_object(engine_path) -> None:
    """benchmark_engine accepts a TRTEngine instance directly."""
    eng = TRTEngine(engine_path, warmup=False)
    result = benchmark_engine(eng, iterations=ITERS)
    assert isinstance(result, BenchmarkResult)
    assert len(result.latency.raw) == ITERS
    del eng


def test_benchmark_engine_warmup_disabled(engine_path) -> None:
    """benchmark_engine works with warmup disabled."""
    result = benchmark_engine(
        engine_path,
        iterations=ITERS,
        warmup_iterations=0,
        warmup=False,
    )
    assert isinstance(result, BenchmarkResult)


@pytest.mark.parametrize("iterations", [1, 5])
def test_benchmark_engine_iterations(engine_path, iterations) -> None:
    """benchmark_engine respects custom iteration counts."""
    result = benchmark_engine(engine_path, iterations=iterations, warmup=False)
    assert len(result.latency.raw) == iterations
