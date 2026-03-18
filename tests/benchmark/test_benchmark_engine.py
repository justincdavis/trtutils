# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for benchmark_engine() in trtutils._benchmark."""

from __future__ import annotations

import pytest

from trtutils import BenchmarkResult, Metric, TRTEngine, benchmark_engine

from .conftest import ITERS, WARMUP_ITERS


@pytest.mark.parametrize(
    "input_source",
    [
        pytest.param("path", id="path"),
        pytest.param("string", id="string-path"),
        pytest.param("engine", id="trt-engine"),
    ],
)
def test_benchmark_engine(engine_path, input_source) -> None:
    """benchmark_engine returns valid BenchmarkResult for Path, str, and TRTEngine inputs."""
    if input_source == "path":
        target = engine_path
    elif input_source == "string":
        target = str(engine_path)
    else:
        target = TRTEngine(engine_path, warmup=False)
    result = benchmark_engine(
        target,
        iterations=ITERS,
        warmup_iterations=WARMUP_ITERS,
    )
    if input_source == "engine":
        del target
    assert isinstance(result, BenchmarkResult)
    assert isinstance(result.latency, Metric)
    assert len(result.latency.raw) == ITERS
    assert all(v > 0 for v in result.latency.raw)
    assert result.latency.mean > 0
    assert result.latency.median > 0
    assert result.latency.min > 0
    assert result.latency.max > 0


@pytest.mark.parametrize(
    "iterations",
    [pytest.param(1, id="1-iter"), pytest.param(5, id="5-iter")],
)
def test_benchmark_engine_iterations(engine_path, iterations) -> None:
    """benchmark_engine respects custom iteration counts."""
    result = benchmark_engine(engine_path, iterations=iterations, warmup=False)
    assert len(result.latency.raw) == iterations
