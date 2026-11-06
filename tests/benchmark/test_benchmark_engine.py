# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for benchmark_engine() in trtutils._benchmark."""

from __future__ import annotations

from unittest.mock import MagicMock

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
@pytest.mark.parametrize(
    "cuda_graph",
    [
        pytest.param(False, id="no-cuda-graph"),
        pytest.param(True, id="cuda-graph", marks=pytest.mark.cuda_graph),
    ],
)
def test_benchmark_engine(engine_path, input_source, cuda_graph) -> None:
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
        cuda_graph=cuda_graph,
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


@pytest.mark.parametrize(
    ("warmup", "expected_extra"),
    [
        pytest.param(None, 3, id="default-warms-up"),
        pytest.param(True, 3, id="warmup"),
        pytest.param(False, 0, id="no-warmup"),
    ],
)
def test_benchmark_engine_warmup_existing_engine(warmup, expected_extra) -> None:
    """None warms up like True for an existing engine; False skips the warmup calls."""
    engine = MagicMock()
    benchmark_engine(engine, iterations=1, warmup_iterations=3, warmup=warmup)
    assert engine.mock_execute.call_count == 1 + expected_extra


@pytest.mark.parametrize(
    ("warmup", "expected"),
    [
        pytest.param(None, True, id="default-warms-up"),
        pytest.param(True, True, id="warmup"),
        pytest.param(False, False, id="no-warmup"),
    ],
)
def test_benchmark_engine_warmup_path(monkeypatch, warmup, expected) -> None:
    """A path input forwards a resolved warmup flag to TRTEngine."""
    mock_cls = MagicMock()
    monkeypatch.setattr("trtutils._benchmark.TRTEngine", mock_cls)
    benchmark_engine("fake.engine", iterations=1, warmup=warmup)
    assert mock_cls.call_args.kwargs["warmup"] is expected
