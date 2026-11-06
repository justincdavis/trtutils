# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for benchmark_engines() in trtutils._benchmark."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trtutils import BenchmarkResult, benchmark_engines

from .conftest import ITERS, WARMUP_ITERS


@pytest.mark.parametrize(
    "count",
    [pytest.param(1, id="single"), pytest.param(2, id="multiple")],
)
@pytest.mark.parametrize(
    "cuda_graph",
    [
        pytest.param(False, id="no-cuda-graph"),
        pytest.param(True, id="cuda-graph", marks=pytest.mark.cuda_graph),
    ],
)
def test_benchmark_engines_sequential(engine_path, count, cuda_graph) -> None:
    """Sequential benchmark_engines returns correct count of valid BenchmarkResults."""
    results = benchmark_engines(
        [engine_path] * count,
        iterations=ITERS,
        warmup_iterations=WARMUP_ITERS,
        cuda_graph=cuda_graph,
    )
    assert isinstance(results, list)
    assert len(results) == count
    assert all(isinstance(r, BenchmarkResult) for r in results)
    for result in results:
        assert len(result.latency.raw) == ITERS


@pytest.mark.parametrize(
    "cuda_graph",
    [
        pytest.param(False, id="no-cuda-graph"),
        pytest.param(True, id="cuda-graph", marks=pytest.mark.cuda_graph),
    ],
)
def test_benchmark_engines_parallel(engine_path, cuda_graph) -> None:
    """Parallel mode returns valid BenchmarkResults with positive latencies."""
    results = benchmark_engines(
        [engine_path],
        iterations=ITERS,
        warmup_iterations=WARMUP_ITERS,
        parallel=True,
        cuda_graph=cuda_graph,
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


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        pytest.param([("a.engine", None, 1)], [("a.engine", None, 1)], id="device-only"),
        pytest.param([("a.engine", 0, 1)], [("a.engine", 0, 1)], id="dla-and-device"),
        pytest.param(
            [("a.engine", 0), "b.engine", ("c.engine", None, 2)],
            [("a.engine", 0, None), "b.engine", ("c.engine", None, 2)],
            id="mixed",
        ),
    ],
)
def test_benchmark_engines_parallel_forwards_devices(entries, expected) -> None:
    """Parallel mode hands each engine's device to ParallelTRTEngines."""
    expected = [(Path(e[0]), *e[1:]) if isinstance(e, tuple) else Path(e) for e in expected]
    with patch("trtutils._benchmark.ParallelTRTEngines") as mock_cls:
        mock_cls.return_value = MagicMock()
        benchmark_engines(entries, iterations=1, warmup_iterations=1, parallel=True)
    assert mock_cls.call_args.args[0] == expected


def test_benchmark_engines_parallel_explicit_device(engine_path) -> None:
    """Parallel mode runs with explicit (engine, dla_core, device) tuples."""
    results = benchmark_engines(
        [(engine_path, None, 0)] * 2,
        iterations=ITERS,
        warmup_iterations=WARMUP_ITERS,
        parallel=True,
    )
    assert len(results) == 1
    assert len(results[0].latency.raw) == ITERS
