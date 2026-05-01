# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for trtutils.jetson._benchmark -- JetsonBenchmarkResult, benchmark_engine, benchmark_engines."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("jetsontools")

from trtutils import Metric
from trtutils.jetson import (
    JetsonBenchmarkResult,
    benchmark_engine,
    benchmark_engines,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.jetson

ITERS = 100
WARMUP_ITERS = 5


def test_jetson_benchmark_result() -> None:
    """JetsonBenchmarkResult exposes latency/power_draw/energy Metric fields and a readable repr."""
    data = [0.1, 0.2, 0.3, 0.4, 0.5]
    latency = Metric(data)
    power_draw = Metric(data)
    energy = Metric(data)
    result = JetsonBenchmarkResult(latency, power_draw, energy)

    field_names = {f.name for f in dataclasses.fields(JetsonBenchmarkResult)}
    assert field_names == {"latency", "power_draw", "energy"}
    assert result.latency is latency
    assert result.power_draw is power_draw
    assert result.energy is energy
    for text in (str(result), repr(result)):
        assert "JetsonBenchmarkResult(latency=" in text
        assert "power_draw=" in text
        assert "energy=" in text


@pytest.mark.parametrize(
    "as_string",
    [pytest.param(False, id="path"), pytest.param(True, id="string-path")],
)
def test_benchmark_engine(engine_path: Path, as_string: bool) -> None:
    """benchmark_engine returns a JetsonBenchmarkResult with valid metrics for Path and str inputs."""
    target = str(engine_path) if as_string else engine_path
    result = benchmark_engine(
        target,
        iterations=ITERS,
        warmup_iterations=WARMUP_ITERS,
        tegra_interval=1,
    )
    assert isinstance(result, JetsonBenchmarkResult)
    assert isinstance(result.latency, Metric)
    assert isinstance(result.power_draw, Metric)
    assert isinstance(result.energy, Metric)
    assert len(result.latency.raw) == ITERS
    assert len(result.power_draw.raw) > 0
    assert len(result.energy.raw) > 0


@pytest.mark.parametrize(
    ("parallel", "expected_count"),
    [
        pytest.param(False, 2, id="sequential"),
        pytest.param(True, 1, id="parallel"),
    ],
)
def test_benchmark_engines(engine_path: Path, parallel: bool, expected_count: int) -> None:
    """benchmark_engines returns one result per engine sequentially, or one combined result in parallel."""
    results = benchmark_engines(
        [engine_path, engine_path],
        iterations=ITERS,
        warmup_iterations=WARMUP_ITERS,
        tegra_interval=1,
        parallel=parallel,
    )
    assert len(results) == expected_count
    for result in results:
        assert isinstance(result, JetsonBenchmarkResult)
        assert len(result.latency.raw) == ITERS
        assert len(result.power_draw.raw) > 0
        assert len(result.energy.raw) > 0
