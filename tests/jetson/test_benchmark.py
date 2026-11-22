# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for trtutils.jetson._benchmark -- JetsonBenchmarkResult, benchmark_engine, benchmark_engines."""

from __future__ import annotations

import dataclasses
from contextlib import nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

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
@pytest.mark.parametrize(
    "cuda_graph",
    [
        pytest.param(False, id="no-cuda-graph"),
        pytest.param(True, id="cuda-graph", marks=pytest.mark.cuda_graph),
    ],
)
@pytest.mark.jetson
def test_benchmark_engine(engine_path: Path, as_string: bool, cuda_graph: bool) -> None:
    """benchmark_engine returns a JetsonBenchmarkResult with valid metrics for Path and str inputs."""
    target = str(engine_path) if as_string else engine_path
    result = benchmark_engine(
        target,
        iterations=ITERS,
        warmup_iterations=WARMUP_ITERS,
        tegra_interval=1,
        cuda_graph=cuda_graph,
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
@pytest.mark.parametrize(
    "cuda_graph",
    [
        pytest.param(False, id="no-cuda-graph"),
        pytest.param(True, id="cuda-graph", marks=pytest.mark.cuda_graph),
    ],
)
@pytest.mark.jetson
def test_benchmark_engines(
    engine_path: Path,
    parallel: bool,
    expected_count: int,
    cuda_graph: bool,
) -> None:
    """benchmark_engines returns one result per engine sequentially, or one combined result in parallel."""
    results = benchmark_engines(
        [engine_path, engine_path],
        iterations=ITERS,
        warmup_iterations=WARMUP_ITERS,
        tegra_interval=1,
        parallel=parallel,
        cuda_graph=cuda_graph,
    )
    assert len(results) == expected_count
    for result in results:
        assert isinstance(result, JetsonBenchmarkResult)
        assert len(result.latency.raw) == ITERS
        assert len(result.power_draw.raw) > 0
        assert len(result.energy.raw) > 0


MODULE = "trtutils.jetson._benchmark"


def _mock_tegra(monkeypatch, has_sample: list[bool]) -> None:
    """Mock the tegrastats pipeline: one 1 s inference window per entry, 1.0 W when sampled."""
    sample = [{"VDD_TOTAL": 1.0}]
    per_inference = [((float(i), i + 1.0), sample if h else []) for i, h in enumerate(has_sample)]
    filtered = sample if any(has_sample) else []
    monkeypatch.setattr(f"{MODULE}.TegraStats", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr(f"{MODULE}.parse_tegrastats", MagicMock())
    monkeypatch.setattr(f"{MODULE}.filter_data", MagicMock(return_value=(filtered, per_inference)))
    monkeypatch.setattr(
        f"{MODULE}.get_powerdraw",
        MagicMock(return_value={"VDD_TOTAL": SimpleNamespace(raw=[1.0], mean=1.0)}),
    )
    monkeypatch.setattr(f"{MODULE}.ParallelTRTEngines", MagicMock())


@pytest.mark.parametrize(
    ("warmup", "expected_extra"),
    [
        pytest.param(None, 3, id="default-warms-up"),
        pytest.param(True, 3, id="warmup"),
        pytest.param(False, 0, id="no-warmup"),
    ],
)
def test_benchmark_engine_warmup_default(monkeypatch, warmup, expected_extra) -> None:
    """None warms up like True for an existing engine; False skips the warmup calls."""
    _mock_tegra(monkeypatch, [True])
    engine = MagicMock()
    result = benchmark_engine(engine, iterations=1, warmup_iterations=3, warmup=warmup)
    assert engine.mock_execute.call_count == 1 + expected_extra
    assert result.power_draw.raw == [1.0]
    # energy = mean power (1.0) * inference duration (1.0)
    assert result.energy.raw == [1.0]


def _run_single(**kwargs: object) -> JetsonBenchmarkResult:
    return benchmark_engine(MagicMock(), warmup=False, **kwargs)


def _run_parallel(**kwargs: object) -> JetsonBenchmarkResult:
    (result,) = benchmark_engines([MagicMock()], parallel=True, warmup=False, **kwargs)
    return result


RUNNERS = [pytest.param(_run_single, id="single"), pytest.param(_run_parallel, id="parallel")]


@pytest.mark.parametrize("run", RUNNERS)
def test_benchmark_no_samples_raises(monkeypatch, run) -> None:
    """No tegrastats sample inside any inference window raises an actionable error."""
    _mock_tegra(monkeypatch, [False])
    with pytest.raises(RuntimeError, match="tegra_interval"):
        run(iterations=1, tegra_interval=500)


@pytest.mark.parametrize("run", RUNNERS)
def test_benchmark_partial_samples_warns(monkeypatch, run) -> None:
    """Inferences without a sample are dropped from energy, with a warning."""
    _mock_tegra(monkeypatch, [True, False])
    warn = MagicMock()
    monkeypatch.setattr(f"{MODULE}.LOG.warning", warn)
    result = run(iterations=2)
    assert result.energy.raw == [1.0]
    warn.assert_called_once()
    assert "1 of 2" in warn.call_args.args[0]


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(benchmark_engine, id="engine"),
        pytest.param(lambda e, **kw: benchmark_engines([e], **kw), id="engines"),
        pytest.param(lambda e, **kw: benchmark_engines([e], parallel=True, **kw), id="parallel"),
    ],
)
def test_benchmark_rejects_no_iterations(monkeypatch, run) -> None:
    """Zero iterations raises before tegrastats or any engine is started."""
    tegra = MagicMock()
    monkeypatch.setattr(f"{MODULE}.TegraStats", tegra)
    engine = MagicMock()
    with pytest.raises(ValueError, match="iterations must be at least 1"):
        run(engine, iterations=0)
    tegra.assert_not_called()
    engine.mock_execute.assert_not_called()
