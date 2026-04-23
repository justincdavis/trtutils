# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for trtutils.jetson._benchmark -- JetsonBenchmarkResult, benchmark_engine, benchmark_engines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.gpu, pytest.mark.jetson]

ITERS = 100
WARMUP_ITERS = 5


# ============================================================================
# JetsonBenchmarkResult dataclass
# ============================================================================


class TestJetsonBenchmarkResult:
    """Test JetsonBenchmarkResult attributes, str, and repr."""

    def test_attributes(self) -> None:
        """Result dataclass exposes latency, power_draw, and energy Metric fields."""
        pytest.importorskip("jetsontools")
        from trtutils import Metric
        from trtutils.jetson import JetsonBenchmarkResult

        data = [0.1, 0.2, 0.3, 0.4, 0.5]
        latency = Metric(data)
        power_draw = Metric(data)
        energy = Metric(data)
        result = JetsonBenchmarkResult(latency, power_draw, energy)

        assert result.latency is latency
        assert result.power_draw is power_draw
        assert result.energy is energy
        assert result.latency.mean == pytest.approx(0.3)
        assert result.power_draw.mean == pytest.approx(0.3)
        assert result.energy.mean == pytest.approx(0.3)

    def test_str_representation(self) -> None:
        """str(result) contains the class name and field names."""
        pytest.importorskip("jetsontools")
        from trtutils import Metric
        from trtutils.jetson import JetsonBenchmarkResult

        data = [1.0, 2.0]
        result = JetsonBenchmarkResult(Metric(data), Metric(data), Metric(data))

        text = str(result)
        assert "JetsonBenchmarkResult(latency=" in text
        assert "power_draw=" in text
        assert "energy=" in text

    def test_repr_representation(self) -> None:
        """repr(result) contains the class name and field names."""
        pytest.importorskip("jetsontools")
        from trtutils import Metric
        from trtutils.jetson import JetsonBenchmarkResult

        data = [1.0, 2.0]
        result = JetsonBenchmarkResult(Metric(data), Metric(data), Metric(data))

        text = repr(result)
        assert "JetsonBenchmarkResult(latency=" in text
        assert "power_draw=" in text
        assert "energy=" in text


# ============================================================================
# benchmark_engine
# ============================================================================


class TestBenchmarkEngine:
    """Test benchmark_engine with a single engine."""

    def test_single_engine(self, engine_path: Path) -> None:
        """Benchmarking a single engine returns a JetsonBenchmarkResult with valid metrics."""
        pytest.importorskip("jetsontools")
        from trtutils import Metric
        from trtutils.jetson import JetsonBenchmarkResult, benchmark_engine

        result = benchmark_engine(
            engine_path,
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

    def test_string_path_input(self, engine_path: Path) -> None:
        """benchmark_engine accepts a string path and produces valid results."""
        pytest.importorskip("jetsontools")
        from trtutils.jetson import JetsonBenchmarkResult, benchmark_engine

        result = benchmark_engine(
            str(engine_path),
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
            tegra_interval=1,
        )

        assert isinstance(result, JetsonBenchmarkResult)
        assert len(result.latency.raw) == ITERS


# ============================================================================
# benchmark_engines
# ============================================================================


class TestBenchmarkEngines:
    """Test benchmark_engines with sequential and parallel modes."""

    def test_sequential(self, engine_path: Path) -> None:
        """Benchmarking multiple engines sequentially returns one result per engine."""
        pytest.importorskip("jetsontools")
        from trtutils.jetson import JetsonBenchmarkResult, benchmark_engines

        num_engines = 2
        results = benchmark_engines(
            [engine_path] * num_engines,
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
            tegra_interval=1,
        )

        assert len(results) == num_engines
        assert all(isinstance(r, JetsonBenchmarkResult) for r in results)
        for result in results:
            assert len(result.latency.raw) == ITERS
            assert len(result.power_draw.raw) > 0
            assert len(result.energy.raw) > 0

    def test_parallel(self, engine_path: Path) -> None:
        """Benchmarking engines in parallel returns a single combined result."""
        pytest.importorskip("jetsontools")
        from trtutils.jetson import JetsonBenchmarkResult, benchmark_engines

        results = benchmark_engines(
            [engine_path, engine_path],
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
            tegra_interval=1,
            parallel=True,
        )

        assert len(results) == 1
        assert isinstance(results[0], JetsonBenchmarkResult)
        assert len(results[0].latency.raw) == ITERS
        assert len(results[0].power_draw.raw) > 0
        assert len(results[0].energy.raw) > 0
