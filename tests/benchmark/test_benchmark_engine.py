# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for benchmark_engine() in trtutils._benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from .conftest import ITERS, WARMUP_ITERS

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.gpu]


# ============================================================================
# benchmark_engine with path input
# ============================================================================


class TestBenchmarkEnginePath:
    """Test benchmark_engine() when given an engine path."""

    def test_returns_benchmark_result(self, engine_path: Path) -> None:
        """benchmark_engine returns a BenchmarkResult instance."""
        from trtutils import BenchmarkResult, benchmark_engine

        result = benchmark_engine(
            engine_path,
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
        )
        assert isinstance(result, BenchmarkResult)

    def test_latency_is_metric(self, engine_path: Path) -> None:
        """The latency field is a Metric instance."""
        from trtutils import Metric, benchmark_engine

        result = benchmark_engine(
            engine_path,
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
        )
        assert isinstance(result.latency, Metric)

    def test_raw_length_matches_iterations(self, engine_path: Path) -> None:
        """Raw latency data length matches the requested iteration count."""
        from trtutils import benchmark_engine

        result = benchmark_engine(
            engine_path,
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
        )
        assert len(result.latency.raw) == ITERS

    def test_values_positive(self, engine_path: Path) -> None:
        """All raw latency values are positive."""
        from trtutils import benchmark_engine

        result = benchmark_engine(
            engine_path,
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
        )
        assert all(v > 0 for v in result.latency.raw)

    def test_stats_populated(self, engine_path: Path) -> None:
        """Mean, median, min, max are all populated and positive."""
        from trtutils import benchmark_engine

        result = benchmark_engine(
            engine_path,
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
        )
        assert result.latency.mean > 0
        assert result.latency.median > 0
        assert result.latency.min > 0
        assert result.latency.max > 0

    def test_string_path_works(self, engine_path: Path) -> None:
        """benchmark_engine accepts a string path."""
        from trtutils import BenchmarkResult, benchmark_engine

        result = benchmark_engine(
            str(engine_path),
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
        )
        assert isinstance(result, BenchmarkResult)

    def test_verbose_flag(self, engine_path: Path) -> None:
        """benchmark_engine accepts verbose=True without error."""
        from trtutils import BenchmarkResult, benchmark_engine

        result = benchmark_engine(
            engine_path,
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
            verbose=True,
        )
        assert isinstance(result, BenchmarkResult)

    def test_warmup_disabled(self, engine_path: Path) -> None:
        """benchmark_engine works with warmup=False."""
        from trtutils import BenchmarkResult, benchmark_engine

        result = benchmark_engine(
            engine_path,
            iterations=ITERS,
            warmup_iterations=0,
            warmup=False,
        )
        assert isinstance(result, BenchmarkResult)
        assert len(result.latency.raw) == ITERS


# ============================================================================
# benchmark_engine with TRTEngine object
# ============================================================================


class TestBenchmarkEngineObject:
    """Test benchmark_engine() when given a pre-created TRTEngine."""

    def test_accepts_engine_object(self, engine_path: Path) -> None:
        """benchmark_engine accepts a TRTEngine instance directly."""
        from trtutils import BenchmarkResult, TRTEngine, benchmark_engine

        eng = TRTEngine(engine_path, warmup=False)
        result = benchmark_engine(eng, iterations=ITERS)
        assert isinstance(result, BenchmarkResult)
        del eng

    def test_engine_object_with_warmup(self, engine_path: Path) -> None:
        """benchmark_engine warms up a pre-created engine when warmup=True."""
        from trtutils import BenchmarkResult, TRTEngine, benchmark_engine

        eng = TRTEngine(engine_path, warmup=False)
        result = benchmark_engine(
            eng,
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
            warmup=True,
        )
        assert isinstance(result, BenchmarkResult)
        assert len(result.latency.raw) == ITERS
        del eng

    def test_engine_object_without_warmup(self, engine_path: Path) -> None:
        """benchmark_engine skips warmup on pre-created engine when warmup=False."""
        from trtutils import BenchmarkResult, TRTEngine, benchmark_engine

        eng = TRTEngine(engine_path, warmup=False)
        result = benchmark_engine(eng, iterations=ITERS, warmup=False)
        assert isinstance(result, BenchmarkResult)
        assert len(result.latency.raw) == ITERS
        del eng


# ============================================================================
# benchmark_engine iteration counts
# ============================================================================


class TestBenchmarkEngineIterations:
    """Test benchmark_engine() with various iteration counts."""

    def test_single_iteration(self, engine_path: Path) -> None:
        """benchmark_engine works with a single iteration."""
        from trtutils import benchmark_engine

        result = benchmark_engine(engine_path, iterations=1, warmup=False)
        assert len(result.latency.raw) == 1

    def test_custom_iterations(self, engine_path: Path) -> None:
        """benchmark_engine respects a custom iteration count."""
        from trtutils import benchmark_engine

        custom_iters = 5
        result = benchmark_engine(
            engine_path,
            iterations=custom_iters,
            warmup=False,
        )
        assert len(result.latency.raw) == custom_iters
