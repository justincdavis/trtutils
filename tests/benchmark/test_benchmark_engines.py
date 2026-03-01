# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for benchmark_engines() in trtutils._benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from .conftest import ITERS, WARMUP_ITERS

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.gpu]


# ============================================================================
# Sequential benchmarking (parallel=False / default)
# ============================================================================


class TestBenchmarkEnginesSequential:
    """Test benchmark_engines() in sequential (non-parallel) mode."""

    def test_returns_list(self, engine_path: Path) -> None:
        """benchmark_engines returns a list."""
        from trtutils import benchmark_engines

        results = benchmark_engines(
            [engine_path, engine_path],
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
        )
        assert isinstance(results, list)

    def test_result_count_matches_engines(self, engine_path: Path) -> None:
        """Number of results matches number of engines passed in."""
        from trtutils import benchmark_engines

        num_engines = 2
        results = benchmark_engines(
            [engine_path] * num_engines,
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
        )
        assert len(results) == num_engines

    def test_all_results_are_benchmark_result(self, engine_path: Path) -> None:
        """Every result is a BenchmarkResult instance."""
        from trtutils import BenchmarkResult, benchmark_engines

        results = benchmark_engines(
            [engine_path, engine_path],
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
        )
        assert all(isinstance(r, BenchmarkResult) for r in results)

    def test_correct_iteration_count(self, engine_path: Path) -> None:
        """Each result has the correct number of raw latency samples."""
        from trtutils import benchmark_engines

        results = benchmark_engines(
            [engine_path, engine_path],
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
        )
        for result in results:
            assert len(result.latency.raw) == ITERS

    def test_single_engine(self, engine_path: Path) -> None:
        """benchmark_engines works with a single engine in the list."""
        from trtutils import BenchmarkResult, benchmark_engines

        results = benchmark_engines(
            [engine_path],
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
        )
        assert len(results) == 1
        assert isinstance(results[0], BenchmarkResult)


# ============================================================================
# Parallel benchmarking (parallel=True)
# ============================================================================


class TestBenchmarkEnginesParallel:
    """Test benchmark_engines() in parallel mode."""

    def test_returns_single_result(self, engine_path: Path) -> None:
        """Parallel mode returns a list with exactly one result."""
        from trtutils import benchmark_engines

        results = benchmark_engines(
            [engine_path, engine_path],
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
            parallel=True,
        )
        assert len(results) == 1

    def test_result_is_benchmark_result(self, engine_path: Path) -> None:
        """The single parallel result is a BenchmarkResult instance."""
        from trtutils import BenchmarkResult, benchmark_engines

        results = benchmark_engines(
            [engine_path, engine_path],
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
            parallel=True,
        )
        assert isinstance(results[0], BenchmarkResult)

    def test_iteration_count(self, engine_path: Path) -> None:
        """Parallel result has the correct number of raw latency samples."""
        from trtutils import benchmark_engines

        results = benchmark_engines(
            [engine_path, engine_path],
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
            parallel=True,
        )
        assert len(results[0].latency.raw) == ITERS

    def test_positive_latency(self, engine_path: Path) -> None:
        """All parallel latency values are positive."""
        from trtutils import benchmark_engines

        results = benchmark_engines(
            [engine_path, engine_path],
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
            parallel=True,
        )
        assert all(v > 0 for v in results[0].latency.raw)


# ============================================================================
# Tuple input (engine_path, dla_core)
# ============================================================================


class TestBenchmarkEnginesTupleInput:
    """Test benchmark_engines() with tuple-format engine specifications."""

    def test_tuple_with_dla_core(self, engine_path: Path) -> None:
        """benchmark_engines accepts (path, dla_core) tuple entries."""
        from trtutils import BenchmarkResult, benchmark_engines

        results = benchmark_engines(
            [(engine_path, None)],
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
        )
        assert len(results) == 1
        assert isinstance(results[0], BenchmarkResult)

    def test_string_paths_in_tuples(self, engine_path: Path) -> None:
        """benchmark_engines accepts string paths inside tuples."""
        from trtutils import BenchmarkResult, benchmark_engines

        results = benchmark_engines(
            [(str(engine_path), None)],
            iterations=ITERS,
            warmup_iterations=WARMUP_ITERS,
        )
        assert len(results) == 1
        assert isinstance(results[0], BenchmarkResult)
