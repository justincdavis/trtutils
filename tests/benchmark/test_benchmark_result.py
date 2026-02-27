# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for the BenchmarkResult dataclass in trtutils._benchmark."""

from __future__ import annotations

import pytest


@pytest.mark.cpu
class TestBenchmarkResultInit:
    """Test BenchmarkResult construction."""

    def test_basic_construction(self) -> None:
        """BenchmarkResult can be constructed with a Metric instance."""
        from trtutils._benchmark import BenchmarkResult, Metric

        latency = Metric([0.1, 0.2, 0.3])
        result = BenchmarkResult(latency)
        assert result is not None

    def test_latency_is_metric_instance(self) -> None:
        """The latency field is a Metric instance."""
        from trtutils._benchmark import BenchmarkResult, Metric

        latency = Metric([0.1, 0.2, 0.3, 0.4, 0.5])
        result = BenchmarkResult(latency)
        assert isinstance(result.latency, Metric)
        assert result.latency.mean == 0.3


@pytest.mark.cpu
class TestBenchmarkResultIsDataclass:
    """Test that BenchmarkResult is a proper dataclass."""

    def test_is_dataclass(self) -> None:
        """BenchmarkResult should be a dataclass instance."""
        import dataclasses

        from trtutils._benchmark import BenchmarkResult, Metric

        result = BenchmarkResult(Metric([1.0]))
        assert dataclasses.is_dataclass(result)

    def test_has_expected_fields(self) -> None:
        """BenchmarkResult dataclass has a 'latency' field."""
        import dataclasses

        from trtutils._benchmark import BenchmarkResult

        field_names = {f.name for f in dataclasses.fields(BenchmarkResult)}
        assert "latency" in field_names


@pytest.mark.cpu
class TestBenchmarkResultStringRepresentation:
    """Test __str__ and __repr__ of BenchmarkResult."""

    def test_str_contains_benchmark_result(self) -> None:
        """__str__ contains 'BenchmarkResult'."""
        from trtutils._benchmark import BenchmarkResult, Metric

        result = BenchmarkResult(Metric([0.1, 0.2, 0.3]))
        assert "BenchmarkResult" in str(result)

    def test_repr_contains_benchmark_result(self) -> None:
        """__repr__ contains 'BenchmarkResult'."""
        from trtutils._benchmark import BenchmarkResult, Metric

        result = BenchmarkResult(Metric([0.1, 0.2, 0.3]))
        assert "BenchmarkResult" in repr(result)

    def test_str_contains_latency(self) -> None:
        """__str__ includes 'latency='."""
        from trtutils._benchmark import BenchmarkResult, Metric

        result = BenchmarkResult(Metric([0.1, 0.2, 0.3]))
        assert "latency=" in str(result)

    def test_repr_contains_latency(self) -> None:
        """__repr__ includes 'latency='."""
        from trtutils._benchmark import BenchmarkResult, Metric

        result = BenchmarkResult(Metric([0.1, 0.2, 0.3]))
        assert "latency=" in repr(result)
