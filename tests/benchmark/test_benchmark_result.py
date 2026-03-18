# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for the BenchmarkResult dataclass in trtutils._benchmark."""

from __future__ import annotations

import dataclasses

import pytest

from trtutils._benchmark import BenchmarkResult, Metric


@pytest.mark.cpu
def test_benchmark_result_latency() -> None:
    """BenchmarkResult stores a Metric and exposes its stats."""
    latency = Metric([0.1, 0.2, 0.3, 0.4, 0.5])
    result = BenchmarkResult(latency)
    assert isinstance(result.latency, Metric)
    assert result.latency.mean == 0.3


@pytest.mark.cpu
def test_benchmark_result_fields() -> None:
    """BenchmarkResult dataclass has a 'latency' field."""
    field_names = {f.name for f in dataclasses.fields(BenchmarkResult)}
    assert field_names == {"latency"}


@pytest.mark.cpu
def test_benchmark_result_str_repr() -> None:
    """str/repr contain 'BenchmarkResult' and 'latency='."""
    result = BenchmarkResult(Metric([0.1, 0.2, 0.3]))
    assert "BenchmarkResult" in str(result)
    assert "latency=" in str(result)
    assert "BenchmarkResult" in repr(result)
    assert "latency=" in repr(result)
