# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for the BenchmarkResult dataclass in trtutils._benchmark."""

from __future__ import annotations

import dataclasses

import pytest

from trtutils._benchmark import BenchmarkResult, Metric


@pytest.mark.cpu
def test_benchmark_result() -> None:
    """BenchmarkResult stores a Metric, exposes stats, and has correct fields/repr."""
    latency = Metric([0.1, 0.2, 0.3, 0.4, 0.5])
    result = BenchmarkResult(latency)
    assert isinstance(result.latency, Metric)
    assert result.latency.mean == 0.3
    field_names = {f.name for f in dataclasses.fields(BenchmarkResult)}
    assert field_names == {"latency"}
    assert "BenchmarkResult" in str(result)
    assert "latency=" in str(result)
    assert "BenchmarkResult" in repr(result)
    assert "latency=" in repr(result)
