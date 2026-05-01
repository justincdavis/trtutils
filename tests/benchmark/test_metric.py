# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for the Metric dataclass in trtutils._benchmark."""

from __future__ import annotations

import dataclasses
import math
from statistics import stdev

import pytest

from trtutils._benchmark import Metric


@pytest.mark.cpu
def test_metric_statistics() -> None:
    """Metric computes mean, median, min, max, std, ci95 correctly."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    metric = Metric(data)
    assert metric.raw is data
    assert metric.mean == 3.0
    assert metric.median == 3.0
    assert metric.min == 1.0
    assert metric.max == 5.0
    expected_std = stdev(data)
    assert math.isclose(metric.std, expected_std)
    expected_ci95 = 1.96 * expected_std / math.sqrt(len(data))
    assert math.isclose(metric.ci95, expected_ci95)
    field_names = {f.name for f in dataclasses.fields(Metric)}
    assert field_names == {"raw", "mean", "median", "min", "max", "std", "ci95"}
    s = str(metric)
    assert s.startswith("Metric(")
    assert "std=" in s
    assert "ci95=" in s
    assert "[1.0, 2.0, 3.0, 4.0, 5.0]" not in s
    r = repr(metric)
    assert r.startswith("Metric(")
    assert "std=" in r
    assert "ci95=" in r
    assert "[1.0, 2.0, 3.0, 4.0, 5.0]" not in r


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("data", "expected_mean", "expected_median", "expected_min", "expected_max"),
    [
        pytest.param([42.0], 42.0, 42.0, 42.0, 42.0, id="single-element"),
        pytest.param([1.0, 3.0], 2.0, 2.0, 1.0, 3.0, id="two-elements"),
        pytest.param([1.0, 2.0, 3.0, 4.0], 2.5, 2.5, 1.0, 4.0, id="even-length"),
        pytest.param([1, 2, 3], 2.0, 2.0, 1, 3, id="integer-data"),
        pytest.param([-3.0, -1.0, -2.0], -2.0, -2.0, -3.0, -1.0, id="negative-values"),
    ],
)
def test_metric_edge_cases(data, expected_mean, expected_median, expected_min, expected_max) -> None:
    """Metric handles edge-case inputs correctly."""
    metric = Metric(data)
    assert metric.mean == expected_mean
    assert metric.median == expected_median
    assert metric.min == expected_min
    assert metric.max == expected_max


@pytest.mark.cpu
def test_metric_single_element_std() -> None:
    """Single-element Metric has std=0 and ci95=0."""
    metric = Metric([42.0])
    assert metric.std == 0.0
    assert metric.ci95 == 0.0


@pytest.mark.cpu
def test_metric_empty_raises() -> None:
    """Metric raises ValueError when given an empty list."""
    with pytest.raises(ValueError, match="Raw data cannot be empty"):
        Metric([])
