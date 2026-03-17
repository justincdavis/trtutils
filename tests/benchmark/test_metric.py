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
    assert metric.mean == 3.0
    assert metric.median == 3.0
    assert metric.min == 1.0
    assert metric.max == 5.0
    expected_std = stdev(data)
    assert math.isclose(metric.std, expected_std)
    expected_ci95 = 1.96 * expected_std / math.sqrt(len(data))
    assert math.isclose(metric.ci95, expected_ci95)
    s = str(metric)
    assert "std=" in s
    assert "ci95=" in s
    r = repr(metric)
    assert "std=" in r
    assert "ci95=" in r


@pytest.mark.cpu
def test_metric_median_even() -> None:
    """Median is the average of two middle elements for even-length data."""
    metric = Metric([1.0, 2.0, 3.0, 4.0])
    assert metric.median == 2.5


@pytest.mark.cpu
def test_metric_single_element() -> None:
    """Single-element list sets all stats to that value, std=0, ci95=0."""
    metric = Metric([42.0])
    assert metric.mean == 42.0
    assert metric.median == 42.0
    assert metric.min == 42.0
    assert metric.max == 42.0
    assert metric.std == 0.0
    assert metric.ci95 == 0.0


@pytest.mark.cpu
def test_metric_two_elements() -> None:
    """Metric works with exactly two elements and computes std/ci95."""
    data = [1.0, 3.0]
    metric = Metric(data)
    assert metric.mean == 2.0
    assert metric.min == 1.0
    assert metric.max == 3.0
    expected_std = stdev(data)
    assert math.isclose(metric.std, expected_std)
    expected_ci95 = 1.96 * expected_std / math.sqrt(len(data))
    assert math.isclose(metric.ci95, expected_ci95)


@pytest.mark.cpu
def test_metric_integer_data() -> None:
    """Metric accepts integer data."""
    metric = Metric([1, 2, 3])
    assert metric.mean == 2.0


@pytest.mark.cpu
def test_metric_negative_values() -> None:
    """Metric handles negative values correctly."""
    metric = Metric([-3.0, -1.0, -2.0])
    assert metric.min == -3.0
    assert metric.max == -1.0


@pytest.mark.cpu
def test_metric_empty_raises() -> None:
    """Metric raises ValueError when given an empty list."""
    with pytest.raises(ValueError, match="Raw data cannot be empty"):
        Metric([])


@pytest.mark.cpu
def test_metric_raw_identity() -> None:
    """The raw field is the exact same object passed in."""
    data = [10.0, 20.0, 30.0]
    metric = Metric(data)
    assert metric.raw is data


@pytest.mark.cpu
def test_metric_fields() -> None:
    """Metric dataclass has all expected field names."""
    field_names = {f.name for f in dataclasses.fields(Metric)}
    assert field_names == {"raw", "mean", "median", "min", "max", "std", "ci95"}


@pytest.mark.cpu
def test_metric_str_repr() -> None:
    """str/repr start with 'Metric(' and don't expose raw list."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    metric = Metric(data)
    assert str(metric).startswith("Metric(")
    assert repr(metric).startswith("Metric(")
    assert "[1.0, 2.0, 3.0, 4.0, 5.0]" not in str(metric)
    assert "[1.0, 2.0, 3.0, 4.0, 5.0]" not in repr(metric)
