# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for the Metric dataclass in trtutils._benchmark."""

from __future__ import annotations

import pytest


@pytest.mark.cpu
class TestMetricInit:
    """Test Metric construction and basic data handling."""

    def test_basic_construction(self) -> None:
        """Metric can be constructed from a list of floats."""
        from trtutils._benchmark import Metric

        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        metric = Metric(data)
        assert metric is not None

    def test_single_element(self) -> None:
        """Metric works with a single-element list."""
        from trtutils._benchmark import Metric

        metric = Metric([42.0])
        assert metric.mean == 42.0
        assert metric.median == 42.0
        assert metric.min == 42.0
        assert metric.max == 42.0

    def test_integer_data(self) -> None:
        """Metric accepts integer data."""
        from trtutils._benchmark import Metric

        metric = Metric([1, 2, 3])
        assert metric.mean == 2.0
        assert metric.min == 1
        assert metric.max == 3

    def test_empty_raises_value_error(self) -> None:
        """Metric raises ValueError when given an empty list."""
        from trtutils._benchmark import Metric

        with pytest.raises(ValueError, match="Raw data cannot be empty"):
            Metric([])

    def test_negative_values(self) -> None:
        """Metric handles negative values correctly."""
        from trtutils._benchmark import Metric

        metric = Metric([-3.0, -1.0, -2.0])
        assert metric.min == -3.0
        assert metric.max == -1.0

    def test_raw_preserves_input_data(self) -> None:
        """The raw field preserves the original input data."""
        from trtutils._benchmark import Metric

        data = [10.0, 20.0, 30.0]
        metric = Metric(data)
        assert metric.raw == data

    def test_two_elements(self) -> None:
        """Metric works with exactly two elements."""
        from trtutils._benchmark import Metric

        metric = Metric([1.0, 3.0])
        assert metric.mean == 2.0
        assert metric.min == 1.0
        assert metric.max == 3.0


@pytest.mark.cpu
class TestMetricStatistics:
    """Test computed statistics on Metric."""

    def test_mean(self) -> None:
        """Mean is computed correctly."""
        from trtutils._benchmark import Metric

        metric = Metric([1.0, 2.0, 3.0, 4.0, 5.0])
        assert metric.mean == 3.0

    def test_median_odd_count(self) -> None:
        """Median is the middle element for odd-length data."""
        from trtutils._benchmark import Metric

        metric = Metric([5.0, 1.0, 3.0])
        assert metric.median == 3.0

    def test_median_even_count(self) -> None:
        """Median is the average of two middle elements for even-length data."""
        from trtutils._benchmark import Metric

        metric = Metric([1.0, 2.0, 3.0, 4.0])
        assert metric.median == 2.5

    def test_min_max(self) -> None:
        """Min and max are computed correctly."""
        from trtutils._benchmark import Metric

        metric = Metric([7.0, 2.0, 9.0, 1.0, 5.0])
        assert metric.min == 1.0
        assert metric.max == 9.0


@pytest.mark.cpu
class TestMetricStringRepresentation:
    """Test __str__ and __repr__ of Metric."""

    def test_str_format(self) -> None:
        """__str__ includes formatted mean, median, min, max."""
        from trtutils._benchmark import Metric

        metric = Metric([1.0, 2.0, 3.0, 4.0, 5.0])
        s = str(metric)
        assert "mean=3.000" in s
        assert "median=3.000" in s
        assert "min=1.000" in s
        assert "max=5.000" in s

    def test_repr_format(self) -> None:
        """__repr__ includes mean, median, min, max."""
        from trtutils._benchmark import Metric

        metric = Metric([1.0, 2.0, 3.0, 4.0, 5.0])
        r = repr(metric)
        assert "mean=3.0" in r
        assert "median=3.0" in r

    def test_str_starts_with_metric(self) -> None:
        """__str__ starts with 'Metric('."""
        from trtutils._benchmark import Metric

        metric = Metric([1.0, 2.0])
        assert str(metric).startswith("Metric(")

    def test_raw_not_in_str(self) -> None:
        """Raw data list is not exposed in __str__ or __repr__."""
        from trtutils._benchmark import Metric

        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        metric = Metric(data)
        assert "[1.0, 2.0, 3.0, 4.0, 5.0]" not in str(metric)
        assert "[1.0, 2.0, 3.0, 4.0, 5.0]" not in repr(metric)


@pytest.mark.cpu
class TestMetricIsDataclass:
    """Test that Metric is a proper dataclass."""

    def test_is_dataclass(self) -> None:
        """Metric should be a dataclass instance."""
        import dataclasses

        from trtutils._benchmark import Metric

        metric = Metric([1.0])
        assert dataclasses.is_dataclass(metric)

    def test_has_expected_fields(self) -> None:
        """Metric dataclass has all expected field names."""
        import dataclasses

        from trtutils._benchmark import Metric

        field_names = {f.name for f in dataclasses.fields(Metric)}
        expected = {"raw", "mean", "median", "min", "max"}
        assert expected == field_names

    def test_raw_preserves_data(self) -> None:
        """The raw field contains the exact input data."""
        from trtutils._benchmark import Metric

        data = [0.1, 0.2, 0.3]
        metric = Metric(data)
        assert metric.raw is data
