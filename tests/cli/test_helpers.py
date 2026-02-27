# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for pure helper functions in trtutils.__main__."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.__main__ import _is_raw_outputs, _parse_shapes_arg


# ---------------------------------------------------------------------------
# _is_raw_outputs
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestIsRawOutputs:
    """Tests for _is_raw_outputs type guard."""

    def test_empty_list_returns_true(self) -> None:
        """An empty list is treated as raw outputs."""
        assert _is_raw_outputs([]) is True

    def test_list_of_ndarrays_returns_true(self) -> None:
        """A flat list of ndarrays is raw output."""
        outputs = [np.zeros((1, 10)), np.ones((1, 5))]
        assert _is_raw_outputs(outputs) is True

    def test_list_of_lists_returns_false(self) -> None:
        """A nested list (batched outputs) is not raw output."""
        outputs = [[np.zeros((1, 10))], [np.ones((1, 5))]]
        assert _is_raw_outputs(outputs) is False

    def test_single_ndarray_returns_true(self) -> None:
        """A single-element list containing one ndarray is raw output."""
        outputs = [np.array([1.0, 2.0, 3.0])]
        assert _is_raw_outputs(outputs) is True

    def test_single_nested_list_returns_false(self) -> None:
        """A single-element nested list is not raw output."""
        outputs = [[np.array([1.0, 2.0])]]
        assert _is_raw_outputs(outputs) is False


# ---------------------------------------------------------------------------
# _parse_shapes_arg
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestParseShapesArg:
    """Tests for _parse_shapes_arg shape specification parser."""

    def test_none_returns_none(self) -> None:
        """None input returns None."""
        assert _parse_shapes_arg(None) is None

    def test_empty_list_returns_none(self) -> None:
        """Empty list returns None."""
        assert _parse_shapes_arg([]) is None

    def test_single_shape_parses_correctly(self) -> None:
        """A single well-formed shape spec parses to a list with one tuple."""
        result = _parse_shapes_arg(["input:1,3,224,224"])
        assert result is not None
        assert len(result) == 1
        assert result[0] == ("input", (1, 3, 224, 224))

    def test_multiple_shapes_parse_correctly(self) -> None:
        """Multiple shape specs parse into corresponding tuples."""
        result = _parse_shapes_arg(["input:1,3,224,224", "mask:1,1,224,224"])
        assert result is not None
        assert len(result) == 2
        assert result[0] == ("input", (1, 3, 224, 224))
        assert result[1] == ("mask", (1, 1, 224, 224))

    def test_missing_colon_raises_value_error(self) -> None:
        """A spec without a colon separator raises ValueError."""
        with pytest.raises(ValueError, match="Invalid --shapes specification"):
            _parse_shapes_arg(["input_1_3_224_224"])

    def test_non_integer_dims_raises_value_error(self) -> None:
        """Non-integer dimension values raise ValueError."""
        with pytest.raises(ValueError, match="Invalid dimension"):
            _parse_shapes_arg(["input:1,three,224,224"])

    def test_empty_dims_raises_value_error(self) -> None:
        """An empty dimension string (after colon) raises ValueError."""
        with pytest.raises(ValueError, match="No dimensions provided"):
            _parse_shapes_arg(["input:"])

    def test_single_dimension_works(self) -> None:
        """A single-dimension shape spec is valid."""
        result = _parse_shapes_arg(["scalar:1"])
        assert result is not None
        assert result[0] == ("scalar", (1,))
