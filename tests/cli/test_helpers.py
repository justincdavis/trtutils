# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for pure helper functions in trtutils.__main__."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.__main__ import _is_raw_outputs, _parse_shapes_arg

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        pytest.param([], True, id="empty"),
        pytest.param([np.zeros((1, 10)), np.ones((1, 5))], True, id="flat-ndarrays"),
        pytest.param([np.array([1.0, 2.0, 3.0])], True, id="single-ndarray"),
        pytest.param([[np.zeros((1, 10))], [np.ones((1, 5))]], False, id="nested"),
        pytest.param([[np.array([1.0, 2.0])]], False, id="single-nested"),
    ],
)
def test_is_raw_outputs(outputs, expected) -> None:
    """Type guard correctly distinguishes flat vs nested output lists."""
    assert _is_raw_outputs(outputs) is expected


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        pytest.param(None, None, id="none"),
        pytest.param([], None, id="empty"),
        pytest.param(["input:1,3,224,224"], [("input", (1, 3, 224, 224))], id="single"),
        pytest.param(
            ["input:1,3,224,224", "mask:1,1,224,224"],
            [("input", (1, 3, 224, 224)), ("mask", (1, 1, 224, 224))],
            id="multiple",
        ),
        pytest.param(["scalar:1"], [("scalar", (1,))], id="single-dim"),
    ],
)
def test_parse_shapes_arg_valid(arg, expected) -> None:
    """Well-formed shape specs parse to the expected tuples."""
    assert _parse_shapes_arg(arg) == expected


@pytest.mark.parametrize(
    ("arg", "match"),
    [
        pytest.param(["input_1_3_224_224"], "Invalid --shapes specification", id="missing-colon"),
        pytest.param(["input:1,three,224,224"], "Invalid dimension", id="non-integer"),
        pytest.param(["input:"], "No dimensions provided", id="empty-dims"),
    ],
)
def test_parse_shapes_arg_invalid(arg, match) -> None:
    """Malformed shape specs raise ValueError with a descriptive message."""
    with pytest.raises(ValueError, match=match):
        _parse_shapes_arg(arg)
