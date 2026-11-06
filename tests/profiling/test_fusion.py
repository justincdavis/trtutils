# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/profiling/_fusion.py -- layer fusion name mapping."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from trtutils._log import LOG
from trtutils.profiling._fusion import (
    build_fused_layer_map,
    resolve_fused_layer_value,
    strip_myelin_suffix,
)

pytestmark = pytest.mark.cpu

_ONNX = ["/x", "/a", "/b", "/c", "/y"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        pytest.param("/model.0/conv/Conv_myl0_4", "/model.0/conv/Conv", id="suffix"),
        pytest.param("/a_myl12_345", "/a", id="multi-digit"),
        pytest.param("/a", "/a", id="no-suffix"),
        pytest.param("/a_myl0_4/b", "/a_myl0_4/b", id="suffix-not-at-end"),
        pytest.param("/a_myl0", "/a_myl0", id="single-number"),
        pytest.param("/a_myl_4", "/a_myl_4", id="missing-digits"),
        pytest.param("", "", id="empty"),
    ],
)
def test_strip_myelin_suffix(name, expected) -> None:
    """Only a trailing _myl<N>_<N> is removed."""
    assert strip_myelin_suffix(name) == expected


@pytest.mark.parametrize(
    ("profiled", "onnx", "expected"),
    [
        pytest.param([], _ONNX, {}, id="empty"),
        pytest.param(["/a", "/b"], _ONNX, {}, id="unfused-plain"),
        pytest.param(
            ["/a + /b + /c"],
            None,
            dict.fromkeys(("/a", "/b", "/c"), ("/a + /b + /c", 3)),
            id="gpu-three-no-onnx",
        ),
        pytest.param(
            ["/a + /b"],
            _ONNX,
            dict.fromkeys(("/a", "/b"), ("/a + /b", 2)),
            id="gpu-two",
        ),
        pytest.param(
            ["/a + /b", "/c + /y"],
            _ONNX,
            {"/a": ("/a + /b", 2), "/b": ("/a + /b", 2), "/c": ("/c + /y", 2), "/y": ("/c + /y", 2)},
            id="gpu-two-groups",
        ),
        pytest.param(
            ["{ForeignNode[/a.../c]}"],
            _ONNX,
            dict.fromkeys(("/a", "/b", "/c"), ("{ForeignNode[/a.../c]}", 3)),
            id="dla-range",
        ),
        pytest.param(
            ["{ForeignNode[/x.../x]}"],
            _ONNX,
            {"/x": ("{ForeignNode[/x.../x]}", 1)},
            id="dla-range-same-endpoints",
        ),
        pytest.param(
            ["{ForeignNode[/a.../missing]}"],
            _ONNX,
            {},
            id="dla-range-last-missing-ignored",
        ),
        pytest.param(
            ["{ForeignNode[/missing.../c]}"],
            _ONNX,
            {},
            id="dla-range-first-missing-ignored",
        ),
        pytest.param(
            ["{ForeignNode[/b]}"],
            _ONNX,
            {"/b": ("{ForeignNode[/b]}", 1)},
            id="dla-single",
        ),
        pytest.param(["{ForeignNode[/missing]}"], _ONNX, {}, id="dla-single-missing"),
        pytest.param(["{ForeignNode[/a.../c]}"], None, {}, id="dla-range-no-onnx"),
        pytest.param(["{ForeignNode[/b]}"], None, {}, id="dla-single-no-onnx"),
        pytest.param(["{ForeignNode[/a.../b.../c]}"], _ONNX, {}, id="dla-three-part-ignored"),
        pytest.param(
            ["/a_myl0_4"],
            _ONNX,
            {"/a": ("/a_myl0_4", 1)},
            id="myelin-suffix",
        ),
        pytest.param(["/missing_myl0_4"], _ONNX, {}, id="myelin-base-not-in-onnx"),
        pytest.param(["/a_myl0_4"], None, {}, id="myelin-no-onnx"),
        pytest.param(
            ["/x", "/a + /b", "{ForeignNode[/c]}", "/y_myl1_2"],
            _ONNX,
            {
                "/a": ("/a + /b", 2),
                "/b": ("/a + /b", 2),
                "/c": ("{ForeignNode[/c]}", 1),
                "/y": ("/y_myl1_2", 1),
            },
            id="mixed",
        ),
    ],
)
def test_build_fused_layer_map(profiled, onnx, expected) -> None:
    """Each fusion format maps constituents to (fused_name, count)."""
    assert build_fused_layer_map(profiled, onnx) == expected


@pytest.mark.parametrize(
    ("layer", "values", "fused_map", "expected"),
    [
        pytest.param("/a", {"/a": 2.0, "/b": 1.0}, {}, 2.0, id="exact"),
        pytest.param("/a", {"/a": 2.0, "f": 9.0}, {"/a": ("f", 3)}, 2.0, id="exact-beats-fused"),
        pytest.param("/a", {"f": 6.0}, {"/a": ("f", 3)}, 2.0, id="fused-split-evenly"),
        pytest.param("/a", {"f": 5.0}, {"/a": ("f", 1)}, 5.0, id="fused-single"),
        pytest.param("/a", {"/a_myl0_4": 1.5}, {}, 1.5, id="substring-fallback"),
        pytest.param("/a", {"other": 1.0}, {"/a": ("f", 2)}, None, id="fused-name-not-profiled"),
        pytest.param("/a", {}, {}, None, id="missing-empty"),
        pytest.param("/a", {"/b": 1.0}, {}, None, id="missing"),
    ],
)
def test_resolve_fused_layer_value(layer, values, fused_map, expected) -> None:
    """Resolution order is exact, fused split, substring, then None."""
    assert resolve_fused_layer_value(layer, values, fused_map) == expected


@pytest.mark.parametrize(
    ("verbose", "values", "warned"),
    [
        pytest.param(True, {}, True, id="verbose-missing"),
        pytest.param(None, {}, False, id="quiet-missing"),
        pytest.param(True, {"/a": 1.0}, False, id="verbose-found"),
    ],
)
def test_resolve_fused_layer_value_warning(verbose, values, warned) -> None:
    """A warning naming the layer and label is logged only for verbose misses."""
    with patch.object(LOG, "warning") as warning:
        resolve_fused_layer_value("/a", values, {}, verbose=verbose, label="FP16")
    if warned:
        warning.assert_called_once_with("Layer /a not found in FP16 profile")
    else:
        warning.assert_not_called()
