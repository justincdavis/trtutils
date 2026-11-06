# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/profiling/_optimize.py -- profiled layer chunking."""

from __future__ import annotations

import pytest

from trtutils.profiling._optimize import _build_chunk_map

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize(
    ("layers", "keys", "times"),
    [
        pytest.param([], [], {}, id="empty"),
        pytest.param(
            [("reformat", 1.0), ("pad", 0.5)],
            ["__prologue__"],
            {"__prologue__": 1.5},
            id="prologue-only",
        ),
        pytest.param(
            [("reformat", 1.0), ("/a", 2.0), ("copy", 0.25), ("/b", 3.0)],
            ["__prologue__", "/a", "/b"],
            {"__prologue__": 1.0, "/a": 2.25, "/b": 3.0},
            id="prologue-and-anchors",
        ),
        pytest.param(
            [("/a", 1.0), ("x", 1.0), ("y", 1.0), ("/b", 2.0)],
            ["/a", "/b"],
            {"/a": 3.0, "/b": 2.0},
            id="non-anchors-join-previous",
        ),
        pytest.param(
            [("/a_myl0_1", 1.0), ("tmp", 1.0), ("/a_myl0_2", 2.0)],
            ["/a"],
            {"/a": 4.0},
            id="myelin-anchors-merge",
        ),
        pytest.param(
            [("/a", 1.0), ("/b", 2.0), ("/a", 4.0)],
            ["/a", "/b"],
            {"/a": 5.0, "/b": 2.0},
            id="repeated-anchor-accumulates",
        ),
        pytest.param(
            [("a", 1.0), ("/a", 1.0)],
            ["__prologue__", "/a"],
            {"__prologue__": 1.0, "/a": 1.0},
            id="no-leading-slash-is-not-anchor",
        ),
    ],
)
def test_build_chunk_map(layers, keys, times) -> None:
    """Layers are grouped under the most recent anchor, in first-seen order."""
    out_keys, out_times = _build_chunk_map(layers)
    assert out_keys == keys
    assert out_times == times
