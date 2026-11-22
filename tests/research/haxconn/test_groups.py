# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/haxconn/_groups.py -- transition points and layer groups."""

from __future__ import annotations

import pytest

from trtutils.inspect import LayerInfo
from trtutils.research.haxconn import LayerGroup, fused_spans, group_layers, transition_points
from trtutils.research.haxconn._groups import constant_layers, match_engine_layers

pytestmark = pytest.mark.cpu


def _layer(index, name, layer_type, inputs, outputs, *, dla=True) -> LayerInfo:
    return LayerInfo(index, name, layer_type, None, 0, 0, dla, tuple(outputs), tuple(inputs))


def _resnet_block() -> list[LayerInfo]:
    # conv -> relu -> [conv -> conv] + skip -> add -> relu, then a GPU-only head
    return [
        _layer(0, "/c0/Conv", "CONVOLUTION", ["in"], ["c0"]),
        _layer(1, "/r0/Relu", "ACTIVATION", ["c0"], ["r0"]),
        _layer(2, "/c1/Conv", "CONVOLUTION", ["r0"], ["c1"]),
        _layer(3, "/c2/Conv", "CONVOLUTION", ["c1"], ["c2"]),
        _layer(4, "/add/Add", "ELEMENTWISE", ["c2", "r0"], ["add"]),
        _layer(5, "/r1/Relu", "ACTIVATION", ["add"], ["r1"]),
        _layer(6, "/shape/Constant", "CONSTANT", [], ["shape"]),
        _layer(7, "/flat/Reshape", "SHUFFLE", ["r1", "shape"], ["flat"]),
        _layer(8, "/topk/TopK", "TOPK", ["flat"], ["out"], dla=False),
    ]


def test_constant_layers_reach_a_fixed_point() -> None:
    """A shuffle of a constant counts as constant even when it comes after its consumer."""
    types = ["SHUFFLE", "ELEMENTWISE", "CONSTANT"]
    inputs = [["k"], ["x", "k_shaped"], []]
    outputs = [["k_shaped"], ["y"], ["k"]]
    assert constant_layers(types, inputs, outputs) == {0, 2}


def test_transition_points_skip_connections_and_constants() -> None:
    """Cuts need exactly one forward data tensor; skips block them and constants do not count."""
    layers = _resnet_block()
    # 3 and 4 lie inside the skip connection; 7 has the constant shape crossing too
    assert transition_points(layers) == [1, 2, 5, 6, 7, 8]
    assert transition_points(layers, spans=[(0, 1), (5, 7)]) == [2, 5, 8]


def test_transition_points_reject_backward_edges() -> None:
    """A data tensor produced after its consumer makes the cut it spans invalid."""
    layers = [
        _layer(0, "a", "CONVOLUTION", ["in"], ["a"]),
        _layer(1, "b", "ELEMENTWISE", ["a", "late"], ["b"]),
        _layer(2, "c", "CONVOLUTION", ["a"], ["late"]),
    ]
    assert transition_points(layers) == [1]


def test_fused_spans_parse_tensorrt_names() -> None:
    """Fused engine layers in +, || and PWN() form and subgraphs map to the layers they run."""
    layers = _resnet_block()
    engine = [
        "Reformatting CopyNode for Input Tensor 0 to /c0/Conv + /r0/Relu",
        "/c0/Conv + /r0/Relu",
        "/c1/Conv || /c2/Conv",
        "PWN(PWN(/add/Add), /r1/Relu)",
        "{ForeignNode[/c1/Conv.../add/Add]}",
        "/topk/TopK",
    ]
    assert fused_spans(engine, layers) == [(0, 1), (0, 1), (2, 3), (4, 5), (2, 4)]
    assert match_engine_layers(engine, layers)[4] == [2, 3, 4]


@pytest.mark.parametrize(
    ("max_groups", "weights", "expected"),
    [
        pytest.param(
            None,
            None,
            [
                LayerGroup(0, 1, dla=True),
                LayerGroup(2, 4, dla=True),
                LayerGroup(5, 5, dla=True),
                LayerGroup(6, 7, dla=True),
                LayerGroup(8, 8, dla=False),
            ],
            id="every-cut",
        ),
        pytest.param(
            2,
            [5.0, 5.0, 1.0, 1.0, 1.0, 9.0, 0.0, 9.0, 1.0],
            [LayerGroup(0, 7, dla=True), LayerGroup(8, 8, dla=False)],
            id="merge-cheapest-alike",
        ),
        pytest.param(1, None, [LayerGroup(0, 8, dla=False)], id="one-group"),
    ],
)
def test_group_layers(max_groups, weights, expected) -> None:
    """Groups cover every layer, constants-only groups join a neighbour, merging keeps DLA runs."""
    spans = [(0, 1)]
    assert group_layers(_resnet_block(), weights, spans, max_groups) == expected


def test_group_layers_rejects_bad_input() -> None:
    """An empty network or a cap below one raises."""
    with pytest.raises(ValueError, match="at least one layer"):
        group_layers([])
    with pytest.raises(ValueError, match="max_groups must be at least 1"):
        group_layers(_resnet_block(), max_groups=0)
