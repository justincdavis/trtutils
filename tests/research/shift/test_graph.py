# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/shift/_graph.py -- the standalone confidence graph."""

from __future__ import annotations

import pytest

from trtutils.research.shift import ConfidenceGraph

# two models, four samples, two bins:
# a bins [2, 2, 1, 2] -> node a2 = mean(1.0, 0.8, 0.6) = 0.8, a1 = 0.2
# b bins [2, 1, 1, 2] -> node b2 = mean(1.0, 0.8) = 0.9, b1 = mean(0.1, 0.3) = 0.2
# edge counts: a2->b2 2, a2->b1 1, a1->b1 1, b2->a2 2, b1->a2 1, b1->a1 1
CONFIDENCES = {"a": [0.9, 0.9, 0.4, 0.9], "b": [0.8, 0.3, 0.3, 0.8]}
ACCURACIES = {"a": [1.0, 0.8, 0.2, 0.6], "b": [1.0, 0.1, 0.3, 0.8]}


@pytest.mark.cpu
def test_graph_structure() -> None:
    """Nodes hold mean accuracy per bin and edges hold per-source normalized, inverted counts."""
    graph = ConfidenceGraph(CONFIDENCES, ACCURACIES, num_bins=2)
    assert graph.models == ["a", "b"]
    assert graph.nodes == pytest.approx(
        {("a", 2): 0.8, ("a", 1): 0.2, ("b", 2): 0.9, ("b", 1): 0.2},
    )
    edges = graph.edges
    assert edges[("a", 2)] == pytest.approx({("b", 2): 0.0, ("b", 1): 0.5})
    assert edges[("a", 1)] == pytest.approx({("b", 1): 0.0})
    assert edges[("b", 1)] == pytest.approx({("a", 2): 0.0, ("a", 1): 0.0})


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("model", "confidence", "threshold", "expected"),
    [
        # a2 reaches b2 at 0, b1 at 0.5 and a1 (via b1) at 0.5: weight ((1 - 0.5) / 1)^2 = 0.25
        pytest.param("a", 0.9, 1.0, {"a": 0.85 / 1.25, "b": 0.95 / 1.25}, id="far-nodes-weighted"),
        # b1 and a1 sit at the threshold, so their weight is the 1e-8 floor
        pytest.param("a", 0.9, 0.5, {"a": 0.8, "b": 0.9}, id="far-nodes-at-threshold"),
        pytest.param("a", 0.9, 0.4, {"a": 0.8, "b": 0.9}, id="far-nodes-excluded"),
        # a1 -> b1 -> a2 -> b2 are all free, so every node is reached at distance 0
        pytest.param("a", 0.4, 0.4, {"a": 0.5, "b": 0.55}, id="multi-hop"),
        pytest.param("b", 0.75, 0.4, {"a": 0.8, "b": 0.9}, id="other-source"),
        # bin 0 (no detection) was never seen
        pytest.param("a", 0.0, 1.0, {}, id="unseen-bin"),
    ],
)
def test_graph_predict(model, confidence, threshold, expected) -> None:
    """predict() averages reachable nodes per model, weighted by distance to the threshold."""
    graph = ConfidenceGraph(CONFIDENCES, ACCURACIES, num_bins=2, distance_threshold=threshold)
    prediction = graph.predict(model, confidence)
    assert set(prediction) == set(expected)
    for name, value in expected.items():
        assert prediction[name] == pytest.approx(value, abs=1e-6)


@pytest.mark.cpu
def test_graph_outlier_percentile_and_min_connects() -> None:
    """Counts above the percentile are clamped, and edges at or below min_connects are dropped."""
    clamped = ConfidenceGraph(CONFIDENCES, ACCURACIES, num_bins=2, outlier_percentile=50.0)
    # a2 counts {b2: 2, b1: 1}, median 1.5: b2 clamps to cost 0, b1 costs 1 - 1 / 1.5
    assert clamped.edges[("a", 2)] == pytest.approx({("b", 2): 0.0, ("b", 1): 1.0 / 3.0})

    pruned = ConfidenceGraph(CONFIDENCES, ACCURACIES, num_bins=2, min_connects=1)
    assert pruned.edges[("a", 2)] == pytest.approx({("b", 2): 0.0})
    assert pruned.edges[("b", 1)] == {}
    assert pruned.predict("b", 0.3) == pytest.approx({"b": 0.2})


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        pytest.param(0.0, 0, id="zero"),
        pytest.param(1e-6, 1, id="tiny"),
        pytest.param(0.3, 3, id="edge-float-safe"),
        pytest.param(0.31, 4, id="above-edge"),
        pytest.param(1.0, 10, id="one"),
        pytest.param(1.5, 10, id="clamp-high"),
        pytest.param(-0.5, 0, id="clamp-low"),
    ],
)
def test_graph_bin(confidence, expected) -> None:
    """bin() maps a confidence to ceil(confidence * num_bins), clamped to [0, num_bins]."""
    graph = ConfidenceGraph({"m": [0.5]}, {"m": [0.5]})
    assert graph.bin(confidence) == expected


@pytest.mark.cpu
def test_graph_single_model() -> None:
    """A graph of one model predicts only that model's bin accuracy."""
    graph = ConfidenceGraph({"m": [0.15, 0.95]}, {"m": [0.1, 0.9]})
    assert graph.predict("m", 0.92) == pytest.approx({"m": 0.9})
    assert graph.edges == {}


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("confidences", "accuracies", "kwargs", "match"),
    [
        pytest.param({"a": [0.1]}, {"b": [0.1]}, {}, "same, non-empty set", id="model-mismatch"),
        pytest.param({}, {}, {}, "same, non-empty set", id="empty"),
        pytest.param({"a": [0.1, 0.2]}, {"a": [0.1]}, {}, "same number", id="length-mismatch"),
        pytest.param(
            {"a": [0.1], "b": [0.1, 0.2]},
            {"a": [0.1], "b": [0.1, 0.2]},
            {},
            "same number",
            id="models-differ-in-samples",
        ),
        pytest.param({"a": [0.1]}, {"a": [0.1]}, {"num_bins": 0}, "num_bins", id="bins"),
        pytest.param(
            {"a": [0.1]}, {"a": [0.1]}, {"distance_threshold": 0.0}, "distance", id="threshold"
        ),
        pytest.param(
            {"a": [0.1]}, {"a": [0.1]}, {"outlier_percentile": 0.0}, "outlier", id="percentile"
        ),
    ],
)
def test_graph_invalid(confidences, accuracies, kwargs, match) -> None:
    """Mismatched inputs and out-of-range parameters raise ValueError."""
    with pytest.raises(ValueError, match=match):
        ConfidenceGraph(confidences, accuracies, **kwargs)


@pytest.mark.cpu
def test_graph_predict_unknown_model() -> None:
    """predict() on a model that is not in the graph raises ValueError."""
    graph = ConfidenceGraph(CONFIDENCES, ACCURACIES)
    with pytest.raises(ValueError, match="Unknown model"):
        graph.predict("c", 0.5)
