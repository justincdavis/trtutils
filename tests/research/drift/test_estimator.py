# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/drift/_estimator.py -- confidence to F1 linear models."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.research.drift import LinearEstimator

pytestmark = pytest.mark.cpu

CONF_A = [0.1, 0.5, 0.9]
CONFIDENCES = {"a": CONF_A, "b": [0.5 * c + 0.15 for c in CONF_A]}
ACCURACIES = {"a": [0.5 * c + 0.25 for c in CONF_A], "b": [1.0 - c for c in CONF_A]}


@pytest.mark.parametrize(
    ("source", "confidence"),
    [
        pytest.param("a", 0.3, id="from-a"),
        pytest.param("b", 0.3, id="from-b"),
    ],
)
def test_exact_lines_recovered(source, confidence) -> None:
    """An exact linear relation is recovered for every target, from every source model."""
    estimator = LinearEstimator(CONFIDENCES, ACCURACIES)
    conf_a = confidence if source == "a" else (confidence - 0.15) / 0.5
    predicted = estimator.predict(source, confidence)
    assert predicted == pytest.approx({"a": 0.5 * conf_a + 0.25, "b": 1.0 - conf_a})
    assert estimator.models == ["a", "b"]


def test_matches_polyfit_on_noisy_data() -> None:
    """The fit equals numpy's degree-1 least squares (sklearn's LinearRegression)."""
    rng = np.random.default_rng(0)
    conf = rng.random(200)
    accuracies = {"a": 0.7 * conf + rng.normal(0, 0.1, 200), "b": rng.random(200)}
    confidences = {"a": conf, "b": rng.random(200)}
    estimator = LinearEstimator(confidences, accuracies)
    for target in ("a", "b"):
        slope, intercept = np.polyfit(conf, accuracies[target], 1)
        assert estimator.predict("a", 0.42)[target] == pytest.approx(slope * 0.42 + intercept)


def test_constant_confidence_predicts_mean() -> None:
    """A source with constant confidence predicts each target's mean."""
    estimator = LinearEstimator({"a": [0.5, 0.5, 0.5]}, {"a": [0.2, 0.4, 0.9]})
    assert estimator.predict("a", 0.9)["a"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("metric", "accuracies", "scale"),
    [
        pytest.param("median", ACCURACIES, 0.5 / 0.5, id="median"),
        pytest.param("max", ACCURACIES, 0.5 / 0.9, id="max"),
        pytest.param("mean", {"a": [0.0] * 3, "b": [0.0] * 3}, 1.0, id="zero-guard"),
    ],
)
def test_scale(metric, accuracies, scale) -> None:
    """The scale is 0.5 over the pooled accuracy, and 1.0 when that is zero."""
    assert LinearEstimator(CONFIDENCES, accuracies, metric).scale == pytest.approx(scale)


@pytest.mark.parametrize(
    ("confidences", "accuracies", "match"),
    [
        pytest.param({"a": [0.1]}, {"b": [0.1]}, "same, non-empty set", id="names"),
        pytest.param({}, {}, "same, non-empty set", id="empty"),
        pytest.param({"a": [0.1, 0.2]}, {"a": [0.1]}, "number of samples", id="lengths"),
        pytest.param({"a": []}, {"a": []}, "number of samples", id="no-samples"),
    ],
)
def test_invalid_inputs(confidences, accuracies, match) -> None:
    """Mismatched models or sample counts raise ValueError."""
    with pytest.raises(ValueError, match=match):
        LinearEstimator(confidences, accuracies)


def test_unknown_model() -> None:
    """Predicting from an unknown model raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model"):
        LinearEstimator(CONFIDENCES, ACCURACIES).predict("c", 0.5)
