# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/shift/_types.py -- traits, config and JSON round trip."""

from __future__ import annotations

import dataclasses
import json

import pytest

from trtutils.research.shift import (
    ModelTraits,
    ShiftConfig,
    load_characterization,
    reduce_values,
    save_characterization,
)

TRAITS = {
    "big_gpu": ModelTraits(
        latency_ms=12.5,
        energy_mj=140.0,
        load_time_s=1.2,
        load_memory_mb=300.0,
        confidences=[[0.9, 0.5], [], [0.7]],
        ious=[[0.8, 0.6], [], [0.0, 0.4]],
    ),
    "small_dla": ModelTraits(
        latency_ms=4.0,
        energy_mj=20.0,
        load_time_s=0.5,
        load_memory_mb=80.0,
        confidences=[[0.3], [0.2], []],
        ious=[[0.5, 0.0], [], [0.0, 0.0]],
    ),
}


@pytest.mark.cpu
def test_characterization_roundtrip(tmp_path) -> None:
    """save_characterization -> load_characterization returns equal traits in the documented schema."""
    path = tmp_path / "traits.json"
    save_characterization(TRAITS, path)
    assert load_characterization(path) == TRAITS
    raw = json.loads(path.read_text())
    assert set(raw) == set(TRAITS)
    assert set(raw["big_gpu"]) == {f.name for f in dataclasses.fields(ModelTraits)}


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("metric", "confidences", "accuracies"),
    [
        pytest.param("mean", [0.7, 0.0, 0.7], [0.7, 1.0, 0.2], id="mean"),
        pytest.param("median", [0.7, 0.0, 0.7], [0.7, 1.0, 0.2], id="median"),
        pytest.param("min", [0.5, 0.0, 0.7], [0.6, 1.0, 0.0], id="min"),
        pytest.param("max", [0.9, 0.0, 0.7], [0.8, 1.0, 0.4], id="max"),
    ],
)
def test_frame_reductions(metric, confidences, accuracies) -> None:
    """Per-image confidences and accuracies reduce per metric; empty images follow the documented rule."""
    big = TRAITS["big_gpu"]
    assert big.frame_confidences(metric) == pytest.approx(confidences)
    assert big.frame_accuracies(metric) == pytest.approx(accuracies)
    # no ground truth and detections scores 0, no ground truth and no detections scores 1
    assert TRAITS["small_dla"].frame_accuracies(metric)[1] == 0.0


@pytest.mark.cpu
def test_reduce_values_unknown_metric() -> None:
    """An unknown reducer name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown metric"):
        reduce_values([0.1], "mode")


@pytest.mark.cpu
def test_shift_config_presets() -> None:
    """The default config holds the paper's values and shiftdetector() the reference repo's."""
    assert dataclasses.asdict(ShiftConfig()) == {
        "accuracy_threshold": 0.25,
        "momentum": 30,
        "distance_threshold": 0.5,
        "accuracy_weight": 1.0,
        "energy_weight": 0.5,
        "latency_weight": 0.5,
        "similarity": "min",
        "confidence_metric": "mean",
        "accuracy_metric": "mean",
        "num_bins": 10,
        "outlier_percentile": 100.0,
        "min_connects": 0,
    }
    assert dataclasses.asdict(ShiftConfig.shiftdetector()) == {
        "accuracy_threshold": 0.5,
        "momentum": 10,
        "distance_threshold": 1.0,
        "accuracy_weight": 0.85,
        "energy_weight": 0.10,
        "latency_weight": 0.10,
        "similarity": "product",
        "confidence_metric": "max",
        "accuracy_metric": "max",
        "num_bins": 10,
        "outlier_percentile": 80.0,
        "min_connects": 1,
    }


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"similarity": "max"}, "Unknown similarity", id="similarity"),
        pytest.param({"confidence_metric": "mode"}, "Unknown metric", id="confidence-metric"),
        pytest.param({"accuracy_metric": "sum"}, "Unknown metric", id="accuracy-metric"),
        pytest.param({"momentum": 0}, "momentum", id="momentum"),
    ],
)
def test_shift_config_invalid(kwargs, match) -> None:
    """Invalid config values raise ValueError."""
    with pytest.raises(ValueError, match=match):
        ShiftConfig(**kwargs)
