# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/drift/_types.py -- traits, config and JSON round trip."""

from __future__ import annotations

import dataclasses
import json

import pytest

from trtutils.research.drift import (
    DriftConfig,
    ModelTraits,
    load_characterization,
    save_characterization,
)

pytestmark = pytest.mark.cpu

TRAITS = {
    "big_gpu": ModelTraits(
        input_size=(1280, 1280),
        dla=False,
        latency_ms=[20.0, 22.0, 30.0],
        confidences=[[0.9, 0.5], [], [0.7], [0.4]],
        true_positives=[2, 0, 0, 0],
        ground_truths=[3, 0, 1, 0],
        energy_mj=[200.0, 210.0],
        pair_latency_ms={"small_dla": [24.0, 26.0]},
        pair_energy_mj={"small_dla": [250.0]},
    ),
    "small_dla": ModelTraits(
        input_size=(480, 320),
        dla=True,
        latency_ms=[10.0],
        confidences=[[0.3], [], [], [0.2, 0.6]],
        true_positives=[1, 0, 0, 1],
        ground_truths=[3, 0, 1, 2],
    ),
}


def test_characterization_roundtrip(tmp_path) -> None:
    """Saving then loading gives equal traits, with the dataclass fields as JSON keys."""
    path = tmp_path / "traits.json"
    save_characterization(TRAITS, path)
    raw = json.loads(path.read_text())
    assert set(raw) == set(TRAITS)
    assert set(raw["big_gpu"]) == {f.name for f in dataclasses.fields(ModelTraits)}
    loaded = load_characterization(path)
    assert loaded == TRAITS
    assert isinstance(loaded["small_dla"].input_size, tuple)


def test_frame_f1() -> None:
    """F1 is 2tp / (detections + ground truth), 1.0 for an empty image with no detections."""
    # detections+tp over gt: 2 of 3 -> 0.8; none/none -> 1.0; one wrong -> 0.0; one, no gt -> 0.0
    assert TRAITS["big_gpu"].frame_f1() == pytest.approx([0.8, 1.0, 0.0, 0.0])
    # 1 of 1 over 3 -> 0.5; none/none; none over 1 -> 0.0; 1 of 2 over 2 -> 0.5
    assert TRAITS["small_dla"].frame_f1() == pytest.approx([0.5, 1.0, 0.0, 0.5])


@pytest.mark.parametrize(
    ("metric", "confidences", "latency"),
    [
        pytest.param("mean", [0.7, 0.0, 0.7, 0.4], 24.0, id="mean"),
        pytest.param("max", [0.9, 0.0, 0.7, 0.4], 30.0, id="max"),
        pytest.param("min", [0.5, 0.0, 0.7, 0.4], 20.0, id="min"),
        pytest.param("median", [0.7, 0.0, 0.7, 0.4], 22.0, id="median"),
    ],
)
def test_reductions(metric, confidences, latency) -> None:
    """Confidences and latency samples reduce with the chosen metric."""
    trait = TRAITS["big_gpu"]
    assert trait.frame_confidences(metric) == pytest.approx(confidences)
    assert trait.latency(metric) == pytest.approx(latency)


def test_pair_latency() -> None:
    """Measured pairs reduce their samples; unmeasured pairs are None."""
    assert TRAITS["big_gpu"].pair_latency("small_dla") == pytest.approx(25.0)
    assert TRAITS["big_gpu"].pair_latency("other") is None
    assert TRAITS["small_dla"].pair_latency("big_gpu") is None


def test_config_presets() -> None:
    """The default is the paper's config; upstream() switches the disagreeing fields."""
    paper = DriftConfig()
    upstream = DriftConfig.upstream(["v10m_640"])
    assert (paper.hp_padding, paper.lp_accuracy_goal, paper.prediction_window) == (64, 0.5, 1)
    assert (paper.dense_groups, paper.anneal_counter, paper.lp_latency_test) == (
        "rectangles",
        "frames",
        "sum",
    )
    assert upstream == dataclasses.replace(
        paper,
        hp_padding=128,
        hp_square=True,
        dense_groups="blocks",
        anneal_counter="packs",
        lp_accuracy_goal=0.4,
        prediction_window=10,
        hp_trusted_models=("v10m_640",),
        lp_exclude_hp_model=True,
        lp_latency_test="schedule",
    )


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"counter_decrement": "some"}, id="counter-decrement"),
        pytest.param({"dense_groups": "triangles"}, id="dense-groups"),
        pytest.param({"anneal_counter": "seconds"}, id="anneal-counter"),
        pytest.param({"lp_latency_test": "max"}, id="latency-test"),
        pytest.param({"confidence_metric": "mode"}, id="metric"),
        pytest.param({"hp_padding": -1}, id="padding"),
        pytest.param({"grid_divisor": 0}, id="grid-divisor"),
        pytest.param({"annealing_floor": 1.5}, id="floor"),
        pytest.param({"fps": 0.0}, id="fps"),
        pytest.param({"lp_accuracy_goal": 1.1}, id="goal"),
        pytest.param({"prediction_window": 0}, id="window"),
        pytest.param({"merge_iou": 0.0}, id="merge-iou"),
    ],
)
def test_config_invalid(overrides) -> None:
    """Out-of-range values and unknown choices raise ValueError."""
    with pytest.raises(ValueError):  # noqa: PT011
        DriftConfig(**overrides)


def test_config_trusted_models_become_tuple() -> None:
    """Trusted models given as a list are stored as a tuple."""
    assert DriftConfig(hp_trusted_models=["a", "b"]).hp_trusted_models == ("a", "b")
