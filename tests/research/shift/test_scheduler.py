# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/shift/_scheduler.py -- NCC and the per-frame scheduler."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from trtutils.research.shift import ModelTraits, ShiftConfig, ShiftScheduler, ncc

FULL_BOX = (0, 0, 64, 64)


def _traits(big_conf, big_iou, small_conf, small_iou, energy=(100.0, 10.0)):
    # one detection per image; big is accurate and expensive, small is cheap
    return {
        "big": ModelTraits(
            latency_ms=10.0,
            energy_mj=energy[0],
            load_time_s=1.0,
            load_memory_mb=100.0,
            confidences=[[c] for c in big_conf],
            ious=[[i] for i in big_iou],
        ),
        "small": ModelTraits(
            latency_ms=2.0,
            energy_mj=energy[1],
            load_time_s=0.5,
            load_memory_mb=10.0,
            confidences=[[c] for c in small_conf],
            ious=[[i] for i in small_iou],
        ),
    }


# small at conf 0.2 predicts big 0.9 / small 0.3; small at conf 0.5 predicts big 0.1 / small 0.3
TRAITS = _traits([0.9, 0.1], [0.9, 0.1], [0.2, 0.5], [0.3, 0.3])


def _image(seed):
    return np.random.default_rng(seed).integers(0, 256, (64, 64, 3), dtype=np.uint8)


def _dets(conf):
    return [(FULL_BOX, conf, 0)]


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("make_pair", "expected"),
    [
        pytest.param(lambda a: (a, a), 1.0, id="identical"),
        pytest.param(lambda a: (a, 255 - a), -1.0, id="inverted"),
        pytest.param(lambda a: (a, a * 0.5 + 10), 1.0, id="affine"),
        pytest.param(lambda a: (a, np.full_like(a, 7)), 0.0, id="constant"),
    ],
)
def test_ncc(make_pair, expected) -> None:
    """ncc() is the zero-mean normalized cross-correlation, 0 when an image is constant."""
    base = np.random.default_rng(0).integers(0, 256, (32, 32)).astype(np.float64)
    image1, image2 = make_pair(base)
    assert ncc(image1, image2) == pytest.approx(expected)


@pytest.mark.cpu
def test_ncc_shape_mismatch() -> None:
    """ncc() on different shapes raises ValueError."""
    with pytest.raises(ValueError, match="same shape"):
        ncc(np.zeros((4, 4)), np.zeros((4, 5)))


@pytest.mark.cpu
def test_scheduler_gate() -> None:
    """The first frame and scene changes reschedule; a stable, confident scene keeps the model."""
    config = ShiftConfig(accuracy_weight=0.0, energy_weight=1.0, latency_weight=0.0)
    scheduler = ShiftScheduler(TRAITS, config)
    image = _image(1)
    # first frame: no previous frame, so the gate fails and the cheapest pair wins
    assert scheduler("big", image, _dets(0.9)) == "small"
    # same frame and box: similarity 1 * confidence 0.9 holds the gate
    assert scheduler("big", image, _dets(0.9)) == "big"
    # a new scene drops the similarity to ~0
    assert scheduler("big", _image(2), _dets(0.9)) == "small"
    # a confident but unseen confidence bin keeps the model when the gate fails
    assert scheduler("small", _image(3), _dets(0.95)) == "small"
    # no detections: confidence 0 never holds the gate, and bin 0 is unseen here
    assert scheduler("big", image, []) == "big"


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("weights", "threshold", "expected"),
    [
        pytest.param((1.0, 0.0, 0.0), 0.25, "big", id="accuracy-only"),
        pytest.param((0.0, 1.0, 0.0), 0.25, "small", id="energy-only"),
        pytest.param((0.0, 0.0, 1.0), 0.25, "small", id="latency-only"),
        # small's 0.3 is below the goal, so only big is a candidate
        pytest.param((0.0, 1.0, 0.0), 0.5, "big", id="threshold-filters"),
        # nothing reaches 0.95, so every predicted pair is a candidate again
        pytest.param((0.0, 1.0, 0.0), 0.95, "small", id="threshold-fallback"),
        # paper weights: small 0.3 + 0.45 + 0.4 beats big 0.9
        pytest.param((1.0, 0.5, 0.5), 0.25, "small", id="paper-weights"),
    ],
)
def test_scheduler_scoring(weights, threshold, expected) -> None:
    """The weighted score of accuracy, inverted energy and inverted latency picks the pair."""
    config = ShiftConfig(
        accuracy_threshold=threshold,
        accuracy_weight=weights[0],
        energy_weight=weights[1],
        latency_weight=weights[2],
    )
    scheduler = ShiftScheduler(TRAITS, config)
    assert scheduler("small", _image(1), _dets(0.2)) == expected


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("momentum", "expected"),
    [
        # the latest prediction alone says big is worse than small
        pytest.param(1, "small", id="momentum-1"),
        # big averages (0.9 + 0.1) / 2 = 0.5, still above small's 0.3
        pytest.param(2, "big", id="momentum-2"),
    ],
)
def test_scheduler_momentum_and_reset(momentum, expected) -> None:
    """Predictions average over the last momentum reschedules, and reset() clears them."""
    config = ShiftConfig(
        momentum=momentum, accuracy_weight=1.0, energy_weight=0.0, latency_weight=0.0
    )
    scheduler = ShiftScheduler(TRAITS, config)
    assert scheduler("small", _image(1), _dets(0.2)) == "big"
    assert scheduler("small", _image(2), _dets(0.5)) == expected
    scheduler.reset()
    assert scheduler("small", _image(3), _dets(0.5)) == "small"


@pytest.mark.cpu
@pytest.mark.parametrize("similarity", ["min", "product"])
def test_scheduler_similarity(similarity) -> None:
    """The gate combines frame NCC and box NCC by min or product."""
    rng = np.random.default_rng(5)
    image1 = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    noise = rng.integers(-80, 80, (64, 64, 3))
    image2 = np.clip(image1.astype(np.int64) + noise, 0, 255).astype(np.uint8)
    box = (8, 8, 56, 56)

    def gray(image, size):
        return cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), size)

    frame_ncc = ncc(gray(image1, (112, 112)), gray(image2, (112, 112)))
    crop1, crop2 = image1[8:56, 8:56], image2[8:56, 8:56]
    box_ncc = ncc(gray(crop1, (24, 24)), gray(crop2, (24, 24)))
    expected = min(frame_ncc, box_ncc) if similarity == "min" else frame_ncc * box_ncc
    assert 0.0 < expected < 1.0

    # both pairs predict above any threshold here, so a failed gate always picks small
    traits = _traits([0.9], [0.95], [0.2], [0.95])

    # the gate (similarity * confidence 0.9) holds just below its value and fails just above
    for offset, keeps in ((-0.01, True), (0.01, False)):
        config = ShiftConfig(
            similarity=similarity,
            accuracy_threshold=expected * 0.9 + offset,
            accuracy_weight=0.0,
            energy_weight=1.0,
            latency_weight=0.0,
        )
        scheduler = ShiftScheduler(traits, config)
        scheduler("big", image1, [(box, 0.9, 0)])
        result = scheduler("big", image2, [(box, 0.9, 0)])
        assert (result == "big") is keeps


@pytest.mark.cpu
def test_scheduler_zero_energy() -> None:
    """Pairs without energy measurements (non-Jetson) score on accuracy and latency only."""
    traits = _traits([0.9, 0.1], [0.9, 0.1], [0.2, 0.5], [0.3, 0.3], energy=(0.0, 0.0))
    config = ShiftConfig(accuracy_weight=0.0, energy_weight=1.0, latency_weight=1.0)
    scheduler = ShiftScheduler(traits, config)
    assert scheduler("small", _image(1), _dets(0.2)) == "small"
    assert scheduler.graph.models == ["big", "small"]


@pytest.mark.cpu
def test_scheduler_unknown_model() -> None:
    """Scheduling for a model that was not characterized raises ValueError."""
    scheduler = ShiftScheduler(TRAITS)
    with pytest.raises(ValueError, match="Unknown model"):
        scheduler("tiny", _image(1), _dets(0.5))
