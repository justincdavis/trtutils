# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/drift/_scheduler.py -- region finding and Algorithm 2."""

from __future__ import annotations

import dataclasses

import pytest

from trtutils.research.drift import (
    DriftConfig,
    DriftScheduler,
    FrameResult,
    ModelTraits,
    Schedule,
    hp_region,
)

pytestmark = pytest.mark.cpu

SIZE = (1920, 1080)
REGION = (100, 100, 900, 700)  # 800x600: 640 inputs keep the scale, 320 inputs do not
SMALL_REGION = (100, 100, 400, 400)  # 300x300: both keep the scale


def _trait(size, latency, f1, *, dla=False, pairs=None):
    # two images, four detections (confidence 0.2 then 0.8) and four ground-truth boxes
    # each, so F1 = tp / 4; a constant F1 predicts that F1 from any confidence
    f1 = (f1, f1) if isinstance(f1, float) else f1
    return ModelTraits(
        input_size=(size, size),
        dla=dla,
        latency_ms=[latency],
        confidences=[[0.2] * 4, [0.8] * 4],
        true_positives=[round(value * 4) for value in f1],
        ground_truths=[4, 4],
        pair_latency_ms={name: [value] for name, value in (pairs or {}).items()},
    )


TRAITS = {
    "base": _trait(1280, 30.0, 1.0),
    "m640": _trait(640, 10.0, 1.0, pairs={"dla": 12.5}),
    "n320": _trait(320, 4.0, 0.75),
    "dla": _trait(480, 12.0, 0.5, dla=True),
}
# all F1s pool to a median of 0.875, so the goal 0.25 compares predictions with 0.4375
CONFIG = DriftConfig(lp_accuracy_goal=0.25)
BASELINE = Schedule("base", None, parallel=False, latency_ms=30.0)


def _previous(box=(100, 100, 200, 200), conf=0.8, schedule=BASELINE):
    detections = [(box, conf, 0)]
    return FrameResult(detections, detections, [], schedule)


def _with(name, **changes: object):
    return {**TRAITS, name: dataclasses.replace(TRAITS[name], **changes)}


def test_hp_region() -> None:
    """The region pads the box around priority detections only, clipped to the frame."""
    detections = [
        ((100, 100, 200, 200), 0.9, 0),
        ((300, 50, 400, 120), 0.8, 0),
        ((10, 10, 20, 20), 0.9, 1),
        ((1850, 1000, 1900, 1070), 0.9, 2),
    ]
    assert hp_region(detections, {0}, SIZE, 64) == (36, 0, 464, 264)
    assert hp_region(detections, {0, 2}, SIZE, 64) == (36, 0, 1920, 1080)
    assert hp_region(detections, {5}, SIZE) is None
    assert hp_region(detections, {1}, SIZE, 0, min_area=100.0) is None
    assert hp_region(detections, {1}, SIZE, 0, min_area=99.0) == (10, 10, 20, 20)


def test_picks_fastest_schedule() -> None:
    """The fastest accurate HP model pairs with the LP model of the fastest schedule (DLA, concurrent)."""
    scheduler = DriftScheduler(TRAITS, config=CONFIG)
    assert scheduler.baseline == BASELINE
    assert scheduler(_previous(), REGION, SIZE) == Schedule(
        "m640", "dla", parallel=True, latency_ms=12.5
    )


@pytest.mark.parametrize(
    ("traits", "config", "previous", "region"),
    [
        pytest.param(TRAITS, CONFIG, None, REGION, id="first-frame"),
        pytest.param(TRAITS, CONFIG, FrameResult([], [], [], BASELINE), REGION, id="no-detections"),
        pytest.param(TRAITS, CONFIG, _previous(), None, id="no-region"),
        pytest.param(TRAITS, CONFIG, _previous(), (100, 100, 1300, 900), id="region-too-big"),
        pytest.param(
            TRAITS, DriftConfig(lp_accuracy_goal=1.0), _previous(), REGION, id="goal-unreachable"
        ),
        pytest.param(_with("base", latency_ms=[13.0]), CONFIG, _previous(), REGION, id="not-faster"),
    ],
)
def test_falls_back_to_baseline(traits, config, previous, region) -> None:
    """Every give-up condition of Algorithm 2 returns the baseline on the full frame."""
    scheduler = DriftScheduler(traits, config=config)
    assert scheduler(previous, region, SIZE) == scheduler.baseline


@pytest.mark.parametrize(
    ("traits", "config", "expected"),
    [
        pytest.param(
            _with("dla", latency_ms=[25.0]),
            CONFIG,
            Schedule("m640", "n320", parallel=False, latency_ms=14.0),
            id="sum-test-rejects-slow-dla",
        ),
        pytest.param(
            _with("dla", latency_ms=[25.0]),
            dataclasses.replace(CONFIG, lp_latency_test="schedule"),
            Schedule("m640", "dla", parallel=True, latency_ms=12.5),
            id="schedule-test-uses-measured-pair",
        ),
        pytest.param(
            _with("m640", pair_latency_ms={}),
            CONFIG,
            Schedule("m640", "dla", parallel=True, latency_ms=12.0),
            id="unmeasured-pair-uses-max",
        ),
        pytest.param(
            TRAITS,
            DriftConfig(lp_accuracy_goal=0.5),
            Schedule("m640", "m640", parallel=False, latency_ms=20.0),
            id="lp-may-repeat-hp",
        ),
        pytest.param(
            TRAITS,
            DriftConfig(lp_accuracy_goal=0.5, lp_exclude_hp_model=True),
            BASELINE,
            id="lp-excludes-hp",
        ),
    ],
)
def test_config_switches(traits, config, expected) -> None:
    """The latency test, unmeasured pairs and HP/LP exclusion change the chosen schedule."""
    assert DriftScheduler(traits, config=config)(_previous(), REGION, SIZE) == expected


@pytest.mark.parametrize(
    ("trusted", "hp"),
    [
        pytest.param((), "m640", id="accuracy-checked"),
        pytest.param(("n320",), "n320", id="trusted-skips-check"),
    ],
)
def test_trusted_hp_models(trusted, hp) -> None:
    """Trusted HP models skip the accuracy check (n320 is predicted below the baseline)."""
    config = dataclasses.replace(CONFIG, hp_trusted_models=trusted)
    assert DriftScheduler(TRAITS, config=config)(_previous(), SMALL_REGION, SIZE).hp == hp


@pytest.mark.parametrize(
    ("region", "expected_hp"),
    [
        pytest.param((0, 0, 1066, 500), "m640", id="at-threshold"),
        pytest.param((0, 0, 1068, 500), "base", id="past-threshold"),
    ],
)
def test_side_length_rule(region, expected_hp) -> None:
    """A model qualifies when imgsz / long side >= (1280 / 1920) * (1 - box size / 1920)."""
    # 192 px boxes relax the threshold to 0.6, so 640 px covers a long side up to 1066.7
    previous = _previous(box=(100, 100, 292, 292))
    assert DriftScheduler(TRAITS, config=CONFIG)(previous, region, SIZE).hp == expected_hp


def test_prediction_window_and_reset() -> None:
    """Predictions average over the window; reset clears them."""
    # n320 is as accurate as the baseline only when the confidence is high (0.8)
    traits = _with("n320", true_positives=[1, 4])
    scheduler = DriftScheduler(traits, config=dataclasses.replace(CONFIG, prediction_window=2))
    assert scheduler(_previous(conf=0.8), SMALL_REGION, SIZE).hp == "n320"
    scheduler.reset()
    assert scheduler(_previous(conf=0.2), SMALL_REGION, SIZE).hp == "m640"
    # the average of 0.25 and 1.0 keeps n320 below the baseline
    assert scheduler(_previous(conf=0.8), SMALL_REGION, SIZE).hp == "m640"
    scheduler.reset()
    assert scheduler(_previous(conf=0.8), SMALL_REGION, SIZE).hp == "n320"


def test_default_baseline_and_errors() -> None:
    """The default baseline is the most accurate GPU model; DLA, unknown or no GPU models raise."""
    traits = _with("base", true_positives=[3, 3])
    assert DriftScheduler(traits).baseline.hp == "m640"
    assert DriftScheduler(TRAITS, baseline="n320").baseline == Schedule(
        "n320", None, parallel=False, latency_ms=4.0
    )
    with pytest.raises(ValueError, match="must be one of the GPU models"):
        DriftScheduler(TRAITS, baseline="dla")
    with pytest.raises(ValueError, match="must be one of the GPU models"):
        DriftScheduler(TRAITS, baseline="missing")
    with pytest.raises(ValueError, match="at least one GPU model"):
        DriftScheduler({"dla": TRAITS["dla"]})
