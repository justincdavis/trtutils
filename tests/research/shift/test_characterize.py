# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/shift/_characterize.py and _shift.py -- on real engines."""

from __future__ import annotations

import gc
import time
import weakref
from functools import partial

import pytest

from tests.conftest import DATA_DIR
from trtutils import FLAGS
from trtutils.image import Detector
from trtutils.research.shift import (
    ModelTraits,
    Shift,
    ShiftConfig,
    characterize,
    load_characterization,
    save_characterization,
)

YOLOV10_ONNX = DATA_DIR / "yolov10" / "yolov10n_640.onnx"


@pytest.fixture(scope="module")
def engine(build_test_engine):
    """A cached YOLOv10n engine, skipping if the ONNX is missing."""
    if not YOLOV10_ONNX.exists():
        pytest.skip(f"missing {YOLOV10_ONNX}")
    return build_test_engine(YOLOV10_ONNX)


@pytest.fixture(scope="module")
def frames(images):
    """Two real test frames with objects."""
    return [images["horse"].array, images["people"].array]


def _assert_same_detections(actual, expected) -> None:
    assert len(actual) == len(expected)
    for (box, score, cls), (ref_box, ref_score, ref_cls) in zip(actual, expected):
        assert cls == ref_cls
        assert score == pytest.approx(ref_score, abs=1e-3)
        assert all(abs(a - b) <= 2 for a, b in zip(box, ref_box))


def test_characterize_roundtrip(engine, frames, tmp_path, monkeypatch) -> None:
    """Characterize -> save -> load -> Shift.end2end on a real engine under two preprocessors."""
    # tegrastats writes its temp file to the working directory
    monkeypatch.chdir(tmp_path)
    reference = Detector(engine, warmup=True)
    ground_truth = [[d[0] for d in reference.end2end(frame)] for frame in frames]
    del reference
    assert all(len(boxes) > 0 for boxes in ground_truth)

    models = {
        "yolov10n_cuda": partial(Detector, engine, warmup=True),
        "yolov10n_cpu": partial(Detector, engine, preprocessor="cpu", warmup=True),
    }
    traits = characterize(models, frames, ground_truth, iterations=200, warmup_iterations=10)
    assert list(traits) == list(models)
    for trait in traits.values():
        assert [len(ious) for ious in trait.ious] == [len(boxes) for boxes in ground_truth]
        assert [len(confs) > 0 for confs in trait.confidences] == [True, True]
        # ground truth is this engine's own output
        assert min(trait.frame_accuracies("min")) > 0.9
        assert trait.latency_ms > 0.0
        assert trait.load_time_s > 0.0
        assert trait.load_memory_mb > 0.0
        assert (trait.energy_mj > 0.0) is FLAGS.IS_JETSON

    path = tmp_path / "traits.json"
    save_characterization(traits, path)
    loaded = load_characterization(path)
    assert loaded == traits
    shift = Shift(models, loaded)
    for frame in frames:
        assert len(shift.end2end(frame)) > 0


@pytest.mark.parametrize(
    "preprocessor",
    [
        pytest.param("cuda", id="cuda"),
        pytest.param("cpu", id="cpu"),
        pytest.param("trt", id="trt"),
    ],
)
@pytest.mark.parametrize(
    "cuda_graph",
    [
        pytest.param(False, id="no-graph"),
        pytest.param(True, id="graph", marks=pytest.mark.cuda_graph),
    ],
)
def test_background_load_matches_reference(engine, frames, preprocessor, cuda_graph) -> None:
    """Detectors built in the background thread match a main-thread reference, and evicted ones are freed."""
    reference = Detector(engine, preprocessor=preprocessor, cuda_graph=cuda_graph, warmup=True)
    expected = [reference.end2end(frame) for frame in frames]
    del reference

    # every confidence bin is characterized; a goal of 1.0 never holds the gate and leaves
    # every pair a candidate, so energy alone picks the pair
    names = ["first", "second", "third"]
    traits = {
        name: ModelTraits(
            latency_ms=1.0,
            energy_mj=energy,
            load_time_s=0.1,
            load_memory_mb=100.0,
            confidences=[[k / 10] for k in range(11)],
            ious=[[accuracy]] * 11,
        )
        for name, energy, accuracy in zip(names, (30.0, 10.0, 20.0), (0.6, 0.5, 0.5))
    }
    config = ShiftConfig(
        accuracy_threshold=1.0, accuracy_weight=0.0, energy_weight=1.0, latency_weight=0.0
    )
    factory = partial(
        Detector, engine, preprocessor=preprocessor, cuda_graph=cuda_graph, warmup=True
    )
    shift = Shift(dict.fromkeys(names, factory), traits, config, memory_budget=200.0)
    assert shift.current == "first"

    def run_until(condition) -> None:
        deadline = time.monotonic() + 60.0
        index = 0
        while True:
            _assert_same_detections(shift.end2end(frames[index % 2]), expected[index % 2])
            index += 1
            if condition() and index > 2:
                return
            assert time.monotonic() < deadline, "background load did not finish"

    run_until(lambda: shift.current == "second")
    assert shift.loaded == ["first", "second"]

    first = weakref.ref(shift._loaded["first"])
    shift._load("third")
    assert shift.loaded == ["second"]
    gc.collect()
    assert first() is None
    run_until(lambda: "third" in shift.loaded)
    assert shift.current == "second"
