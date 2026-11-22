# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/shift/_characterize.py and _shift.py -- streaming and real engines."""

from __future__ import annotations

import gc
import time
import weakref
from functools import partial

import cv2
import numpy as np
import pytest

from tests.conftest import DATA_DIR
from trtutils import FLAGS
from trtutils.image import Detector
from trtutils.research.shift import (
    ModelTraits,
    Shift,
    ShiftConfig,
    _characterize,
    characterize,
    load_characterization,
    save_characterization,
)

YOLOV10_ONNX = DATA_DIR / "yolov10" / "yolov10n_640.onnx"
NUM_IMAGES = 5
SCALES = {"big": 1.0, "small": 0.5}


class FakeDetector:
    """Finds one box scaled by its pair, scored by the frame's brightness."""

    engine = None

    def __init__(self, scale) -> None:
        """Remember the pair's box scale."""
        self.scale = scale

    def end2end(self, image):
        height, width = image.shape[:2]
        box = (0, 0, round(width * self.scale), round(height * self.scale))
        return [(box, float(image.mean()) / 255.0 * self.scale, 0)]


class DecodeTracker:
    """Wraps cv2.imread, counting decodes and the peak number of live decoded images."""

    def __init__(self, imread) -> None:
        """Start with no decodes."""
        self.imread = imread
        self.calls = 0
        self.live = 0
        self.peak = 0

    def __call__(self, path) -> np.ndarray:
        image = self.imread(path)
        self.calls += 1
        self.live += 1
        self.peak = max(self.peak, self.live)
        weakref.finalize(image, self._release)
        return image

    def _release(self) -> None:
        self.live -= 1


@pytest.fixture
def labeled_images(tmp_path):
    """Lossless PNGs of different sizes, their decoded arrays, and two boxes per image."""
    rng = np.random.default_rng(0)
    paths, arrays, ground_truth = [], [], []
    for index in range(NUM_IMAGES):
        height, width = 24 + 4 * index, 32 + 8 * index
        path = tmp_path / f"{index}.png"
        cv2.imwrite(str(path), rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
        paths.append(path)
        arrays.append(cv2.imread(str(path)))
        ground_truth.append([(0, 0, width, height), (0, 0, width // 2, height // 2)])
    return paths, arrays, ground_truth


@pytest.fixture
def _fake_measurements(monkeypatch) -> None:
    """Constant memory and benchmark results, so characterize runs without a GPU."""
    monkeypatch.setattr(_characterize, "_used_memory_mb", lambda: 0.0)
    monkeypatch.setattr(_characterize, "_engine_memory_mb", lambda _engine: 1.0)
    monkeypatch.setattr(_characterize, "_measure", lambda *_args, **_kwargs: (2.0, 3.0))


def _fake_models(on_build=None):
    def _factory(scale):
        if on_build is not None:
            on_build()
        return FakeDetector(scale)

    return {name: partial(_factory, scale) for name, scale in SCALES.items()}


@pytest.mark.cpu
@pytest.mark.usefixtures("_fake_measurements")
@pytest.mark.parametrize(
    "source",
    [
        pytest.param("paths", id="paths"),
        pytest.param("strings", id="strings"),
        pytest.param("generators", id="generators"),
    ],
)
def test_characterize_streams_paths(labeled_images, source, monkeypatch) -> None:
    """Paths decode one at a time per pair, nothing up front, and match array results."""
    paths, arrays, ground_truth = labeled_images
    expected = characterize(_fake_models(), arrays, ground_truth)
    for name, trait in expected.items():
        assert [c for (c,) in trait.confidences] == [
            d[1] for a in arrays for d in FakeDetector(SCALES[name]).end2end(a)
        ]

    images = {
        "paths": paths,
        "strings": [str(path) for path in paths],
        "generators": (path for path in paths),
    }[source]
    labels = (boxes for boxes in ground_truth) if source == "generators" else ground_truth
    tracker = DecodeTracker(cv2.imread)
    monkeypatch.setattr(cv2, "imread", tracker)
    builds: list[int] = []
    traits = characterize(_fake_models(lambda: builds.append(tracker.calls)), images, labels)

    # warmup build, then one build per pair, each after the previous pair's full pass
    assert builds == [0, 0, NUM_IMAGES]
    assert tracker.calls == NUM_IMAGES * len(SCALES)
    # the frame being decoded plus the one it replaces
    assert tracker.peak <= 2
    assert list(traits) == list(expected)
    for name, trait in traits.items():
        assert trait.confidences == expected[name].confidences
        assert trait.ious == expected[name].ious
        assert (trait.latency_ms, trait.energy_mj, trait.load_memory_mb) == (2.0, 3.0, 1.0)


@pytest.mark.cpu
@pytest.mark.usefixtures("_fake_measurements")
@pytest.mark.parametrize(
    ("models", "num_images", "num_labels", "generators", "error", "match"),
    [
        pytest.param({}, 2, 2, False, ValueError, "at least one model", id="no-models"),
        pytest.param(None, 0, 0, False, ValueError, "at least one image", id="no-images"),
        pytest.param(None, 2, 1, False, ValueError, "2 images but 1", id="mismatch"),
        pytest.param(None, 1, 2, True, ValueError, "1 images but 2", id="mismatch-generators"),
        pytest.param(None, -1, 1, False, FileNotFoundError, "Cannot read", id="missing-file"),
    ],
)
def test_characterize_invalid(
    labeled_images, tmp_path, models, num_images, num_labels, generators, error, match
) -> None:
    """No models or images, mismatched lengths, and unreadable paths raise."""
    paths, _arrays, ground_truth = labeled_images
    images = paths[:num_images] if num_images >= 0 else [tmp_path / "missing.png"]
    labels = ground_truth[:num_labels]
    if generators:
        images, labels = iter(images), iter(labels)
    with pytest.raises(error, match=match):
        characterize(models if models is not None else _fake_models(), images, labels)


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
