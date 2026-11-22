# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/drift/_characterize.py and _drift.py -- real and fake engines."""

from __future__ import annotations

import dataclasses
from collections import Counter
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tests.conftest import DATA_DIR, ENGINES_DIR
from trtutils import FLAGS
from trtutils.image import Detector
from trtutils.models import YOLOX
from trtutils.research.drift import (
    Drift,
    DriftConfig,
    ModelTraits,
    characterize,
    load_characterization,
    save_characterization,
)

YOLOV10N_640 = DATA_DIR / "yolov10" / "yolov10n_640.onnx"
YOLOV10N_320 = DATA_DIR / "yolov10" / "yolov10n_320.onnx"
YOLOXS_320 = DATA_DIR / "yolox" / "yoloxs_320.onnx"
PRIORITY = {0, 17}  # COCO person and horse
VALUES = [1, 2, 3, 4]  # fake labeled images are filled with one value each
FAKE_TRUTH = [((0, 0, 10, 10), 1)]
ITERATIONS = 6


class FakeDetector:
    """Detects two boxes whose confidence and class come from the image's fill value."""

    def __init__(self, name, log, *, dla=False, scale=1.0) -> None:
        """Stand in for one model; every call is logged as ("run", name, value)."""
        self.name = name
        self.log = log
        self.scale = scale
        self.input_shape = (64, 64)
        self.engine = SimpleNamespace(dla_core=0 if dla else None)

    def end2end(self, image):
        value = int(image[0, 0, 0])
        self.log.append(("run", self.name, value))
        return [((0, 0, 10, 10), self.scale * value / 10, value % 2), ((20, 20, 30, 30), 0.25, 0)]


@pytest.fixture
def fake(monkeypatch):
    """Fake GPU and DLA models sharing one call log, with timing replaced by one call per step."""

    def _measure(step, iterations, _warmup_iterations):
        for index in range(iterations):
            step(index)
        return [1.0] * iterations, []

    monkeypatch.setattr("trtutils.research.drift._characterize._measure", _measure)
    log = []
    models = {
        "gpu": partial(FakeDetector, "gpu", log),
        "dla": partial(FakeDetector, "dla", log, dla=True, scale=0.5),
    }
    return models, log


def _image(value):
    return np.full((8, 8, 3), value, dtype=np.uint8)


@pytest.fixture(scope="module")
def engines(build_test_engine):
    """Cached YOLOv10n engines at 640 and 320, skipping if an ONNX is missing."""
    for onnx in (YOLOV10N_640, YOLOV10N_320):
        if not onnx.exists():
            pytest.skip(f"missing {onnx}")
    return {"n640": build_test_engine(YOLOV10N_640), "n320": build_test_engine(YOLOV10N_320)}


@pytest.fixture(scope="module")
def dla_engine():
    """A cached YOLOX-s 320 engine (with NMS) on DLA core 0, skipping without a DLA or the ONNX."""
    if not FLAGS.HAS_DLA:
        pytest.skip("DLA tests require DLA hardware")
    if not YOLOXS_320.exists():
        pytest.skip(f"missing {YOLOXS_320}")
    version = ".".join(str(v) for v in FLAGS.TRT_VERSION)
    path = ENGINES_DIR / f"yoloxs_320_nms_dla0_trt{version}.engine"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        YOLOX.build(YOLOXS_320, path, dla_core=0, default_device="dla", gpu_fallback=True)
    return path


@pytest.fixture(scope="module")
def frames(images):
    """Two real test frames with objects."""
    return [images["horse"].array, images["people"].array]


@pytest.fixture(scope="module")
def canvas(images):
    """The people image in the corner of a black 1080p frame, so the HP region is small."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    people = images["people"].array
    frame[: people.shape[0], : people.shape[1]] = people
    return frame


def _assert_same_detections(actual, expected) -> None:
    assert len(actual) == len(expected)
    for (box, score, cls), (ref_box, ref_score, ref_cls) in zip(actual, expected):
        assert cls == ref_cls
        assert score == pytest.approx(ref_score, abs=1e-3)
        assert all(abs(a - b) <= 2 for a, b in zip(box, ref_box))


def test_characterize_roundtrip(engines, frames, tmp_path) -> None:
    """Characterize -> save -> load -> Drift.end2end on real GPU engines."""
    reference = Detector(engines["n640"], warmup=True)
    expected = [reference.end2end(frame) for frame in frames]
    del reference
    ground_truth = [[(d[0], d[2]) for d in detections] for detections in expected]
    assert all(len(boxes) > 0 for boxes in ground_truth)

    models = {name: partial(Detector, path, warmup=True) for name, path in engines.items()}
    traits = characterize(models, frames, ground_truth, iterations=50, warmup_iterations=5)
    assert list(traits) == list(models)
    # ground truth is n640's own output
    assert traits["n640"].frame_f1() == [1.0, 1.0]
    for name, size in (("n640", 640), ("n320", 320)):
        trait = traits[name]
        assert (trait.input_size, trait.dla) == ((size, size), False)
        assert len(trait.latency_ms) == 50
        assert trait.latency() > 0.0
        assert (len(trait.energy_mj) > 0) is FLAGS.IS_JETSON
        assert trait.pair_latency_ms == {}
        assert trait.ground_truths == [len(boxes) for boxes in ground_truth]

    path = tmp_path / "traits.json"
    save_characterization(traits, path)
    loaded = load_characterization(path)
    assert loaded == traits

    drift = Drift(models, loaded, PRIORITY, baseline="n640", seed=0)
    for frame, reference_detections in zip(frames, expected):
        # a new frame size starts on the baseline, which sees the full frame
        _assert_same_detections(drift.end2end(frame), reference_detections)
        assert len(drift.end2end(frame)) > 0
    drift.close()


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
def test_varying_input_sizes_match_fresh_detectors(
    engines, canvas, preprocessor, cuda_graph
) -> None:
    """One detector fed crops of changing size matches a fresh detector per crop."""
    crops = [
        np.ascontiguousarray(canvas[0:500, 0:700]),
        np.ascontiguousarray(canvas[100:1080, 0:1920]),
        np.ascontiguousarray(canvas[0:300, 50:350]),
    ]
    factory = partial(
        Detector, engines["n640"], preprocessor=preprocessor, cuda_graph=cuda_graph, warmup=True
    )
    expected = []
    for crop in crops:
        fresh = factory()
        expected.append(fresh.end2end(crop))
        del fresh
    detector = factory()
    for index in range(2 * len(crops)):
        _assert_same_detections(detector.end2end(crops[index % 3]), expected[index % 3])


@pytest.mark.dla
@pytest.mark.jetson
def test_dla_pairs_and_concurrent_run(engines, dla_engine, canvas) -> None:
    """Pairs are measured on the DLA, and concurrent frames leave every GPU CUDA graph working."""
    models = {
        "n640": partial(Detector, engines["n640"], warmup=True),
        "n320": partial(Detector, engines["n320"], warmup=True),
        "xs_dla": partial(YOLOX, dla_engine, dla_core=0, cuda_graph=False, warmup=True),
    }
    reference = Detector(engines["n640"], warmup=True)
    ground_truth = [[(d[0], d[2]) for d in reference.end2end(canvas)]]
    del reference
    traits = characterize(models, [canvas], ground_truth, iterations=30, warmup_iterations=5)
    assert traits["xs_dla"].dla
    for gpu in ("n640", "n320"):
        assert len(traits[gpu].pair_latency_ms["xs_dla"]) == 30
        assert traits[gpu].pair_latency("xs_dla") > 0.0

    # make the baseline look slow so split schedules always qualify; this checks the
    # runtime mechanics on hardware, not which schedule is fastest
    traits["n640"] = dataclasses.replace(traits["n640"], latency_ms=[1000.0])
    config = DriftConfig(lp_accuracy_goal=0.0, hp_trusted_models=("n320",), lp_exclude_hp_model=True)
    drift = Drift(models, traits, PRIORITY, baseline="n640", config=config, seed=0)
    parallel = 0
    for _ in range(300):
        drift.end2end(canvas)
        assert drift.last is not None
        parallel += int(drift.last.schedule.parallel)
    assert parallel > 0
    for name in ("n640", "n320"):
        assert drift._detectors[name].engine._cuda_graph_enabled
    drift.close()


@pytest.mark.cpu
@pytest.mark.parametrize(
    "container",
    [
        pytest.param(list, id="list"),
        pytest.param(iter, id="generator"),
    ],
)
def test_characterize_traits_from_any_iterable(fake, container) -> None:
    """Lists and one-shot iterators of images and ground truth give the same, exact traits."""
    models, _ = fake
    traits = characterize(
        models,
        container([_image(v) for v in VALUES]),
        container([FAKE_TRUTH for _ in VALUES]),
        iterations=ITERATIONS,
    )
    latency = [1.0] * ITERATIONS
    expected = {
        "gpu": ModelTraits(
            input_size=(64, 64),
            dla=False,
            latency_ms=latency,
            confidences=[[v / 10, 0.25] for v in VALUES],
            true_positives=[v % 2 for v in VALUES],
            ground_truths=[1] * len(VALUES),
            pair_latency_ms={"dla": latency},
            pair_energy_mj={"dla": []},
        ),
        "dla": ModelTraits(
            input_size=(64, 64),
            dla=True,
            latency_ms=latency,
            confidences=[[v / 20, 0.25] for v in VALUES],
            true_positives=[v % 2 for v in VALUES],
            ground_truths=[1] * len(VALUES),
        ),
    }
    assert traits == expected


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("kwargs", "latency_values"),
    [
        pytest.param({}, [1, 2, 3, 4, 1, 2], id="default-all-labeled"),
        pytest.param({"latency_frames": 2}, [1, 2, 1, 2, 1, 2], id="default-first-two"),
        pytest.param(
            {"latency_images": [_image(50), _image(60)], "latency_frames": 1},
            [50, 60, 50, 60, 50, 60],
            id="explicit-latency-images",
        ),
    ],
)
def test_characterize_streams_paths(fake, tmp_path, monkeypatch, kwargs, latency_values) -> None:
    """Each image is read once, run on every model before the next read; latency reuses frames."""
    models, log = fake
    paths = []
    for value in VALUES:
        paths.append(tmp_path / f"{value}.png")
        cv2.imwrite(str(paths[-1]), _image(value))
    imread = cv2.imread

    def _logged_imread(path):
        log.append(("read", int(Path(path).stem)))
        return imread(path)

    monkeypatch.setattr(cv2, "imread", _logged_imread)
    traits = characterize(
        models, (p for p in paths), (FAKE_TRUTH for _ in paths), iterations=ITERATIONS, **kwargs
    )
    assert traits["gpu"].confidences == [[v / 10, 0.25] for v in VALUES]

    # warmup on the first image, then image outer, model inner
    expected = [("read", 1)] + [("run", "gpu", 1)] * 3 + [("run", "dla", 1)] * 3
    for value in VALUES:
        if value != VALUES[0]:
            expected.append(("read", value))
        expected += [("run", "gpu", value), ("run", "dla", value)]
    assert log[: len(expected)] == expected

    # solo loops per model, then the concurrent pair loop, with no further reads
    timing = log[len(expected) :]
    solo = [("run", name, v) for name in ("gpu", "dla") for v in latency_values]
    assert timing[: len(solo)] == solo
    pair = Counter(("run", name, v) for name in ("gpu", "dla") for v in latency_values)
    assert Counter(timing[len(solo) :]) == pair


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("images", "ground_truth", "kwargs", "match"),
    [
        pytest.param(
            [_image(1)] * 3, [FAKE_TRUTH] * 2, {}, "got 3 images but 2", id="list-mismatch"
        ),
        pytest.param(
            iter([_image(1)] * 3), iter([FAKE_TRUTH] * 2), {}, "fewer", id="more-images-iter"
        ),
        pytest.param(
            iter([_image(1)] * 2), iter([FAKE_TRUTH] * 3), {}, "more", id="more-truth-iter"
        ),
        pytest.param([], [], {}, "at least one labeled image", id="empty-list"),
        pytest.param(iter([]), iter([]), {}, "at least one labeled image", id="empty-iter"),
        pytest.param(
            [_image(1)], [FAKE_TRUTH], {"latency_images": iter([])}, "empty", id="no-latency"
        ),
        pytest.param(
            [_image(1)], [FAKE_TRUTH], {"latency_frames": 0}, "at least 1", id="latency-frames"
        ),
    ],
)
def test_characterize_invalid_inputs(fake, images, ground_truth, kwargs, match) -> None:
    """Mismatched lengths, even of iterators, empty inputs and bad latency settings raise."""
    models, _ = fake
    with pytest.raises(ValueError, match=match):
        characterize(models, images, ground_truth, iterations=ITERATIONS, **kwargs)
