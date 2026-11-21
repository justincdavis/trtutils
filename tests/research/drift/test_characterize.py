# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/drift/_characterize.py and _drift.py -- on real engines."""

from __future__ import annotations

import dataclasses
from functools import partial

import numpy as np
import pytest

from tests.conftest import DATA_DIR, ENGINES_DIR
from trtutils import FLAGS
from trtutils.image import Detector
from trtutils.models import YOLOX
from trtutils.research.drift import (
    Drift,
    DriftConfig,
    characterize,
    load_characterization,
    save_characterization,
)

YOLOV10N_640 = DATA_DIR / "yolov10" / "yolov10n_640.onnx"
YOLOV10N_320 = DATA_DIR / "yolov10" / "yolov10n_320.onnx"
YOLOXS_320 = DATA_DIR / "yolox" / "yoloxs_320.onnx"
PRIORITY = {0, 17}  # COCO person and horse


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
