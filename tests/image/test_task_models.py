# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/_segmenter.py, _pose_estimator.py, _obb_detector.py -- task models on YOLOv11n engines."""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import pytest

from tests.conftest import DATA_DIR
from trtutils import Buffer, MemoryLocation
from trtutils.image import Classifier, OBBDetector, PoseEstimator, Segmenter
from trtutils.image._schema import (
    OBBDetectorOutputSchema,
    PoseEstimatorOutputSchema,
    SegmenterOutputSchema,
)

PREPROCS = [
    pytest.param("cpu", id="cpu"),
    pytest.param("cuda", id="cuda"),
    pytest.param("trt", id="trt"),
    pytest.param("hybrid", id="hybrid"),
]
TASKS = [
    pytest.param(Segmenter, "seg", "people", id="seg"),
    pytest.param(PoseEstimator, "pose", "people", id="pose"),
    pytest.param(OBBDetector, "obb", "aerial", id="obb"),
]


def _load(images, name: str) -> np.ndarray:
    if name == "aerial":
        return cv2.imread(str(DATA_DIR / "aerial.jpg"))
    return images[name].array


def _fields(results: list) -> list[tuple]:
    """Drop the array fields (masks, keypoints) so results compare with ==."""
    return [tuple(f for f in r if not isinstance(f, np.ndarray)) for r in results]


def _assert_same_results(ours: list, ref: list) -> None:
    """Same count, and each result has a ref result within a pixel with a close score."""
    assert len(ours) == len(ref)
    ref_boxes = np.array([np.asarray(r[0], float) for r in ref]).reshape(len(ref), -1)
    ref_scores = np.array([r[1] for r in ref])
    for result in ours:
        dist = np.abs(ref_boxes[:, :4] - np.asarray(result[0], float)[:4]).max(axis=1)
        j = int(np.argmin(dist))
        assert dist[j] <= 2.0
        assert result[1] == pytest.approx(ref_scores[j], abs=0.02)


def _check_seg(results: list, image: np.ndarray) -> None:
    assert len(results) >= 3
    assert 0 in {cls_id for _bbox, _score, cls_id, _mask in results}
    for _bbox, _score, _cls_id, mask in results:
        assert mask.shape == image.shape[:2]
        assert mask.any()


def _check_pose(results: list, _image: np.ndarray) -> None:
    # people.jpeg holds 3-5 people
    assert 3 <= len(results) <= 5
    for _bbox, _score, keypoints in results:
        assert keypoints.shape == (17, 3)


def _check_obb(results: list, image: np.ndarray) -> None:
    height, width = image.shape[:2]
    assert len(results) > 0
    for (cx, cy, w, h, _angle), score, _cls_id in results:
        assert 0 <= cx <= width
        assert 0 <= cy <= height
        assert w > 0
        assert h > 0
        assert 0.0 <= score <= 1.0


@pytest.mark.parametrize(
    ("model_cls", "task", "image_name", "get_name", "check"),
    [
        pytest.param(Segmenter, "seg", "people", "get_segmentations", _check_seg, id="seg"),
        pytest.param(PoseEstimator, "pose", "people", "get_poses", _check_pose, id="pose"),
        pytest.param(OBBDetector, "obb", "aerial", "get_obbs", _check_obb, id="obb"),
    ],
)
@pytest.mark.parametrize("preprocessor", PREPROCS)
def test_task_model_end2end(
    task_engine,
    images,
    model_cls,
    task: str,
    image_name: str,
    get_name: str,
    check: Callable[[list, np.ndarray], None],
    preprocessor: str,
) -> None:
    """end2end() matches run() + get_*() and finds the expected objects."""
    image = _load(images, image_name)
    model = model_cls(task_engine(task), preprocessor=preprocessor, warmup=False)

    via_run = getattr(model, get_name)(model.run(image))
    via_e2e = model.end2end(image)
    assert _fields(via_run) == _fields(via_e2e)
    check(via_e2e, image)


@pytest.mark.parametrize(
    ("model_cls", "task", "schema"),
    [
        pytest.param(Segmenter, "seg", SegmenterOutputSchema.YOLO, id="seg"),
        pytest.param(PoseEstimator, "pose", PoseEstimatorOutputSchema.YOLO, id="pose"),
        pytest.param(OBBDetector, "obb", OBBDetectorOutputSchema.YOLO, id="obb"),
    ],
)
def test_task_model_output_schema(task_engine, model_cls, task: str, schema) -> None:
    """Task models resolve their output schema from the engine, or from an override."""
    path = task_engine(task)
    assert model_cls(path, warmup=False).output_schema is schema
    assert model_cls(path, output_schema=schema, warmup=False).output_schema is schema
    assert model_cls(path, output_schema="YOLO", warmup=False).output_schema is schema


@pytest.mark.parametrize(
    ("model_cls", "engine_task"),
    [
        pytest.param(Segmenter, "pose", id="seg-on-pose"),
        pytest.param(PoseEstimator, "seg", id="pose-on-seg"),
        pytest.param(OBBDetector, "seg", id="obb-on-seg"),
    ],
)
def test_task_model_rejects_wrong_engine(task_engine, model_cls, engine_task: str) -> None:
    """A task model on another task's engine raises instead of decoding garbage."""
    with pytest.raises(ValueError, match="match no"):
        model_cls(task_engine(engine_task), warmup=False)


@pytest.mark.parametrize(
    "size",
    [
        pytest.param((1280, 719), id="1280x719"),
        pytest.param((1280, 721), id="1280x721"),
        pytest.param((719, 1280), id="719x1280"),
    ],
)
def test_segmenter_masks_fit_odd_sizes(task_engine, images, size) -> None:
    """Letterbox rounding never leaves end2end() masks a pixel off the image size."""
    model = Segmenter(task_engine("seg"), warmup=False)
    image = cv2.resize(images["horse"].array, size)

    for segs in (model.end2end(image), model.end2end([image])[0]):
        assert segs
        assert all(mask.shape == image.shape[:2] for *_rest, mask in segs)


@pytest.mark.parametrize(("model_cls", "task", "image_name"), TASKS)
@pytest.mark.parametrize(
    ("init_kwargs", "call_kwargs"),
    [
        pytest.param({"conf_thres": 0.5}, {"conf_thres": 0.0}, id="conf-thres"),
        pytest.param(
            {"conf_thres": 0.25, "nms_iou_thres": 1.0}, {"nms_iou_thres": 0.0}, id="nms-iou-thres"
        ),
    ],
)
def test_threshold_zero_overrides_default(
    task_engine, images, model_cls, task, image_name, init_kwargs, call_kwargs
) -> None:
    """An explicit per-call threshold of 0.0 is honored instead of the constructor value."""
    image = _load(images, image_name)
    model = model_cls(task_engine(task), warmup=False, **init_kwargs)

    default = model.end2end(image)
    override = model.end2end(image, **call_kwargs)
    if "conf_thres" in call_kwargs:
        assert all(r[1] >= 0.5 for r in default)
        assert any(r[1] < 0.5 for r in override)
    else:
        assert 0 < len(override) < len(default)


def test_segmenter_mask_thres(task_engine, images) -> None:
    """The constructor mask_thres applies to run(), and an explicit per-call 0.0 overrides it."""
    image = images["horse"].array
    strict = Segmenter(task_engine("seg"), mask_thres=0.99, warmup=False)
    filled = Segmenter(task_engine("seg"), mask_thres=0.0, warmup=False)

    # end2end masks live in a reused canvas, so keep a copy across calls
    default = [(*seg[:3], seg[3].copy()) for seg in strict.end2end(image)]
    override = strict.end2end(image, mask_thres=0.0)
    via_run = filled.get_segmentations(filled.run(image))
    assert default
    assert _fields(default) == _fields(override) == _fields(via_run)
    for small, full, run_full in zip(default, override, via_run):
        assert small[3].sum() < full[3].sum()
        np.testing.assert_array_equal(full[3], run_full[3])


def test_obb_detector_rejects_linear_resize(task_engine) -> None:
    """Non-uniform resizing cannot map rotated boxes back, so it raises."""
    image = cv2.imread(str(DATA_DIR / "aerial.jpg"))
    model = OBBDetector(task_engine("obb"), resize_method="linear", warmup=False)
    with pytest.raises(ValueError, match="uniform scaling"):
        model.end2end(image)


@pytest.mark.parametrize(("model_cls", "task", "image_name"), TASKS)
def test_batch2_engine_matches_batch1(task_engine, images, model_cls, task, image_name) -> None:
    """A batch-2 engine gives each image the results the batch-1 engine gives it alone."""
    image = _load(images, image_name)
    flipped = np.ascontiguousarray(image[:, ::-1])
    single = model_cls(task_engine(task), conf_thres=0.5, warmup=False)
    batch = model_cls(task_engine(task, batch_size=2), conf_thres=0.5, warmup=False)

    results = batch.end2end([image, flipped])
    assert results[0]
    _assert_same_results(results[0], single.end2end(image))
    _assert_same_results(results[1], single.end2end(flipped))


@pytest.mark.parametrize(("model_cls", "task", "image_name"), TASKS)
def test_device_buffer_matches_numpy(task_engine, images, model_cls, task, image_name) -> None:
    """A device Buffer input gives the same results as the numpy image."""
    image = _load(images, image_name)
    model = model_cls(task_engine(task), warmup=False)

    expected = model.end2end(image)
    assert expected
    _assert_same_results(model.end2end(Buffer.from_array(image, MemoryLocation.DEVICE)), expected)


@pytest.mark.parametrize(("model_cls", "task", "image_name"), TASKS)
@pytest.mark.parametrize(
    "max_det", [pytest.param(5, id="max-det-5"), pytest.param(300, id="max-det-300")]
)
def test_max_det_caps_score_sorted_results(
    task_engine, images, model_cls, task, image_name, max_det
) -> None:
    """conf_thres=0.0 keeps at most max_det results, highest scores first."""
    image = _load(images, image_name)
    model = model_cls(task_engine(task), max_det=max_det, warmup=False)

    results = model.end2end(image, conf_thres=0.0)
    scores = [r[1] for r in results]
    assert len(results) == max_det
    assert scores == sorted(scores, reverse=True)


def test_classifier_softmax_false_top_k(task_engine, images) -> None:
    """The cls head already softmaxes; a center crop of the horse gives ultralytics' top-1."""
    image = images["horse"].array
    height, width = image.shape[:2]
    side = min(height, width)
    y0, x0 = (height - side) // 2, (width - side) // 2
    crop = np.ascontiguousarray(image[y0 : y0 + side, x0 : x0 + side])
    model = Classifier(task_engine("cls"), softmax=False, warmup=False)

    top = model.end2end(crop, top_k=1000)
    probs = [conf for _cls_id, conf in top]
    assert sum(probs) == pytest.approx(1.0, abs=1e-2)
    assert probs == sorted(probs, reverse=True)
    assert top[0][0] == 339  # sorrel, ultralytics top-1


def test_segmenter_cpu_no_copy_postprocess_then_cuda(task_engine, images) -> None:
    """A CPU no_copy postprocess of host outputs leaves the CUDA path's canvases working."""
    image = images["horse"].array
    ref = Segmenter(task_engine("seg"), postprocessor="cpu", warmup=False).end2end(image)
    seg = Segmenter(task_engine("seg"), postprocessor="cuda", warmup=False)
    tensor, ratios, padding = seg.preprocess(image)
    raw = seg.run(tensor, ratios, padding, preprocessed=True, postprocess=False)
    host = seg.get_segmentations(seg.postprocess(raw, ratios, padding, no_copy=True))[0]
    assert _fields(host) == _fields(ref)
    got = seg.end2end(image)
    assert _fields(got) == _fields(ref)
    for g, w in zip(got, ref):
        np.testing.assert_array_equal(g[3], w[3])


@pytest.mark.parametrize("image_name", ["people", "horse"])
def test_segmenter_cuda_postprocessor_matches_cpu(task_engine, images, image_name: str) -> None:
    """postprocessor="cuda" gives the CPU results exactly, through run() and end2end()."""
    image = images[image_name].array
    cpu = Segmenter(task_engine("seg"), warmup=False)
    cuda = Segmenter(task_engine("seg"), postprocessor="cuda", warmup=False)

    ref = cpu.end2end(image)
    for segs in (cuda.end2end(image), cuda.get_segmentations(cuda.run(image)), cuda.end2end(image)):
        assert _fields(segs) == _fields(ref)
        for got, want in zip(segs, ref):
            np.testing.assert_array_equal(got[3], want[3])


def test_obb_detector_cuda_postprocessor_matches_cpu(task_engine, images) -> None:
    """postprocessor="cuda" gives the CPU detections exactly, through run() and end2end()."""
    image = _load(images, "aerial")
    cpu = OBBDetector(task_engine("obb"), warmup=False)
    cuda = OBBDetector(task_engine("obb"), postprocessor="cuda", warmup=False)

    ref = cpu.end2end(image)
    assert ref
    for obbs in (cuda.end2end(image), cuda.get_obbs(cuda.run(image)), cuda.end2end(image)):
        assert obbs == ref
    # enclosing boxes use cos / sin, which numpy's float32 does not round like the GPU
    want = cpu.run(image)
    got = cuda.run(image)
    np.testing.assert_allclose(got[0], want[0], rtol=1e-5, atol=1e-3)
    for g, w in zip(got[1:], want[1:]):
        np.testing.assert_array_equal(g, w)


@pytest.mark.parametrize(("model_cls", "task"), [pytest.param(Classifier, "cls", id="cls")])
def test_cuda_postprocessor_unsupported_raises(task_engine, model_cls, task: str) -> None:
    """A task model without a CUDA postprocessor rejects postprocessor="cuda"."""
    with pytest.raises(ValueError, match="postprocessor"):
        model_cls(task_engine(task), postprocessor="cuda", warmup=False)


@pytest.mark.parametrize("image_name", ["people", "horse"])
def test_pose_cuda_postprocessor_matches_cpu(task_engine, images, image_name: str) -> None:
    """postprocessor="cuda" gives the CPU poses exactly, through run() and end2end()."""
    image = images[image_name].array
    cpu = PoseEstimator(task_engine("pose"), warmup=False)
    cuda = PoseEstimator(task_engine("pose"), postprocessor="cuda", warmup=False)

    ref = cpu.end2end(image)
    for poses in (cuda.end2end(image), cuda.get_poses(cuda.run(image)), cuda.end2end(image)):
        assert len(poses) == len(ref)
        for got, want in zip(poses, ref):
            np.testing.assert_array_equal(got[0], want[0])
            assert got[1] == want[1]
            np.testing.assert_array_equal(got[2], want[2])
