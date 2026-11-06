# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CUDA postprocessors match the CPU path: models, detection, classification, and NMS."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tests.conftest import DATA_DIR
from trtutils.core import Buffer
from trtutils.image import Classifier, Detector, OBBDetector
from trtutils.image.postprocessors._common import (
    clip_boxes,
    cxcywh_to_xyxy,
    filter_class_scores,
    nms_indices,
    unletterbox_boxes,
)
from trtutils.image.postprocessors._cpu_classifier import postprocess_classifications
from trtutils.image.postprocessors._cpu_detection import (
    postprocess_efficient_nms,
    postprocess_yolov10,
)
from trtutils.image.postprocessors._cuda import letterbox_table, split_detections
from trtutils.image.postprocessors._cuda_classifier import CUDAClsPostprocessor
from trtutils.image.postprocessors._cuda_detection import CUDADetPostprocessor
from trtutils.image.postprocessors._cuda_nms import YoloNms

_RATIOS = [(0.5, 0.5), (0.25, 0.5)]
_PADDING = [(4.0, 8.0), (2.0, 6.0)]
_INPUT = (64, 48)


def _engine(fixture: str, *args: str):
    """Engine getter: the fixture's path, or its factory called with args."""

    def get(request):
        value = request.getfixturevalue(fixture)
        return value(*args) if args else value

    return get


# every model with a CUDA postprocessor: (model class, engine getter, test image)
_CUDA_MODELS = [
    pytest.param(Detector, _engine("yolov10_engine"), "horse", id="detector"),
    pytest.param(Classifier, _engine("task_engine", "cls"), "horse", id="classifier"),
    pytest.param(OBBDetector, _engine("task_engine", "obb"), "aerial", id="obb"),
]


def _assert_nested_close(got, exp) -> None:
    if isinstance(exp, np.ndarray):
        assert got.shape == exp.shape
        np.testing.assert_allclose(got, exp, rtol=1e-5, atol=1e-3)
        return
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        _assert_nested_close(g, e)


@pytest.mark.parametrize(("model_cls", "engine", "image_name"), _CUDA_MODELS)
def test_cuda_postprocessor_matches_cpu(request, images, model_cls, engine, image_name) -> None:
    """run() and end2end() with postprocessor="cuda" match the CPU postprocessor."""
    path = engine(request)
    if image_name in images:
        image = images[image_name].array
    else:
        image = cv2.imread(str(DATA_DIR / f"{image_name}.jpg"))
    cpu = model_cls(path, postprocessor="cpu", warmup=False)
    cuda = model_cls(path, postprocessor="cuda", warmup=False)
    # twice: the second run replays the engine's CUDA graph
    for _ in range(2):
        _assert_nested_close(cuda.run(image), cpu.run(image))
        assert len(cuda.end2end(image)) == len(cpu.end2end(image))


def test_invalid_postprocessor_raises(yolov10_engine) -> None:
    """An unknown postprocessor is rejected at construction."""
    with pytest.raises(ValueError, match="Invalid postprocessor"):
        Detector(yolov10_engine, postprocessor="gpu", warmup=False)


def _assert_detections_match(gpu, cpu) -> None:
    assert len(gpu) == len(cpu)
    for got, exp in zip(gpu, cpu):
        # integer ids, as the CPU paths that decode heads (EfficientNMS passes its tensor through)
        assert got[2].tolist() == exp[2].tolist()
        assert got[2].dtype.kind == "i"
        np.testing.assert_allclose(got[0], exp[0], atol=1e-4)
        np.testing.assert_allclose(got[1], exp[1], atol=1e-5)


def test_cuda_yolov10_matches_cpu(cuda_stream, to_device) -> None:
    """Unletterbox and the confidence filter keep the same rows as the CPU path."""
    rng = np.random.default_rng(0)
    rows = rng.standard_normal((2, 5, 6)).astype(np.float32)
    rows[:, :, 4] = np.array([0.9, 0.05, 0.4, 0.0, 0.2], np.float32)
    rows[:, :, 5] = np.array([1, 2, 3, 0, 4], np.float32)
    cpu = postprocess_yolov10([rows.copy()], _RATIOS, _PADDING, 0.3, _INPUT)
    gpu = CUDADetPostprocessor(cuda_stream)([to_device(rows)], _RATIOS, _PADDING, 0.3, _INPUT)
    _assert_detections_match(gpu, cpu)


@pytest.mark.parametrize("ids", [np.float32, np.int32, np.int64])
def test_cuda_efficient_nms_matches_cpu(cuda_stream, to_device, ids) -> None:
    """EfficientNMS keeps the first num_dets rows and the same unletterboxed boxes."""
    num = np.array([[2], [1]], ids if ids != np.float32 else np.int32)
    boxes = np.zeros((2, 4, 4), np.float32)
    boxes[0, 0] = (10, 12, 20, 22)
    boxes[0, 1] = (14, 16, 30, 40)
    boxes[1, 0] = (8, 10, 18, 20)
    scores = np.array([[0.8, 0.2, 0.9, 0.1], [0.05, 0.7, 0.4, 0.3]], np.float32)
    classes = np.array([[3, 1, 0, 2], [4, 5, 6, 7]], ids)
    outputs = [num, boxes, scores, classes]
    cpu = postprocess_efficient_nms(
        [part.copy() for part in outputs], _RATIOS, _PADDING, None, _INPUT
    )
    gpu = CUDADetPostprocessor(cuda_stream)(
        [to_device(part) for part in outputs], _RATIOS, _PADDING, None, _INPUT
    )
    _assert_detections_match(gpu, cpu)


def _reference_nms(head, nms_iou_thres, max_det, n_extra) -> list[list[np.ndarray]]:
    """The CPU decode of a (B, 4 + nc + n_extra, N) head with keypoint extras, as pose does."""
    results = []
    for i, ((rx, ry), (px, py)) in enumerate(zip(_RATIOS, _PADDING)):
        pred, scores, ids = filter_class_scores(head[i], slice(4, -n_extra), None)
        boxes = cxcywh_to_xyxy(pred)
        keep = nms_indices(boxes, scores, ids, None, nms_iou_thres, max_det)
        boxes = unletterbox_boxes(boxes[keep], _RATIOS[i], _PADDING[i], _INPUT)
        kpts = pred[keep, -n_extra:].reshape(len(keep), -1, 3)
        kpts[:, :, 0] = (kpts[:, :, 0] - px) / rx
        kpts[:, :, 1] = (kpts[:, :, 1] - py) / ry
        xy = clip_boxes(kpts[:, :, [0, 1, 0, 1]], _RATIOS[i], _PADDING[i], _INPUT)
        kpts[:, :, :2] = xy[:, :, :2]
        results.append([boxes, scores[keep], ids[keep], kpts])
    return results


@pytest.mark.parametrize("num", [600, 4200], ids=["mask", "over_mask_rows"])
def test_cuda_nms_gather_matches_cpu(cuda_stream, to_device, num) -> None:
    """NMS and gather keep the CPU rows, also past the NMS bit-matrix rows."""
    rng = np.random.default_rng(2)
    nc, n_extra, max_det = 3, 6, 50
    head = np.empty((2, 4 + nc + n_extra, num), np.float32)
    # clustered boxes so NMS suppresses most of them
    head[:, :2] = rng.uniform(0, 48, (2, 2, num))
    head[:, 2:4] = rng.uniform(4, 16, (2, 2, num))
    head[:, 4 : 4 + nc] = rng.uniform(0, 1, (2, nc, num))
    head[:, 4 + nc :] = rng.uniform(-8, 72, (2, n_extra, num))
    cpu = _reference_nms(head, 0.5, max_det, n_extra)

    nms = YoloNms(cuda_stream)
    dets = Buffer.host_mapped((2, max_det, 6), np.float32)
    points = Buffer.host_mapped((2, max_det, n_extra), np.float32)
    counts = Buffer.host_mapped((2,), np.int32)
    letterbox = to_device(letterbox_table(_RATIOS, _PADDING, _INPUT))
    device_head = to_device(head)
    nms(device_head, 4, nc, None, 0.5, max_det)
    nms.gather(
        device_head,
        letterbox.device_ptr,
        dets.device_ptr,
        counts.device_ptr,
        extra_start=4 + nc,
        extras=(points.device_ptr, n_extra),
        points=True,
    )
    nms._sync()
    gpu = split_detections(dets.array, counts.array, points.array.reshape(2, max_det, -1, 3))
    _assert_detections_match(gpu, cpu)
    for got, exp in zip(gpu, cpu):
        np.testing.assert_allclose(got[3], exp[3], atol=1e-4)


def test_cuda_softmax_matches_cpu(cuda_stream, to_device) -> None:
    """The class vector matches a host softmax, and softmax=False copies the logits."""
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((3, 17)).astype(np.float32)
    cuda = CUDAClsPostprocessor(cuda_stream)
    for softmax in (True, False):
        cpu = postprocess_classifications([logits.copy()], softmax=softmax)
        gpu = cuda([to_device(logits)], softmax=softmax)
        for got, exp in zip(gpu, cpu):
            np.testing.assert_allclose(got[0], exp[0], rtol=1e-5, atol=1e-5)
