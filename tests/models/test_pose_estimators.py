# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine-level tests for the pose estimator wrappers in src/trtutils/models/pose_estimators/."""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort
import pytest

from tests.conftest import DATA_DIR, PREPROCESSORS
from trtutils.image._schema import PoseEstimatorOutputSchema
from trtutils.image.postprocessors import postprocess_rtmo_pose
from trtutils.models import RTMO, YuNet

RTMO_MODELS = [
    pytest.param(RTMO, DATA_DIR / "rtmo" / "rtmo_s_640.onnx", id="rtmo-s"),
    pytest.param(RTMO, DATA_DIR / "rtmo" / "rtmo_m_640.onnx", id="rtmo-m"),
    pytest.param(RTMO, DATA_DIR / "rtmo" / "rtmo_l_640.onnx", id="rtmo-l"),
]
YUNET_ONNX = DATA_DIR / "yunet" / "yunet_640.onnx"
# the faces of people.jpeg score 0.7-0.9 once upscaled to 640, under the 0.9 default
YUNET_CONF = 0.6


def _iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(iw, 0) * max(ih, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@pytest.mark.parametrize(("model_cls", "onnx_path"), RTMO_MODELS)
@pytest.mark.parametrize(("preprocessor", "cpu_resize"), PREPROCESSORS)
def test_rtmo_end2end(
    build_model_engine, images, model_cls, onnx_path, preprocessor, cpu_resize
) -> None:
    """run(), get_poses(), and end2end() agree and find the people with 17 keypoints."""
    engine = build_model_engine(model_cls, onnx_path)
    model = model_cls(engine, preprocessor=preprocessor, cpu_resize=cpu_resize, warmup=False)
    assert model.output_schema is PoseEstimatorOutputSchema.RTMO
    image = images["people"].array
    height, width = image.shape[:2]

    via_run = model.get_poses(model.run(image))
    via_e2e = model.end2end(image)
    assert len(via_run) == len(via_e2e)
    for (box_a, score_a, kpts_a), (box_b, score_b, kpts_b) in zip(via_run, via_e2e):
        assert _iou(box_a, box_b) > 0.95
        assert score_a == pytest.approx(score_b, abs=0.02)
        np.testing.assert_allclose(kpts_a[:, :2], kpts_b[:, :2], atol=3.0)

    confident = [pose for pose in via_e2e if pose[1] >= 0.5]
    assert len(confident) >= 3
    for (x1, y1, x2, y2), _, kpts in confident:
        assert 0 <= x1 < x2 <= width
        assert 0 <= y1 < y2 <= height
        assert kpts.shape == (17, 3)


@pytest.mark.parametrize(("model_cls", "onnx_path"), RTMO_MODELS)
def test_rtmo_no_people_is_empty(build_model_engine, images, model_cls, onnx_path) -> None:
    """Images without people give no poses: the engine never sees an empty NMS (TRT 8.6)."""
    engine = build_model_engine(model_cls, onnx_path)
    # 0.3 leaves fp16 margin over the best non-person rows (up to ~0.12)
    model = model_cls(engine, conf_thres=0.3, warmup=False)
    blank = np.full((480, 640, 3), 114, dtype=np.uint8)
    for image in (images["horse"].array, blank):
        dets, keypoints = model.run(image, postprocess=False)
        assert np.asarray(dets).shape == (1, 50, 5)
        assert np.asarray(keypoints).shape == (1, 50, 17, 3)
        assert model.end2end(image) == []


@pytest.mark.correctness
@pytest.mark.parametrize(("model_cls", "onnx_path"), RTMO_MODELS)
def test_rtmo_matches_onnxruntime(build_model_engine, images, model_cls, onnx_path) -> None:
    """The default (fp16) engine finds the people onnxruntime finds, keypoints within a few px."""
    engine = build_model_engine(model_cls, onnx_path)
    model = model_cls(engine, preprocessor="cpu", warmup=False)
    tensor, ratios, padding = model.preprocess(images["people"].array)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    expected = postprocess_rtmo_pose(
        session.run(None, {"input": np.asarray(tensor)}),
        ratios,
        padding,
        0.5,
        0.65,
        input_size=model.input_shape,
    )[0]
    boxes, scores, _, kpts = model.run(tensor, ratios[0], padding[0], 0.5, preprocessed=True)

    assert len(scores) == len(expected[1])
    for box, score, kpt in zip(expected[0], expected[1], expected[3]):
        j = max(range(len(boxes)), key=lambda k: _iou(box, boxes[k]))
        assert _iou(box, boxes[j]) >= 0.9
        # fp16 moves rtmo-m scores by up to ~0.06 on this image
        assert scores[j] == pytest.approx(score, abs=0.08)
        visible = (kpt[:, 2] > 0.5) & (kpts[j, :, 2] > 0.5)
        assert np.linalg.norm(kpt[visible, :2] - kpts[j, visible, :2], axis=1).max() < 3.0


def test_rtmo_rejects_cuda_postprocessor(build_model_engine) -> None:
    """RTMO engines decode on the CPU: asking for the CUDA postprocessor raises."""
    engine = build_model_engine(RTMO, DATA_DIR / "rtmo" / "rtmo_s_640.onnx")
    model = RTMO(engine, warmup=False)
    assert model._postprocessor == "cpu"
    with pytest.raises(ValueError, match="supports YOLO, not RTMO"):
        RTMO(engine, postprocessor="cuda", warmup=False)


@pytest.mark.parametrize(("preprocessor", "cpu_resize"), PREPROCESSORS)
def test_yunet_end2end(build_model_engine, images, preprocessor, cpu_resize) -> None:
    """run(), get_poses(), and end2end() agree and find the faces with 5 landmarks."""
    engine = build_model_engine(YuNet, YUNET_ONNX)
    model = YuNet(engine, preprocessor=preprocessor, cpu_resize=cpu_resize, warmup=False)
    assert model.output_schema is PoseEstimatorOutputSchema.YUNET
    image = images["people"].array
    height, width = image.shape[:2]

    via_run = model.get_poses(model.run(image, conf_thres=YUNET_CONF), YUNET_CONF)
    via_e2e = model.end2end(image, YUNET_CONF)
    assert len(via_e2e) == 3
    assert len(via_run) == len(via_e2e)
    for (box_a, score_a, kpts_a), (box_b, score_b, kpts_b) in zip(via_run, via_e2e):
        assert _iou(box_a, box_b) > 0.95
        assert score_a == pytest.approx(score_b, abs=0.02)
        np.testing.assert_allclose(kpts_a[:, :2], kpts_b[:, :2], atol=1.0)

    for (x1, y1, x2, y2), score, kpts in via_e2e:
        assert 0 <= x1 < x2 <= width
        assert 0 <= y1 < y2 <= height
        assert kpts.shape == (5, 3)
        # the landmarks carry the face score and sit inside the face
        np.testing.assert_allclose(kpts[:, 2], score)
        assert ((kpts[:, :2] >= [x1 - 1, y1 - 1]) & (kpts[:, :2] <= [x2 + 1, y2 + 1])).all()
        # the right eye is left of the left eye in the image, both above the mouth
        assert kpts[0, 0] < kpts[1, 0]
        assert max(kpts[0, 1], kpts[1, 1]) < min(kpts[3, 1], kpts[4, 1])


@pytest.mark.correctness
def test_yunet_matches_opencv(build_model_engine, images) -> None:
    """The default (fp16) engine finds the faces OpenCV's FaceDetectorYN finds on the same input."""
    engine = build_model_engine(YuNet, YUNET_ONNX)
    model = YuNet(engine, preprocessor="cpu", warmup=False)
    tensor, ratios, padding = model.preprocess(images["people"].array)
    (ratio, _), (pad_x, pad_y) = ratios[0], padding[0]
    # the exported graph flips rgb to bgr itself, so opencv gets the rgb letterboxed image
    letterboxed = np.ascontiguousarray(
        np.rint(np.asarray(tensor)[0].transpose(1, 2, 0)).astype(np.uint8)
    )
    width, height = model.input_shape
    detector = cv2.FaceDetectorYN.create(str(YUNET_ONNX), "", (width, height), YUNET_CONF, 0.3)
    expected = detector.detect(letterboxed)[1]
    boxes, scores, _, kpts = model.run(tensor, ratios[0], padding[0], YUNET_CONF, preprocessed=True)

    assert len(scores) == len(expected) == 3
    for face in expected:
        x, y, w, h = face[:4]
        box = [
            (x - pad_x) / ratio,
            (y - pad_y) / ratio,
            (x + w - pad_x) / ratio,
            (y + h - pad_y) / ratio,
        ]
        j = max(range(len(boxes)), key=lambda k: _iou(box, boxes[k]))
        assert _iou(box, boxes[j]) >= 0.95
        assert scores[j] == pytest.approx(face[14], abs=0.01)
        landmarks = (face[4:14].reshape(5, 2) - [pad_x, pad_y]) / ratio
        assert np.linalg.norm(landmarks - kpts[j, :, :2], axis=1).max() < 0.5


def test_yunet_rejects_cuda_postprocessor(build_model_engine) -> None:
    """YuNet engines decode on the CPU: asking for the CUDA postprocessor raises."""
    engine = build_model_engine(YuNet, YUNET_ONNX)
    model = YuNet(engine, warmup=False)
    assert model._postprocessor == "cpu"
    with pytest.raises(ValueError, match="supports YOLO, not YUNET"):
        YuNet(engine, postprocessor="cuda", warmup=False)
