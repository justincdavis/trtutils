# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine-level tests for the pose estimator wrappers in src/trtutils/models/pose_estimators/."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest

from tests.conftest import DATA_DIR, PREPROCESSORS
from trtutils.image._schema import PoseEstimatorOutputSchema
from trtutils.image.postprocessors import postprocess_rtmo_pose
from trtutils.models import RTMO

RTMO_MODELS = [
    pytest.param(RTMO, DATA_DIR / "rtmo" / "rtmo_s_640.onnx", id="rtmo-s"),
    pytest.param(RTMO, DATA_DIR / "rtmo" / "rtmo_m_640.onnx", id="rtmo-m"),
    pytest.param(RTMO, DATA_DIR / "rtmo" / "rtmo_l_640.onnx", id="rtmo-l"),
]


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
