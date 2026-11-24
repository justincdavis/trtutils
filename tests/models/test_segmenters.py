# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine-level tests for the segmenter wrappers in src/trtutils/models/segmenters/."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest

from tests.conftest import DATA_DIR, PREPROCESSORS
from trtutils.image._schema import SegmenterOutputSchema
from trtutils.image.postprocessors import postprocess_yolo_seg
from trtutils.models import YOLOv9Seg, YOLOv12Seg

SEGMENTERS = [
    pytest.param(YOLOv9Seg, DATA_DIR / "yolov9_seg" / "yolov9c-seg_640.onnx", id="yolov9c-seg"),
    pytest.param(YOLOv9Seg, DATA_DIR / "yolov9_seg" / "yolov9e-seg_640.onnx", id="yolov9e-seg"),
    pytest.param(YOLOv12Seg, DATA_DIR / "yolov12_seg" / "yolov12n-seg_640.onnx", id="yolov12n-seg"),
    pytest.param(YOLOv12Seg, DATA_DIR / "yolov12_seg" / "yolov12x-seg_640.onnx", id="yolov12x-seg"),
]


def _iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(iw, 0) * max(ih, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _mask_iou(a, b) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    return np.logical_and(a, b).sum() / np.logical_or(a, b).sum()


@pytest.mark.parametrize(("model_cls", "onnx_path"), SEGMENTERS)
@pytest.mark.parametrize(("preprocessor", "cpu_resize"), PREPROCESSORS)
def test_segmenter_end2end(
    build_model_engine, images, model_cls, onnx_path, preprocessor, cpu_resize
) -> None:
    """run(), get_segmentations(), and end2end() agree and segment the people."""
    engine = build_model_engine(model_cls, onnx_path)
    model = model_cls(engine, preprocessor=preprocessor, cpu_resize=cpu_resize, warmup=False)
    assert model.output_schema is SegmenterOutputSchema.YOLO
    image = images["people"].array
    height, width = image.shape[:2]

    via_run = model.get_segmentations(model.run(image))
    via_e2e = model.end2end(image)
    assert len(via_run) == len(via_e2e)
    for (box_a, score_a, cls_a, mask_a), (box_b, score_b, cls_b, mask_b) in zip(via_run, via_e2e):
        assert _iou(box_a, box_b) > 0.95
        assert score_a == pytest.approx(score_b, abs=0.02)
        assert cls_a == cls_b
        assert _mask_iou(mask_a, mask_b) > 0.95

    people = [seg for seg in via_e2e if seg[1] >= 0.5 and seg[2] == 0]
    assert len(people) >= 3
    for (x1, y1, x2, y2), _, _, mask in people:
        assert 0 <= x1 < x2 <= width
        assert 0 <= y1 < y2 <= height
        assert mask.shape == (height, width)
        assert mask[y1:y2, x1:x2].any()


@pytest.mark.correctness
@pytest.mark.parametrize(("model_cls", "onnx_path"), SEGMENTERS)
@pytest.mark.parametrize("image_name", ["people", "horse"])
def test_segmenter_matches_onnxruntime(
    build_model_engine, images, model_cls, onnx_path, image_name
) -> None:
    """The default (fp16) engine segments what onnxruntime does on the same tensor."""
    engine = build_model_engine(model_cls, onnx_path)
    model = model_cls(engine, preprocessor="cpu", warmup=False)
    tensor, ratios, padding = model.preprocess(images[image_name].array)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    expected = postprocess_yolo_seg(
        session.run(["output0", "output1"], {"images": np.asarray(tensor)}),
        ratios,
        padding,
        0.25,
        0.5,
        input_size=model.input_shape,
    )[0]
    boxes, scores, class_ids, masks = model.run(
        tensor, ratios[0], padding[0], 0.25, preprocessed=True
    )

    confident = np.flatnonzero(expected[1] >= 0.5)
    assert len(confident) > 0
    for i in confident:
        j = max(range(len(boxes)), key=lambda k: _iou(expected[0][i], boxes[k]))
        assert _iou(expected[0][i], boxes[j]) >= 0.9
        assert class_ids[j] == expected[2][i]
        assert scores[j] == pytest.approx(expected[1][i], abs=0.05)
        assert _mask_iou(expected[3][i], masks[j]) >= 0.9
