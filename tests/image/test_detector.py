# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/_detector.py -- Detector inference on a YOLOv10 engine."""

from __future__ import annotations

import pytest

from trtutils.image import Detector


@pytest.mark.parametrize("preprocessor", ["cpu", "cuda", "trt"])
def test_detector_end2end(yolov10_engine, images, preprocessor) -> None:
    """run(), get_detections(), and end2end() agree, honor conf_thres, and find the ground truth."""
    model = Detector(yolov10_engine, preprocessor=preprocessor, warmup=False)
    image = images["horse"]
    height, width = image.array.shape[:2]

    raw = model.run(image.array, postprocess=False)
    assert [(list(o.shape), o.dtype) for o in raw] == model.engine.output_spec

    postprocessed = model.run(image.array)
    bboxes, scores, class_ids = postprocessed
    assert bboxes.shape == (len(scores), 4)
    assert class_ids.shape == scores.shape

    via_run = model.get_detections(postprocessed)
    via_e2e = model.end2end(image.array)
    assert via_run == via_e2e

    assert len(via_run) >= image.gt_det_min
    assert set(image.gt_det_classes) <= {cls_id for _bbox, _score, cls_id in via_run}
    for (x1, y1, x2, y2), score, _cls_id in via_run:
        assert 0 <= x1 <= x2 <= width
        assert 0 <= y1 <= y2 <= height
        assert 0.0 <= score <= 1.0

    strict = model.get_detections(postprocessed, conf_thres=0.5)
    assert len(strict) <= len(via_run)
    assert all(score >= 0.5 for _bbox, score, _cls_id in strict)
