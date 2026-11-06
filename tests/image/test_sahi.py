# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/sahi/_sahi.py -- sliced inference over a Detector."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.image import SAHI, Detector


@pytest.mark.parametrize(
    "slice_size",
    [
        pytest.param(None, id="slice-default"),
        pytest.param((320, 320), id="slice-320"),
    ],
)
def test_sahi_end2end(yolov10_engine, images, slice_size) -> None:
    """SAHI end2end() finds the ground-truth class with in-bounds boxes."""
    sahi = SAHI(Detector(yolov10_engine, warmup=False), slice_size=slice_size)
    image = images["horse"]
    height, width = image.array.shape[:2]

    detections = sahi.end2end(image.array)
    assert set(image.gt_det_classes) <= {cls_id for _bbox, _score, cls_id in detections}
    for (x1, y1, x2, y2), score, _cls_id in detections:
        assert 0 <= x1 <= x2 <= width
        assert 0 <= y1 <= y2 <= height
        assert 0.0 <= score <= 1.0


class _StubDetector:
    """Detector stand-in returning fixed detections for every patch."""

    input_shape = (100, 100)

    def __init__(self, dets) -> None:
        self._dets = dets

    def end2end(self, patches, **_kwargs: object):
        return [list(self._dets) for _ in patches]


# boxes a and b overlap with IoU 0.6
_BOX_A = (0, 0, 100, 100)
_BOX_B = (0, 0, 100, 60)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("dets", "init_kwargs", "call_kwargs", "expected"),
    [
        pytest.param([(_BOX_A, 0.9, 0), (_BOX_B, 0.8, 0)], {}, {}, 1, id="iou-init-merges"),
        pytest.param(
            [(_BOX_A, 0.9, 0), (_BOX_B, 0.8, 0)],
            {},
            {"nms_iou_thres": 0.7},
            2,
            id="iou-call-overrides-init",
        ),
        pytest.param(
            [(_BOX_A, 0.9, 0), (_BOX_B, 0.8, 1)], {}, {}, 2, id="agnostic-init-keeps-classes"
        ),
        pytest.param(
            [(_BOX_A, 0.9, 0), (_BOX_B, 0.8, 1)],
            {},
            {"agnostic_nms": True},
            1,
            id="agnostic-call-overrides-init",
        ),
        pytest.param(
            [(_BOX_A, 0.9, 0), (_BOX_B, 0.8, 1)],
            {"agnostic_nms": True},
            {"agnostic_nms": False},
            2,
            id="agnostic-call-false-overrides-init-true",
        ),
    ],
)
def test_sahi_merge_nms_per_call_overrides(dets, init_kwargs, call_kwargs, expected) -> None:
    """The final merge NMS uses per-call nms_iou_thres/agnostic_nms over init values."""
    sahi = SAHI(_StubDetector(dets), iou_threshold=0.5, **init_kwargs)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert len(sahi.end2end(image, **call_kwargs)) == expected
