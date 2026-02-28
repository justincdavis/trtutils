# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/sahi/_sahi.py -- sliced inference over a Detector."""

from __future__ import annotations

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
