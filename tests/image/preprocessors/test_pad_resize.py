# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/preprocessors/_cpu.py and _image_model.py -- the 'pad' resize."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from trtutils.image import (
    Detector,
    HandInteractionDetector,
    OBBDetector,
    PoseEstimator,
    Segmenter,
)
from trtutils.image.preprocessors._cpu import letterbox, resize_image


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("image_hw", "new_shape", "ratio", "inner"),
    [
        pytest.param((100, 200), (640, 640), 3.2, (320, 640), id="wide-to-square"),
        pytest.param((200, 100), (640, 640), 3.2, (640, 320), id="tall-to-square"),
        pytest.param((100, 200), (384, 640), 1.92, (192, 384), id="wide-to-tall"),
        pytest.param((200, 100), (640, 384), 1.92, (384, 192), id="tall-to-wide"),
        pytest.param((1080, 1920), (640, 640), 1 / 3, (360, 640), id="downscale"),
        pytest.param((50, 50), (100, 100), 2.0, (100, 100), id="exact-no-fill"),
        pytest.param((360, 640), (640, 640), 1.0, (360, 640), id="fits-no-resize"),
        pytest.param((101, 200), (200, 200), 1.0, (101, 200), id="fits-odd-fill"),
    ],
)
def test_pad_resize(image_hw, new_shape, ratio, inner) -> None:
    """Pad is letterbox's resize at the top-left, gray (114) below and right, padding (0, 0)."""
    image = np.random.default_rng(0).integers(0, 256, (*image_hw, 3), dtype=np.uint8)
    out, ratios, padding = resize_image(image, new_shape, "pad")
    assert out.shape == (new_shape[1], new_shape[0], 3)
    assert ratios == pytest.approx((ratio, ratio))
    assert padding == (0.0, 0.0)

    height, width = inner
    expected = image
    if image_hw != inner:
        expected = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(out[:height, :width], expected)
    assert (out[height:] == 114).all()
    assert (out[:, width:] == 114).all()

    # the same pixels letterbox centres
    boxed, boxed_ratios, (pad_x, pad_y) = letterbox(image, new_shape)
    top, left = round(pad_y - 0.1), round(pad_x - 0.1)
    np.testing.assert_array_equal(
        out[:height, :width], boxed[top : top + height, left : left + width]
    )
    assert boxed_ratios == ratios


@pytest.mark.cpu
@pytest.mark.parametrize(
    "model_cls",
    [
        pytest.param(Detector, id="detector"),
        pytest.param(Segmenter, id="segmenter"),
        pytest.param(PoseEstimator, id="pose"),
        pytest.param(OBBDetector, id="obb"),
        pytest.param(HandInteractionDetector, id="hand-interaction"),
    ],
)
def test_image_tasks_reject_pad_resize(model_cls, tmp_path) -> None:
    """Tasks that map results back to the image raise for 'pad' before loading the engine."""
    with pytest.raises(ValueError, match="resize_method='pad'"):
        model_cls(tmp_path / "missing.engine", resize_method="pad")
