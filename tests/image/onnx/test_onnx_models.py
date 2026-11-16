# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/onnx_models.py -- TRT image preprocessing engines."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import tensorrt as trt

from trtutils import Buffer, TRTEngine
from trtutils._flags import FLAGS
from trtutils.image.onnx_models import build_image_preproc, build_image_preproc_imagenet

SCALE = np.array([1.0 / 255.0], dtype=np.float32)
OFFSET = np.array([0.0], dtype=np.float32)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


@pytest.mark.parametrize(
    ("build", "extra_inputs", "reference"),
    [
        pytest.param(
            build_image_preproc,
            [SCALE, OFFSET],
            lambda rgb: rgb * SCALE + OFFSET,
            id="preproc",
        ),
        pytest.param(
            build_image_preproc_imagenet,
            [MEAN, STD],
            lambda rgb: (rgb / 255.0 - MEAN) / STD,
            id="preproc-imagenet",
        ),
    ],
)
@pytest.mark.parametrize(
    ("dtype", "tol"),
    [
        pytest.param(np.float32, 1e-6, id="fp32"),
        pytest.param(
            np.float16,
            4 * np.finfo(np.float16).eps,
            id="fp16",
            marks=pytest.mark.skipif(
                not FLAGS.WEAK_TYPING, reason="strongly typed networks keep the graph's fp32 output"
            ),
        ),
    ],
)
def test_preproc_engine_matches_reference(
    images, build, extra_inputs, reference, dtype, tol
) -> None:
    """A preprocessing engine builds for each output dtype and matches a float64 reference."""
    img = cv2.resize(images["horse"].array, (640, 640))
    rgb = img[np.newaxis, ..., ::-1].transpose(0, 3, 1, 2).astype(np.float64)
    expected = reference(rgb)

    engine = TRTEngine(build((640, 640), np.dtype(dtype), trt_version=trt.__version__), warmup=False)
    inputs = [Buffer.wrap(data) for data in [img[np.newaxis], *extra_inputs]]
    result = engine.execute(inputs)[0]
    assert result.shape == expected.shape
    assert result.dtype == dtype
    np.testing.assert_allclose(result.astype(np.float64), expected, rtol=tol, atol=tol)
