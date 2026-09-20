# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/onnx_models.py -- TRT image preprocessing engines."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import tensorrt as trt

from trtutils import TRTEngine
from trtutils.image.onnx_models import build_image_preproc, build_image_preproc_imagenet
from trtutils.image.preprocessors import preprocess

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@pytest.mark.parametrize(
    ("build", "extra_inputs", "norm", "tol"),
    [
        pytest.param(
            build_image_preproc,
            [np.array([1.0 / 255.0], dtype=np.float32), np.array([0.0], dtype=np.float32)],
            {},
            5e-4,
            id="preproc",
        ),
        pytest.param(
            build_image_preproc_imagenet,
            [
                np.array(MEAN, dtype=np.float32).reshape(1, 3, 1, 1),
                np.array(STD, dtype=np.float32).reshape(1, 3, 1, 1),
            ],
            {"mean": MEAN, "std": STD},
            2e-3,
            id="preproc-imagenet",
        ),
    ],
)
def test_preproc_engine_matches_cpu(images, build, extra_inputs, norm, tol) -> None:
    """A built preprocessing engine reproduces CPU preprocess() on an already-resized image."""
    img = cv2.resize(images["horse"].array, (640, 640))
    expected, _ratios, _padding = preprocess([img], (640, 640), np.dtype(np.float32), **norm)

    engine = TRTEngine(
        build((640, 640), np.dtype(np.float32), trt_version=trt.__version__), warmup=False
    )
    result = engine.execute([img, *extra_inputs])[0]
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    np.testing.assert_allclose(result, expected, rtol=tol, atol=tol)
