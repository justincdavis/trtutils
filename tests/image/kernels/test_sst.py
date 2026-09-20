# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for the scale-swap-transpose kernels in src/trtutils/image/kernels.py."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from trtutils.image import kernels
from trtutils.image.preprocessors import preprocess

from .conftest import run_kernel

SIZE = (640, 640)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@pytest.mark.parametrize(
    ("spec", "dtype", "tol"),
    [
        pytest.param(kernels.SCALE_SWAP_TRANSPOSE, np.float32, 1e-6, id="sst"),
        pytest.param(kernels.SST_FAST, np.float32, 1e-6, id="sst-fast"),
        pytest.param(kernels.SST_FAST_F16, np.float16, 1e-3, id="sst-fast-f16"),
    ],
)
def test_sst(cuda_stream, images, batch_size, spec, dtype, tol) -> None:
    """SST kernels match CPU preprocess() for every batch element within dtype tolerance."""
    img = cv2.resize(images["horse"].array, SIZE)
    expected, _ratios, _padding = preprocess([img], SIZE, np.dtype(np.float32))

    batch = np.stack([img] * batch_size)
    output = np.zeros((batch_size, 3, SIZE[1], SIZE[0]), dtype=dtype)
    result = run_kernel(
        cuda_stream,
        spec,
        [batch, output, 1.0 / 255.0, 0.0, SIZE[1], SIZE[0], batch_size],
        output,
        (SIZE[0] // 32, SIZE[1] // 32, batch_size),
    )
    np.testing.assert_allclose(result, np.repeat(expected, batch_size, axis=0), rtol=tol, atol=tol)


@pytest.mark.parametrize(
    ("spec", "dtype", "tol"),
    [
        pytest.param(kernels.IMAGENET_SST, np.float32, 1e-6, id="imagenet-sst"),
        pytest.param(kernels.IMAGENET_SST_F16, np.float16, 1e-3, id="imagenet-sst-f16"),
    ],
)
def test_imagenet_sst(cuda_stream, images, batch_size, spec, dtype, tol) -> None:
    """ImageNet SST kernels apply mean/std like CPU preprocess() for every batch element."""
    img = cv2.resize(images["horse"].array, SIZE)
    expected, _ratios, _padding = preprocess([img], SIZE, np.dtype(np.float32), mean=MEAN, std=STD)

    batch = np.stack([img] * batch_size)
    output = np.zeros((batch_size, 3, SIZE[1], SIZE[0]), dtype=dtype)
    result = run_kernel(
        cuda_stream,
        spec,
        [
            batch,
            output,
            np.array(MEAN, dtype=np.float32).reshape(1, 3, 1, 1),
            np.array(STD, dtype=np.float32).reshape(1, 3, 1, 1),
            SIZE[1],
            SIZE[0],
            batch_size,
        ],
        output,
        (SIZE[0] // 32, SIZE[1] // 32, batch_size),
    )
    np.testing.assert_allclose(result, np.repeat(expected, batch_size, axis=0), rtol=tol, atol=tol)
