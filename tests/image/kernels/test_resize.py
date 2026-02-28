# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for the letterbox and linear resize kernels in src/trtutils/image/kernels.py."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from cv2ext.image import letterbox

from trtutils.image import kernels

from .conftest import run_kernel

SHAPES = [
    pytest.param((640, 640), id="640x640"),
    pytest.param((640, 480), id="640x480"),
    pytest.param((320, 320), id="320x320"),
]


def _assert_close_uint8(result: np.ndarray, expected: np.ndarray) -> None:
    """Resampled uint8 images agree to within rounding on average."""
    assert result.shape == expected.shape
    assert np.abs(result.astype(np.int32) - expected.astype(np.int32)).mean() < 1.0


@pytest.mark.parametrize("shape", SHAPES)
def test_letterbox(cuda_stream, images, batch_size, shape) -> None:
    """The letterbox kernel matches cv2ext.letterbox for every batch element."""
    img = images["horse"].array
    height, width = img.shape[:2]
    out_width, out_height = shape
    expected, _ratios, _padding = letterbox(img, new_shape=shape)

    scale = min(out_width / width, out_height / height)
    new_width, new_height = int(width * scale), int(height * scale)
    pad_x, pad_y = int((out_width - new_width) / 2), int((out_height - new_height) / 2)

    batch = np.stack([img] * batch_size)
    output = np.zeros((batch_size, out_height, out_width, 3), dtype=np.uint8)
    result = run_kernel(
        cuda_stream,
        kernels.LETTERBOX_RESIZE,
        [
            batch,
            output,
            width,
            height,
            out_width,
            out_height,
            pad_x,
            pad_y,
            new_width,
            new_height,
            img.size,
            output[0].size,
        ],
        output,
        (math.ceil(out_width / 32), math.ceil(out_height / 32), batch_size),
    )
    for element in result:
        _assert_close_uint8(element, expected)


@pytest.mark.parametrize("shape", SHAPES)
def test_linear(cuda_stream, images, batch_size, shape) -> None:
    """The linear resize kernel matches cv2.resize(INTER_LINEAR) for every batch element."""
    img = images["horse"].array
    height, width = img.shape[:2]
    out_width, out_height = shape
    expected = cv2.resize(img, shape, interpolation=cv2.INTER_LINEAR)

    batch = np.stack([img] * batch_size)
    output = np.zeros((batch_size, out_height, out_width, 3), dtype=np.uint8)
    result = run_kernel(
        cuda_stream,
        kernels.LINEAR_RESIZE,
        [batch, output, width, height, out_width, out_height, img.size, output[0].size],
        output,
        (math.ceil(out_width / 32), math.ceil(out_height / 32), batch_size),
    )
    for element in result:
        _assert_close_uint8(element, expected)
