# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/postprocessors/_gpu_ocr.py -- CUDA OCR postprocessing parity."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils import Buffer, MemoryLocation
from trtutils.core._stream import create_stream, destroy_stream
from trtutils.image.postprocessors._gpu_ocr import CtcArgmax, DbBinarize
from trtutils.image.postprocessors._ocr import (
    postprocess_text_detection,
    postprocess_text_recognition,
)


def _ctc(
    rng: np.random.Generator, batch: int, steps: int, classes: int, *, ties: bool
) -> np.ndarray:
    probs = rng.uniform(0, 1, size=(batch, steps, classes)).astype(np.float32)
    if ties:
        # quantized so equal maxima happen, and the lowest class index must win
        probs = np.round(probs * 4) / 4
    return probs


@pytest.mark.parametrize(
    ("batch", "steps", "classes", "ties"),
    [
        pytest.param(1, 40, 6625, False, id="ppocr"),
        pytest.param(6, 80, 6625, False, id="batch6"),
        pytest.param(2, 25, 97, True, id="ties"),
        pytest.param(3, 7, 5, True, id="fewer-classes-than-threads"),
        pytest.param(1, 1, 300, True, id="one-step"),
    ],
)
def test_cuda_ctc_matches_cpu(batch, steps, classes, ties) -> None:
    """Indices and probabilities match postprocess_text_recognition, dtypes included."""
    probs = _ctc(np.random.default_rng(0), batch, steps, classes, ties=ties)
    ref = postprocess_text_recognition([probs])

    stream = create_stream()
    try:
        post = CtcArgmax()
        # twice, then a smaller batch, so the grown buffers are reused
        for run in (probs, probs, probs[:1]):
            ours = post(Buffer.from_array(run, MemoryLocation.DEVICE), stream)
            for got, want in zip(ours, ref):
                for g, w in zip(got, want):
                    np.testing.assert_array_equal(g, w)
                    assert g.dtype == w.dtype
            ref = postprocess_text_recognition([run])
    finally:
        destroy_stream(stream)


def _prob_map(rng: np.random.Generator, batch: int, height: int, width: int) -> np.ndarray:
    """Noisy background with a few high-probability text blobs and a mid-probability one."""
    maps = rng.uniform(0, 0.25, size=(batch, 1, height, width)).astype(np.float32)
    for b in range(batch):
        for i in range(4):
            y, x = 6 + i * (height // 5), int(rng.integers(4, width // 3))
            maps[b, 0, y : y + 10, x : x + width // 2] = rng.uniform(0.7, 0.95)
        maps[b, 0, 2:10, 2:12] = 0.45  # above thresh, below box thresh
    return maps


@pytest.mark.parametrize(
    ("batch", "thresh", "box_thresh", "use_dilation", "padding"),
    [
        pytest.param(1, 0.3, 0.6, False, (0.0, 0.0), id="default"),
        pytest.param(1, 0.3, 0.6, True, (0.0, 0.0), id="dilation"),
        pytest.param(1, 0.5, 0.3, False, (0.0, 0.0), id="low-box-thresh"),
        pytest.param(1, 0.99, 0.6, False, (0.0, 0.0), id="nothing-passes"),
        pytest.param(1, 0.3, 0.6, False, (16.0, 8.0), id="padded"),
        pytest.param(2, 0.3, 0.6, False, (0.0, 0.0), id="batch2"),
    ],
)
def test_cuda_det_matches_cpu(batch, thresh, box_thresh, use_dilation, padding) -> None:
    """Boxes and scores match postprocess_text_detection from the CPU thresholded map."""
    height, width = 96, 160
    maps = _prob_map(np.random.default_rng(1), batch, height, width)
    ratios = [(0.5, 0.5)] * batch
    pads = [padding] * batch
    kwargs = {"use_dilation": use_dilation}
    ref = postprocess_text_detection([maps], ratios, pads, thresh, box_thresh, **kwargs)
    if thresh < 0.9:
        assert any(len(boxes) for boxes, _ in ref)

    stream = create_stream()
    try:
        post = DbBinarize()
        for _ in range(2):
            device = Buffer.from_array(maps, MemoryLocation.DEVICE)
            bitmaps, pred, wait = post(device, thresh, stream)
            ours = postprocess_text_detection(
                [pred], ratios, pads, thresh, box_thresh, bitmaps=bitmaps, wait=wait, **kwargs
            )
            assert len(ours) == len(ref)
            for got, want in zip(ours, ref):
                for g, w in zip(got, want):
                    np.testing.assert_array_equal(g, w)
                    assert g.dtype == w.dtype
    finally:
        destroy_stream(stream)
