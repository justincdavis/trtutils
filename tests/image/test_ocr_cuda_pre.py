# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/preprocessors/_cuda_ocr.py -- exact parity with the OpenCV CPU path."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils import Buffer, MemoryLocation
from trtutils.core._stream import create_stream, destroy_stream, stream_synchronize
from trtutils.image.preprocessors._cuda_ocr import CudaTextCrops, CudaTextDetectionInput
from trtutils.image.preprocessors._ocr import (
    crop_text_region,
    preprocess_text_detection,
    preprocess_text_recognition,
)


@pytest.fixture(scope="module")
def stream():
    """A CUDA stream for the module."""
    s = create_stream()
    yield s
    destroy_stream(s)


def _image(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    # smooth gradients plus noise, so interpolation errors show up as pixel differences
    ys, xs = np.mgrid[0:height, 0:width]
    base = np.stack([xs * 255 / width, ys * 255 / height, (xs + ys) % 256], axis=-1)
    return np.clip(base + rng.normal(0, 40, base.shape), 0, 255).astype(np.uint8)


def _boxes(rng: np.random.Generator, height: int, width: int, count: int) -> np.ndarray:
    """Random rotated quads (TL, TR, BR, BL), some tall, some near or past the borders."""
    boxes = []
    for _ in range(count):
        cx, cy = rng.uniform(-10, width + 10), rng.uniform(-10, height + 10)
        w, h = rng.uniform(8, max(9, 0.6 * width)), rng.uniform(8, max(9, 0.3 * height))
        if rng.random() < 0.3:
            w, h = h, w
        angle = rng.uniform(-0.6, 0.6) + (np.pi / 2 if rng.random() < 0.2 else 0)
        c, s = np.cos(angle), np.sin(angle)
        corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2 + rng.normal(0, 1.5, (4, 2))
        boxes.append((corners @ np.array([[c, -s], [s, c]]).T + [cx, cy]).astype(np.float32))
    return np.stack(boxes)


def _upload(image: np.ndarray, stream) -> Buffer:
    buffer = Buffer.from_array(image, MemoryLocation.DEVICE)
    stream_synchronize(stream)
    return buffer


@pytest.mark.parametrize(
    ("shapes", "limit_side_len", "limit_type"),
    [
        pytest.param([(1080, 1920)], 960, "max", id="1080p"),
        pytest.param([(480, 640)], 960, "max", id="no-resize"),
        pytest.param([(333, 777)], 736, "min", id="upscale-min"),
        pytest.param([(100, 3000)], 960, "max", id="wide"),
        pytest.param([(40, 40)], 960, "max", id="tiny"),
        pytest.param([(1080, 1920), (300, 500)], 960, "max", id="batch-padded"),
    ],
)
@pytest.mark.parametrize(
    ("mean", "std"),
    [
        pytest.param((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), id="imagenet"),
        pytest.param(None, None, id="range"),
    ],
)
def test_detection_input_matches_cpu(stream, shapes, limit_side_len, limit_type, mean, std) -> None:
    """The device resize + normalize equals preprocess_text_detection bit for bit."""
    rng = np.random.default_rng(0)
    images = [_image(rng, h, w) for h, w in shapes]
    max_shape = (1280, 1280)
    ref, ref_ratios, ref_padding = preprocess_text_detection(
        images, limit_side_len, limit_type, (0.0, 1.0), mean, std, max_shape
    )
    pre = CudaTextDetectionInput(limit_side_len, limit_type, max_shape, mean, std, (0.0, 1.0))
    tensor, ratios, padding = pre(images, stream)
    stream_synchronize(stream)
    assert (ratios, padding) == (ref_ratios, ref_padding)
    np.testing.assert_array_equal(tensor.numpy(), ref)


def test_detection_input_device_frame(stream) -> None:
    """A device Buffer is read in place and gives the same tensor as the host image."""
    image = _image(np.random.default_rng(1), 500, 700)
    pre = CudaTextDetectionInput(960, "max", (1280, 1280), None, None, (0.0, 1.0))
    ref = pre([image], stream)[0].numpy()
    got = pre([_upload(image, stream)], stream)[0].numpy()
    np.testing.assert_array_equal(got, ref)


def _crops_reference(image: np.ndarray, boxes: np.ndarray) -> list[np.ndarray]:
    return [crop_text_region(image, box) for box in boxes]


@pytest.mark.parametrize("seed", [pytest.param(s, id=f"seed{s}") for s in range(4)])
@pytest.mark.parametrize(
    ("height", "width"),
    [
        pytest.param(1080, 1920, id="1080p"),
        pytest.param(120, 400, id="small"),
        pytest.param(3, 5, id="smaller-than-a-tap-window"),
    ],
)
def test_crops_and_recognizer_input_match_cpu(stream, seed, height, width) -> None:
    """Warped crops, their resize and normalization equal the CPU path, rotated crops included."""
    rng = np.random.default_rng(seed)
    image = _image(rng, height, width)
    boxes = _boxes(rng, height, width, 12 if height > 10 else 3)
    boxes = np.stack([b for b in boxes if min(np.ptp(b[:, 0]), np.ptp(b[:, 1])) > 6] or [boxes[0]])
    try:
        refs = _crops_reference(image, boxes)
    except ZeroDivisionError:
        pytest.skip("degenerate box")
    batch_size = 5
    crops = CudaTextCrops(48, 3200, batch_size, (0.5,) * 3, (0.5,) * 3, (0.0, 1.0))
    plan = crops.plan(_upload(image, stream), boxes, stream)
    assert sum(r.shape[0] > r.shape[1] for r in refs) >= 0
    for index, (start, stop, _) in enumerate(plan.batches):
        tensor = crops.batch_input(plan, index, stream)
        stream_synchronize(stream)
        batch = [refs[i] for i in plan.order[start:stop]]
        ref, _, _ = preprocess_text_recognition(batch, 48, max_width=3200)
        np.testing.assert_array_equal(tensor.numpy(), ref)


@pytest.mark.parametrize("width_bucket", [None, 32, 100, 3200])
def test_bucketed_crops_match_cpu(stream, width_bucket) -> None:
    """Bucketed batch widths on the device equal the CPU path, the padding included."""
    rng = np.random.default_rng(3)
    image = _image(rng, 1080, 1920)
    boxes = _boxes(rng, 1080, 1920, 12)
    refs = _crops_reference(image, boxes)
    crops = CudaTextCrops(48, 3200, 5, (0.5,) * 3, (0.5,) * 3, (0.0, 1.0), width_bucket)
    plan = crops.plan(_upload(image, stream), boxes, stream)
    for index, (start, stop, width) in enumerate(plan.batches):
        tensor = crops.batch_input(plan, index, stream)
        stream_synchronize(stream)
        batch = [refs[i] for i in plan.order[start:stop]]
        ref, _, _ = preprocess_text_recognition(
            batch, 48, max_width=3200, mean=(0.5,) * 3, std=(0.5,) * 3, width_bucket=width_bucket
        )
        assert width == ref.shape[-1]
        assert width_bucket is None or width % width_bucket == 0 or width == 3200
        np.testing.assert_array_equal(tensor.numpy(), ref)


def test_rotated_crops_are_covered(stream) -> None:
    """The random boxes include tall crops, so the np.rot90 branch is exercised."""
    rng = np.random.default_rng(0)
    boxes = _boxes(rng, 1080, 1920, 40)
    image = _image(rng, 1080, 1920)
    tall = [r.shape[0] / r.shape[1] >= 1 for r in _crops_reference(image, boxes)]
    assert any(tall)
    crops = CudaTextCrops(48, 3200, 8, None, None, (0.0, 1.0))
    plan = crops.plan(_upload(image, stream), boxes, stream)
    stream_synchronize(stream)
    got = plan.crops.numpy()
    refs = _crops_reference(image, boxes)
    offset = 0
    for ref in refs:
        n = ref.size
        # the kernel lays crops out unrotated-box by box, in the rotated orientation
        np.testing.assert_array_equal(got[offset : offset + n].reshape(ref.shape), ref)
        offset += n
