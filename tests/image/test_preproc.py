# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/preprocessors/ -- CPU, CUDA, and TRT preprocessors."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.image.preprocessors import CPUPreprocessor, CUDAPreprocessor, TRTPreprocessor

SIZE = (640, 640)
RANGE = (0.0, 1.0)
DTYPE = np.dtype(np.float32)
IMAGENET = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}


@pytest.mark.parametrize(
    "gpu_cls",
    [
        pytest.param(CUDAPreprocessor, id="cuda"),
        pytest.param(TRTPreprocessor, id="trt"),
    ],
)
@pytest.mark.parametrize("resize", ["linear", "letterbox"])
@pytest.mark.parametrize(
    ("norm", "tol"),
    [
        pytest.param({}, 0.02, id="raw"),
        pytest.param(IMAGENET, 0.1, id="imagenet"),
    ],
)
def test_gpu_matches_cpu(images, gpu_cls, resize, norm, tol) -> None:
    """GPU preprocessing matches CPU output, ratios, and padding for every test image."""
    cpu = CPUPreprocessor(SIZE, RANGE, DTYPE, resize=resize, **norm)
    gpu = gpu_cls(SIZE, RANGE, DTYPE, resize=resize, **norm)
    for image in images.values():
        expected, ratios, padding = cpu.preprocess([image.array])
        result, gpu_ratios, gpu_padding = gpu.preprocess([image.array])
        assert gpu_ratios == ratios
        assert gpu_padding == padding
        assert result.shape == expected.shape
        assert result.dtype == expected.dtype
        assert np.abs(result - expected).mean() < tol


@pytest.mark.parametrize(
    ("preproc_cls", "kwargs"),
    [
        pytest.param(CPUPreprocessor, {}, id="cpu"),
        pytest.param(CUDAPreprocessor, {}, id="cuda"),
        pytest.param(TRTPreprocessor, {"batch_size": 4}, id="trt"),
    ],
)
def test_batch_matches_single(random_images, preproc_cls, kwargs) -> None:
    """Batches of varying size reproduce the single-image tensors, ratios, and padding exactly."""
    preproc = preproc_cls(SIZE, RANGE, DTYPE, **kwargs)
    imgs = random_images(4)
    singles = [preproc.preprocess([img]) for img in imgs]
    for count in (1, 4, 2):
        batch, ratios, padding = preproc.preprocess(imgs[:count])
        assert batch.shape == (count, 3, 640, 640)
        assert batch.dtype == DTYPE
        assert batch.min() >= RANGE[0]
        assert batch.max() <= RANGE[1]
        assert len(ratios) == len(padding) == count
        for i in range(count):
            tensor, single_ratios, single_padding = singles[i]
            np.testing.assert_array_equal(batch[i], tensor[0])
            assert ratios[i] == single_ratios[0]
            assert padding[i] == single_padding[0]


SHAPES = [(1080, 1920), (480, 640), (2160, 3840), (720, 1280), (512, 512)]


@pytest.mark.parametrize(
    "gpu_cls",
    [
        pytest.param(CUDAPreprocessor, id="cuda"),
        pytest.param(TRTPreprocessor, id="trt"),
    ],
)
@pytest.mark.parametrize(
    "memory",
    [
        pytest.param({"pagelocked_mem": False}, id="pageable"),
        pytest.param({"pagelocked_mem": True}, id="pagelocked"),
        pytest.param({"pagelocked_mem": True, "unified_mem": True}, id="unified"),
    ],
)
@pytest.mark.parametrize(
    "calls",
    [
        pytest.param([[s] for s in SHAPES], id="single"),
        pytest.param([[s] * n for n, s in zip((3, 1, 2, 4, 2), SHAPES)], id="homogeneous"),
        pytest.param([SHAPES[:4], SHAPES[1:], SHAPES[::2]], id="heterogeneous"),
    ],
)
def test_reused_staging_matches_fresh(gpu_cls, memory, calls) -> None:
    """Growing and shrinking inputs on one preprocessor match a fresh one per call."""
    rng = np.random.default_rng(0)
    kwargs = {"batch_size": 4} if gpu_cls is TRTPreprocessor else {}
    reused = gpu_cls(SIZE, RANGE, DTYPE, **memory, **kwargs)
    for shapes in calls:
        imgs = [rng.integers(0, 256, (*s, 3), dtype=np.uint8) for s in shapes]
        expected = gpu_cls(SIZE, RANGE, DTYPE, **memory, **kwargs).preprocess(imgs)
        result = reused.preprocess(imgs)
        np.testing.assert_array_equal(result[0], expected[0])
        assert result[1:] == expected[1:]


@pytest.mark.parametrize(
    "gpu_cls",
    [
        pytest.param(CUDAPreprocessor, id="cuda"),
        pytest.param(TRTPreprocessor, id="trt"),
    ],
)
def test_staging_only_grows(gpu_cls) -> None:
    """Smaller inputs reuse the staging buffers; only a larger input reallocates."""
    kwargs = {"batch_size": 4} if gpu_cls is TRTPreprocessor else {}
    preproc = gpu_cls(SIZE, RANGE, DTYPE, **kwargs)
    single, batch = preproc._input_binding, preproc._batch_input_binding
    preproc.preprocess([np.zeros((480, 640, 3), np.uint8)])
    preproc.preprocess([np.zeros((540, 960, 3), np.uint8)] * 4)
    assert preproc._input_binding is single
    assert preproc._batch_input_binding is batch
    preproc.preprocess([np.zeros((2160, 3840, 3), np.uint8)])
    preproc.preprocess([np.zeros((720, 1280, 3), np.uint8)] * 4)
    assert preproc._input_binding is not single
    assert preproc._batch_input_binding is not batch
