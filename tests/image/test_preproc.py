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

GPU_CLASSES = [
    pytest.param(CUDAPreprocessor, id="cuda"),
    pytest.param(TRTPreprocessor, id="trt"),
]


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


@pytest.mark.parametrize("gpu_cls", GPU_CLASSES)
def test_heterogeneous_batch_matches_singles(random_images, gpu_cls) -> None:
    """Heterogeneous batch outputs match preprocessing each image alone."""
    kwargs = {"batch_size": 6} if gpu_cls is TRTPreprocessor else {}
    preproc = gpu_cls(SIZE, RANGE, DTYPE, **kwargs)
    sizes = [(480, 640), (720, 1280), (600, 800), (1080, 1920), (512, 512), (768, 1024)]
    imgs = [random_images(1, height, width)[0] for height, width in sizes]

    singles = [preproc.preprocess([img]) for img in imgs]
    batch, ratios, padding = preproc.preprocess(imgs)

    assert batch.shape[0] == len(imgs)
    for i in range(len(imgs)):
        tensor, single_ratios, single_padding = singles[i]
        np.testing.assert_array_equal(batch[i], tensor[0])
        assert ratios[i] == single_ratios[0]
        assert padding[i] == single_padding[0]


@pytest.mark.parametrize("gpu_cls", GPU_CLASSES)
def test_resolution_switch_matches_fresh(random_images, gpu_cls) -> None:
    """Alternating resolutions match a freshly constructed preprocessor at every step."""
    preproc = gpu_cls(SIZE, RANGE, DTYPE)
    sizes = [(480, 640), (720, 1280), (1080, 1920), (480, 640), (1080, 1920)]

    for height, width in sizes:
        img = random_images(1, height, width)[0]
        result, ratios, padding = preproc.preprocess([img])

        fresh = gpu_cls(SIZE, RANGE, DTYPE)
        expected, exp_ratios, exp_padding = fresh.preprocess([img])

        np.testing.assert_array_equal(result, expected)
        assert ratios == exp_ratios
        assert padding == exp_padding

    # 3 distinct resolutions were seen: the staging pool and resize-arg cache
    # hold exactly one entry per resolution, not one per call
    assert len(preproc._staging_pool) == 3
    assert len(preproc._cached_resize_args) == 3


@pytest.mark.parametrize("gpu_cls", GPU_CLASSES)
def test_homogeneous_batch_grow_then_shrink(random_images, gpu_cls) -> None:
    """Batch of 8 matches singles; shrinking to 2 and back to 8 reuses the batch binding."""
    kwargs = {"batch_size": 8} if gpu_cls is TRTPreprocessor else {}
    preproc = gpu_cls(SIZE, RANGE, DTYPE, **kwargs)
    imgs = random_images(8)
    singles = [preproc.preprocess([img]) for img in imgs]

    def assert_matches(batch, ratios, padding, count) -> None:
        assert batch.shape[0] == count
        for i in range(count):
            tensor, single_ratios, single_padding = singles[i]
            np.testing.assert_array_equal(batch[i], tensor[0])
            assert ratios[i] == single_ratios[0]
            assert padding[i] == single_padding[0]

    # grow: initial capacity is smaller than 8 same-sized images
    batch8, ratios8, padding8 = preproc.preprocess(imgs)
    assert_matches(batch8, ratios8, padding8, 8)
    binding_after_grow = preproc._batch_input_binding

    # shrink: must not reallocate, the buffer only exposes a smaller prefix
    batch2, ratios2, padding2 = preproc.preprocess(imgs[:2])
    assert_matches(batch2, ratios2, padding2, 2)
    assert preproc._batch_input_binding is binding_after_grow

    # grow back within the existing high-water-mark: still no reallocation
    batch8b, ratios8b, padding8b = preproc.preprocess(imgs)
    assert_matches(batch8b, ratios8b, padding8b, 8)
    assert preproc._batch_input_binding is binding_after_grow
