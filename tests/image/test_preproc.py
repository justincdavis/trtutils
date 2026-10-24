# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/preprocessors/ -- CPU, CUDA, and TRT preprocessors."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from cv2ext.image import letterbox, rescale, resize_linear

from trtutils import Buffer, MemoryLocation
from trtutils.image.preprocessors import CPUPreprocessor, CUDAPreprocessor, TRTPreprocessor
from trtutils.image.preprocessors._process import preprocess

SIZE = (640, 640)
RANGE = (0.0, 1.0)
DTYPE = np.dtype(np.float32)
IMAGENET = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
PREPROCS = [
    pytest.param(CPUPreprocessor, {}, id="cpu"),
    pytest.param(CUDAPreprocessor, {}, id="cuda"),
    pytest.param(TRTPreprocessor, {"batch_size": 4}, id="trt"),
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


@pytest.mark.parametrize("norm", [{}, IMAGENET], ids=["raw", "imagenet"])
@pytest.mark.parametrize("count", [1, 3, 8])
def test_trt_runs_submitted_batch(random_images, count, norm) -> None:
    """TRTPreprocessor(batch_size=8) runs its engine at the image count."""
    trt_preproc = TRTPreprocessor(SIZE, RANGE, DTYPE, batch_size=8, **norm)
    imgs = random_images(count)
    batch, _, _ = trt_preproc.preprocess(imgs)
    assert trt_preproc._engine.active_input_shapes[0][0] == count
    singles = [trt_preproc.preprocess([img])[0] for img in imgs]
    np.testing.assert_array_equal(batch, np.concatenate(singles))


def test_trt_batch_size_exceeded_raises(random_images) -> None:
    """Submitting more images than the configured batch size raises ValueError."""
    trt_preproc = TRTPreprocessor(SIZE, RANGE, DTYPE, batch_size=4)
    with pytest.raises(ValueError, match="exceeds configured batch size"):
        trt_preproc.preprocess(random_images(5))


@pytest.mark.parametrize(("preproc_cls", "kwargs"), PREPROCS)
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
    "widths",
    [[2560], [2560] * 4, [2560, 1920, 1280, 2560]],
    ids=["single", "homogeneous", "heterogeneous"],
)
def test_strided_views_match_contiguous(gpu_cls, memory, widths) -> None:
    """Non-contiguous views, alone or in any batch, preprocess like their contiguous copies."""
    rng = np.random.default_rng(0)
    views = [rng.integers(0, 256, (720, w, 3), dtype=np.uint8)[:, ::2] for w in widths]
    kwargs = {"batch_size": 4} if gpu_cls is TRTPreprocessor else {}
    preproc = gpu_cls(SIZE, RANGE, DTYPE, **memory, **kwargs)
    expected = preproc.preprocess([np.ascontiguousarray(v) for v in views])
    result = preproc.preprocess(views)
    np.testing.assert_array_equal(result[0], expected[0])


BUFFER_PLACEMENTS = [
    pytest.param({"location": MemoryLocation.DEVICE}, id="device"),
    pytest.param({"pinned": True, "mapped": True}, id="mapped"),
    pytest.param({"pinned": True, "mapped": False}, id="pinned"),
    pytest.param({"pinned": False}, id="pageable"),
]


@pytest.mark.parametrize(("preproc_cls", "kwargs"), PREPROCS)
@pytest.mark.parametrize("placement", BUFFER_PLACEMENTS)
@pytest.mark.parametrize(
    "shapes",
    [
        pytest.param(SHAPES[:1], id="single"),
        pytest.param(SHAPES[:1] * 4, id="homogeneous"),
        pytest.param(SHAPES[:4], id="heterogeneous"),
    ],
)
def test_buffer_input_matches_ndarray(preproc_cls, kwargs, placement, shapes) -> None:
    """Buffer images, alone or mixed with arrays, preprocess exactly like the arrays."""
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 256, (*s, 3), dtype=np.uint8) for s in shapes]
    mixed = [
        Buffer.from_array(img, **placement) if i % 2 == 0 else img for i, img in enumerate(imgs)
    ]
    preproc = preproc_cls(SIZE, RANGE, DTYPE, **kwargs)
    expected = preproc.preprocess(imgs)
    result = preproc.preprocess(mixed)
    np.testing.assert_array_equal(result[0], expected[0])
    assert result[1:] == expected[1:]
    single = preproc.preprocess(mixed[0])
    np.testing.assert_array_equal(single[0], expected[0][:1])


@pytest.mark.parametrize(("gpu_cls", "kwargs"), PREPROCS[1:])
@pytest.mark.parametrize("position", [0, 1], ids=["first", "second"])
@pytest.mark.parametrize("as_buffer", [False, True], ids=["ndarray", "device"])
def test_gpu_rejects_non_uint8(gpu_cls, kwargs, position, as_buffer) -> None:
    """GPU preprocessors raise on non-uint8 images instead of reading their bytes as uint8."""
    image = np.zeros((480, 640, 3), dtype=np.float32)
    if as_buffer:
        image = Buffer.from_array(image, MemoryLocation.DEVICE)
    preproc = gpu_cls(SIZE, RANGE, DTYPE, **kwargs)
    with pytest.raises(ValueError, match="uint8"):
        preproc.preprocess([image] if position == 0 else [np.zeros((480, 640, 3), np.uint8), image])


def _reference_preprocess(images, input_shape, dtype, input_range, method, mean, std):
    """Run the previous per-image float64 pipeline, stacked with concatenate."""
    tensors = []
    for image in images:
        if method == "letterbox":
            tensor, _, _ = letterbox(image, new_shape=input_shape)
        else:
            tensor, _ = resize_linear(image, new_shape=input_shape)
        tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
        tensor = (tensor / 255.0 - mean) / std if mean is not None else rescale(tensor, input_range)
        tensors.append(np.transpose(tensor[np.newaxis], (0, 3, 1, 2)).astype(dtype))
    return np.concatenate(tensors, axis=0)


@pytest.mark.cpu
@pytest.mark.parametrize("method", ["letterbox", "linear"])
@pytest.mark.parametrize("shape", [(640, 640), (640, 384)], ids=["square", "wide"])
@pytest.mark.parametrize(
    ("input_range", "norm", "dtype", "exact"),
    [
        pytest.param((0.0, 1.0), {}, np.float32, False, id="scale01"),
        pytest.param((-1.0, 1.0), {}, np.float32, False, id="scale-1-1"),
        pytest.param((0.0, 1.0), IMAGENET, np.float32, False, id="imagenet"),
        pytest.param((0.0, 255.0), {}, np.uint8, True, id="uint8-identity"),
        pytest.param((0.0, 1.0), {}, np.float16, False, id="fp16"),
    ],
)
def test_preprocess_matches_reference(method, shape, input_range, norm, dtype, exact) -> None:
    """The single-pass pack matches the previous float64 pipeline."""
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 256, s, dtype=np.uint8) for s in [(480, 640, 3), (137, 251, 3)]]
    mean, std = norm.get("mean"), norm.get("std")
    expected = _reference_preprocess(imgs, shape, dtype, input_range, method, mean, std)
    result, _, _ = preprocess(imgs, shape, np.dtype(dtype), input_range, method, mean, std)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    if exact:
        np.testing.assert_array_equal(result, expected)
    else:
        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-5)


@pytest.mark.cpu
def test_preprocess_writes_into_matching_out(random_images) -> None:
    """Out is used only when its shape and dtype match the batch."""
    imgs = random_images(2)
    matching = np.empty((2, 3, 640, 640), dtype=np.float32)
    assert preprocess(imgs, SIZE, DTYPE, RANGE, out=matching)[0] is matching
    for wrong in (np.empty((3, 3, 640, 640), np.float32), np.empty((2, 3, 640, 640), np.float16)):
        assert preprocess(imgs, SIZE, DTYPE, RANGE, out=wrong)[0] is not wrong


@pytest.mark.cpu
@pytest.mark.parametrize("no_copy", [True, False])
def test_cpu_no_copy_reuses_buffer(random_images, no_copy) -> None:
    """no_copy returns a reused view the next call overwrites, else a private array."""
    cpu = CPUPreprocessor(SIZE, RANGE, DTYPE)
    first, _, _ = cpu.preprocess(random_images(4), no_copy=no_copy)
    snapshot = first.copy()
    second, _, _ = cpu.preprocess(random_images(2), no_copy=no_copy)
    assert np.shares_memory(first, second) == no_copy
    assert np.array_equal(first, snapshot) != no_copy
