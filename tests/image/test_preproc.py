# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/preprocessors/ -- CPU, CUDA, TRT, and hybrid preprocessors."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from trtutils import Buffer, MemoryLocation
from trtutils.core import destroy_stream
from trtutils.image.preprocessors import (
    CPUPreprocessor,
    CUDAPreprocessor,
    HybridPreprocessor,
    TRTPreprocessor,
    _cuda,
    _trt,
)
from trtutils.image.preprocessors._base import crop_geometry
from trtutils.image.preprocessors._cpu import letterbox, preprocess, resize_linear

SIZE = (640, 640)
RANGE = (0.0, 1.0)
DTYPE = np.dtype(np.float32)
IMAGENET = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
PREPROCS = [
    pytest.param(CPUPreprocessor, {}, id="cpu"),
    pytest.param(CUDAPreprocessor, {}, id="cuda"),
    pytest.param(TRTPreprocessor, {"batch_size": 4}, id="trt"),
]
BUFFER_PLACEMENTS = [
    pytest.param({"location": MemoryLocation.DEVICE}, id="device"),
    pytest.param({"pinned": True, "mapped": True}, id="mapped"),
    pytest.param({"pinned": True, "mapped": False}, id="pinned"),
    pytest.param({"pinned": False}, id="pageable"),
]
HYBRID = pytest.param(HybridPreprocessor, {}, id="hybrid")


@pytest.mark.parametrize("resize", ["linear", "letterbox", "center_crop"])
@pytest.mark.parametrize(
    ("norm", "atol"),
    [
        pytest.param({}, 0.0, id="raw"),
        pytest.param(IMAGENET, 1e-6, id="imagenet"),
    ],
)
def test_hybrid_matches_cpu(images, resize, norm, atol) -> None:
    """Hybrid resizes exactly as the CPU path; only the float32 normalization order differs."""
    cpu = CPUPreprocessor(SIZE, RANGE, DTYPE, resize=resize, **norm)
    hybrid = HybridPreprocessor(SIZE, RANGE, DTYPE, resize=resize, **norm)
    batch = [image.array for image in images.values()]
    for imgs in (batch, batch[:1], batch):
        expected, ratios, padding = cpu.preprocess(imgs)
        result, hybrid_ratios, hybrid_padding = hybrid.preprocess(imgs)
        assert hybrid_ratios == ratios
        assert hybrid_padding == padding
        np.testing.assert_allclose(result, expected, rtol=0, atol=atol)


@pytest.mark.parametrize("placement", BUFFER_PLACEMENTS[:2])
def test_hybrid_device_buffers_resize_on_gpu(images, placement) -> None:
    """Device-visible Buffers skip the CPU resize and give the CUDA preprocessor's result."""
    batch = [Buffer.from_array(image.array, **placement) for image in images.values()]
    expected = CUDAPreprocessor(SIZE, RANGE, DTYPE).preprocess(batch)
    result = HybridPreprocessor(SIZE, RANGE, DTYPE).preprocess(batch)
    np.testing.assert_array_equal(result[0], expected[0])
    assert result[1:] == expected[1:]


@pytest.mark.parametrize(
    "gpu_cls",
    [
        pytest.param(CUDAPreprocessor, id="cuda"),
        pytest.param(TRTPreprocessor, id="trt"),
    ],
)
@pytest.mark.parametrize("resize", ["linear", "letterbox", "center_crop"])
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
    "gpu_cls",
    [
        pytest.param(CUDAPreprocessor, id="cuda"),
        pytest.param(TRTPreprocessor, id="trt"),
    ],
)
@pytest.mark.parametrize("resize", ["linear", "letterbox", "center_crop"])
@pytest.mark.parametrize("out_size", [640, 1024])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1280, 1920), id="1920x1280"),
        pytest.param((3000, 4000), id="4000x3000"),
        pytest.param((1999, 3001), id="3001x1999"),
    ],
)
def test_gpu_large_downscale_matches_cpu(gpu_cls, resize, out_size, shape) -> None:
    """Large downscales with fractional letterbox sizes match cv2 pixels and padding."""
    rng = np.random.default_rng(0)
    image = cv2.GaussianBlur(rng.integers(0, 255, (*shape, 3), dtype=np.uint8), (0, 0), 1.5)
    size = (out_size, out_size)
    cpu = CPUPreprocessor(size, RANGE, DTYPE, resize=resize)
    gpu = gpu_cls(size, RANGE, DTYPE, resize=resize)
    expected, ratios, padding = cpu.preprocess([image])
    result, gpu_ratios, gpu_padding = gpu.preprocess([image])
    assert gpu_ratios == ratios
    assert gpu_padding == padding
    assert np.abs(result - expected).mean() < 0.002


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


@pytest.mark.parametrize("resize", ["letterbox", "linear"])
def test_preprocess_without_normalize_returns_uint8_hwc(images, resize) -> None:
    """normalize=False preprocess returns the resized uint8 image, not the unused SST buffer."""
    image = images["horse"].array
    if resize == "letterbox":
        expected, _, _ = letterbox(image, new_shape=SIZE)
    else:
        expected, _ = resize_linear(image, new_shape=SIZE)
    gpu = CUDAPreprocessor(SIZE, RANGE, DTYPE, resize=resize, normalize=False)
    result, _, _ = gpu.preprocess(image)
    assert result.dtype == np.uint8
    assert result.shape == (1, *expected.shape)
    assert np.abs(result[0].astype(np.int32) - expected.astype(np.int32)).mean() < 1.0


def test_trt_batch_size_exceeded_raises(random_images) -> None:
    """Submitting more images than the configured batch size raises ValueError."""
    trt_preproc = TRTPreprocessor(SIZE, RANGE, DTYPE, batch_size=4)
    with pytest.raises(ValueError, match="exceeds configured batch size"):
        trt_preproc.preprocess(random_images(5))


@pytest.mark.parametrize(("preproc_cls", "kwargs"), [*PREPROCS, HYBRID])
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
        # chunked pinned uploads, the second into a larger stage
        pytest.param([[(1280, 1920)], [(1081, 1919)]], id="large-staged"),
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


@pytest.mark.regression
@pytest.mark.parametrize(
    ("gpu_cls", "module"),
    [
        pytest.param(CUDAPreprocessor, _cuda, id="cuda"),
        pytest.param(TRTPreprocessor, _trt, id="trt"),
    ],
)
def test_gpu_destroys_own_stream_once(gpu_cls, module, monkeypatch) -> None:
    """Repeated __del__ calls destroy the stream the preprocessor created exactly once."""
    destroyed: list[object] = []
    monkeypatch.setattr(module, "destroy_stream", destroyed.append)
    preproc = gpu_cls(SIZE, RANGE, DTYPE)
    stream = preproc._stream
    preproc.__del__()
    preproc.__del__()
    del preproc
    assert destroyed == [stream]
    monkeypatch.undo()
    destroy_stream(stream)


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


@pytest.mark.parametrize(("gpu_cls", "kwargs"), [*PREPROCS[1:], HYBRID])
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
        tensor = (
            (tensor / 255.0 - mean) / std
            if mean is not None
            else tensor / 255.0 * (input_range[1] - input_range[0]) + input_range[0]
        )
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
@pytest.mark.parametrize(
    ("image_hw", "output_shape", "crop_pct", "geometry"),
    [
        pytest.param((480, 640), (224, 224), 0.875, (341, 256, 58, 16), id="landscape"),
        pytest.param((640, 480), (224, 224), 0.875, (256, 341, 16, 58), id="portrait"),
        pytest.param((375, 500), (224, 224), 0.9, (330, 248, 53, 12), id="crop-0.9"),
        pytest.param((480, 640), (224, 224), 1.0, (298, 224, 37, 0), id="no-crop"),
        pytest.param((256, 256), (224, 224), 0.875, (256, 256, 16, 16), id="identity-scale"),
        pytest.param((480, 640), (320, 160), 1.0, (320, 240, 0, 40), id="wide-output"),
    ],
)
def test_crop_geometry(image_hw, output_shape, crop_pct, geometry) -> None:
    """Resize to cover floor(output / crop_pct), then a rounded center crop, as torchvision."""
    height, width = image_hw
    result, ratios, padding = crop_geometry(width, height, output_shape, crop_pct)
    assert result == geometry
    resized_w, resized_h, left, top = geometry
    # ratios are the resize scale and padding the negative crop offset
    assert ratios == (resized_w / width, resized_h / height)
    assert padding == (-left, -top)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("image_hw", "crop_pct"),
    [
        pytest.param((480, 640), 0.875, id="landscape"),
        pytest.param((640, 480), 0.875, id="portrait"),
        pytest.param((375, 500), 0.9, id="crop-0.9"),
        pytest.param((480, 640), 1.0, id="no-crop"),
    ],
)
def test_center_crop_matches_resize_then_crop(image_hw, crop_pct) -> None:
    """center_crop equals torchvision Resize(floor(imgsz / crop_pct)) then CenterCrop(imgsz)."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (*image_hw, 3), dtype=np.uint8)
    imgsz = 224
    height, width = image_hw
    short = math.floor(imgsz / crop_pct)
    if height <= width:
        size = (int(short * width / height), short)
    else:
        size = (short, int(short * height / width))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    top = round((resized.shape[0] - imgsz) / 2.0)
    left = round((resized.shape[1] - imgsz) / 2.0)
    expected = cv2.cvtColor(resized[top : top + imgsz, left : left + imgsz], cv2.COLOR_BGR2RGB)

    result, _, _ = preprocess(
        [image], (imgsz, imgsz), np.dtype(np.uint8), (0.0, 255.0), "center_crop", crop_pct=crop_pct
    )
    np.testing.assert_array_equal(result[0].transpose(1, 2, 0), expected)


@pytest.mark.cpu
@pytest.mark.parametrize("crop_pct", [0.0, -0.5, 1.5])
def test_invalid_crop_pct_raises(random_images, crop_pct) -> None:
    """crop_pct outside (0, 1] raises for the preprocess function and CPUPreprocessor."""
    with pytest.raises(ValueError, match="crop_pct"):
        preprocess(random_images(1), SIZE, DTYPE, RANGE, "center_crop", crop_pct=crop_pct)
    with pytest.raises(ValueError, match="crop_pct"):
        CPUPreprocessor(SIZE, RANGE, DTYPE, resize="center_crop", crop_pct=crop_pct)


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


def _half_size_fn(images, *, verbose=None):
    """Halve each image and zero-pad the batch bottom/right to the largest one."""
    resized = [cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)) for img in images]
    height = max(r.shape[0] for r in resized)
    width = max(r.shape[1] for r in resized)
    batch = np.zeros((len(resized), 3, height, width), dtype=np.float32)
    for i, r in enumerate(resized):
        batch[i, :, : r.shape[0], : r.shape[1]] = r.transpose(2, 0, 1) / 255.0
    ratios = [
        (r.shape[1] / img.shape[1], r.shape[0] / img.shape[0]) for r, img in zip(resized, images)
    ]
    padding = [(float(width - r.shape[1]), float(height - r.shape[0])) for r in resized]
    return batch, ratios, padding


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("placement", [None, *BUFFER_PLACEMENTS[1:]])
@pytest.mark.parametrize(
    "shapes",
    [
        pytest.param([(480, 640)], id="single"),
        pytest.param([(480, 640), (480, 640)], id="homogeneous"),
        pytest.param([(480, 640), (720, 320), (101, 99)], id="mixed"),
    ],
)
def test_cpu_preprocess_fn(dtype, placement, shapes) -> None:
    """A resize function preprocesses each batch at its own shape, cast to the dtype."""
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 256, (*s, 3), dtype=np.uint8) for s in shapes]
    inputs = [Buffer.from_array(img, **placement) for img in imgs] if placement else imgs
    cpu = CPUPreprocessor(SIZE, RANGE, np.dtype(dtype), resize=_half_size_fn)
    assert cpu.dynamic
    assert not CPUPreprocessor(SIZE, RANGE, DTYPE).dynamic
    cpu.warmup()
    expected, ratios, padding = _half_size_fn(imgs)
    for result in (cpu.preprocess(inputs), cpu(inputs)):
        assert result[0].dtype == dtype
        np.testing.assert_array_equal(result[0], expected.astype(dtype))
        assert result[1:] == (ratios, padding)
    single = cpu.preprocess(inputs[0])
    np.testing.assert_array_equal(single[0], _half_size_fn(imgs[:1])[0].astype(dtype))


@pytest.mark.cpu
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"resize": "linear"}, id="resize"),
        pytest.param({"no_copy": True}, id="no-copy"),
    ],
)
def test_cpu_preprocess_fn_rejects_fixed_options(kwargs) -> None:
    """Per-call resize and no_copy raise when a resize function is set."""
    cpu = CPUPreprocessor(SIZE, RANGE, DTYPE, resize=_half_size_fn)
    with pytest.raises(ValueError, match="not supported with a preprocessing function"):
        cpu.preprocess(np.zeros((64, 64, 3), np.uint8), **kwargs)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("image_hw", "new_shape", "ratio", "padding", "inner"),
    [
        pytest.param((100, 200), (640, 640), 3.2, (0.0, 160.0), (320, 640), id="wide-to-square"),
        pytest.param((200, 100), (640, 640), 3.2, (160.0, 0.0), (640, 320), id="tall-to-square"),
        pytest.param((50, 50), (100, 100), 2.0, (0.0, 0.0), (100, 100), id="exact-no-pad"),
        pytest.param((101, 200), (200, 200), 1.0, (0.0, 49.5), (101, 200), id="odd-pad-split"),
    ],
)
def test_letterbox(image_hw, new_shape, ratio, padding, inner) -> None:
    """Letterbox keeps the aspect ratio, pads with 114 and reports ratio and padding."""
    image = np.full((*image_hw, 3), 7, dtype=np.uint8)
    out, ratios, pads = letterbox(image, new_shape)
    assert out.shape == (new_shape[1], new_shape[0], 3)
    assert ratios == (ratio, ratio)
    assert pads == padding
    top = round(padding[1] - 0.1)
    left = round(padding[0] - 0.1)
    assert np.all(out[top : top + inner[0], left : left + inner[1]] == 7)
    assert (out == 114).sum() == out.size - inner[0] * inner[1] * 3


@pytest.mark.cpu
def test_resize_linear() -> None:
    """resize_linear matches cv2.resize and returns (new_h / w, new_w / h), as cv2ext did."""
    image = np.random.default_rng(0).integers(0, 255, (30, 60, 3), dtype=np.uint8)
    out, ratios = resize_linear(image, (120, 90))
    assert np.array_equal(out, cv2.resize(image, (120, 90), interpolation=cv2.INTER_LINEAR))
    assert ratios == (1.5, 4.0)
