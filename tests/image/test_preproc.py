# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/preprocessors/ -- CPU resize functions and CUDA/TRT internals."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from trtutils import Buffer, MemoryLocation
from trtutils.core import destroy_stream
from trtutils.image.postprocessors import postprocess_yolov10
from trtutils.image.preprocessors import (
    CPUPreprocessor,
    CUDAPreprocessor,
    TRTPreprocessor,
    _cuda,
    _trt,
)
from trtutils.image.preprocessors._base import RESIZE_METHODS, crop_geometry
from trtutils.image.preprocessors._cpu import letterbox, preprocess, resize_image, resize_linear

SIZE = (640, 640)
NON_SQUARE = [
    pytest.param((640, 384), id="640x384"),
    pytest.param((384, 640), id="384x640"),
]
RANGE = (0.0, 1.0)
DTYPE = np.dtype(np.float32)
IMAGENET = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
BUFFER_PLACEMENTS = [
    pytest.param({"location": MemoryLocation.DEVICE}, id="device"),
    pytest.param({"pinned": True, "mapped": True}, id="mapped"),
    pytest.param({"pinned": True, "mapped": False}, id="pinned"),
    pytest.param({"pinned": False}, id="pageable"),
]


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


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(np.dtype(np.uint8), id="uint8"),
        pytest.param(np.dtype(np.int8), id="int8"),
        pytest.param(np.dtype(np.float64), id="float64"),
    ],
)
def test_cuda_rejects_unsupported_dtype(dtype) -> None:
    """Normalizing to a dtype without a CUDA kernel raises; resize-only accepts it."""
    with pytest.raises(ValueError, match="float32 or float16"):
        CUDAPreprocessor(SIZE, RANGE, dtype)
    CUDAPreprocessor(SIZE, RANGE, dtype, normalize=False)


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
def test_resize_image_rejects_unknown_method() -> None:
    """resize_image raises for an unknown method instead of resizing linearly."""
    with pytest.raises(ValueError, match="Unknown method"):
        resize_image(np.zeros((480, 640, 3), dtype=np.uint8), SIZE, "area")


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
    """resize_linear matches cv2.resize and returns (new_w / w, new_h / h)."""
    image = np.random.default_rng(0).integers(0, 255, (30, 60, 3), dtype=np.uint8)
    out, ratios = resize_linear(image, (120, 90))
    assert np.array_equal(out, cv2.resize(image, (120, 90), interpolation=cv2.INTER_LINEAR))
    assert ratios == (2.0, 3.0)


@pytest.mark.cpu
@pytest.mark.parametrize("method", RESIZE_METHODS)
@pytest.mark.parametrize("size", NON_SQUARE)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1080, 1920), id="1920x1080"),
        pytest.param((1280, 720), id="720x1280"),
    ],
)
def test_preprocess_postprocess_box_roundtrip(method, size, shape) -> None:
    """A box found in the preprocessed image postprocesses back to its original-image box."""
    height, width = shape
    box = (np.array([0.4, 0.45, 0.6, 0.55]) * (width, height, width, height)).astype(int)
    image = np.zeros((*shape, 3), dtype=np.uint8)
    image[box[1] : box[3], box[0] : box[2]] = 255
    tensor, ratios, padding = preprocess([image], size, DTYPE, method=method)
    ys, xs = np.nonzero(tensor[0, 0] > 0.5)
    output = np.array([[[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.9, 0]]], np.float32)
    bboxes = postprocess_yolov10([output], ratios, padding, input_size=size)[0][0]
    # within one model-input pixel
    np.testing.assert_allclose(bboxes[0], box, atol=1 / min(ratios[0]))
