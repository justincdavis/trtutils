# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/preprocessors/ -- CPU, CUDA, and TRT preprocessors."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from cv2ext.image import letterbox, rescale, resize_linear

from trtutils.core import Buffer, MemoryLocation
from trtutils.image.preprocessors import CPUPreprocessor, CUDAPreprocessor, TRTPreprocessor
from trtutils.image.preprocessors._process import preprocess

SIZE = (640, 640)
RANGE = (0.0, 1.0)
DTYPE = np.dtype(np.float32)
IMAGENET = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
# 8 * 3 * 640 * 640 * 4 bytes == ~37.5 MB, above CPUPreprocessor's 16 MB reuse
# threshold, so it forces the grow-only batch buffer path.
_REUSE_BATCH_SIZE = 8


def _old_preprocess_single(
    image: np.ndarray,
    input_shape: tuple[int, int],
    dtype: np.dtype,
    input_range: tuple[float, float],
    method: str,
    mean: tuple[float, float, float] | None,
    std: tuple[float, float, float] | None,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Pre-PR14 per-image pipeline: float64 arithmetic, cast once at the end."""
    if method == "letterbox":
        tensor, ratios, padding = letterbox(image, new_shape=input_shape)
    else:
        tensor, ratios = resize_linear(image, new_shape=input_shape)
        padding = (0.0, 0.0)
    tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
    if mean is not None and std is not None:
        tensor = tensor / 255.0
        tensor = (tensor - mean) / std
    else:
        tensor = rescale(tensor, input_range)
    tensor = tensor[np.newaxis, :]
    tensor = np.transpose(tensor, (0, 3, 1, 2))
    if not tensor.flags["C_CONTIGUOUS"]:
        tensor = np.ascontiguousarray(tensor)
    tensor = tensor.astype(dtype)
    return tensor, ratios, padding


def _old_preprocess(
    images: list[np.ndarray],
    input_shape: tuple[int, int],
    dtype: np.dtype,
    input_range: tuple[float, float],
    method: str,
    mean: tuple[float, float, float] | None,
    std: tuple[float, float, float] | None,
) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
    """Pre-PR14 batch pipeline: per-image tensors stacked with concatenate."""
    tensors: list[np.ndarray] = []
    ratios_list: list[tuple[float, float]] = []
    padding_list: list[tuple[float, float]] = []
    for image in images:
        tensor, ratios, padding = _old_preprocess_single(
            image, input_shape, dtype, input_range, method, mean, std
        )
        tensors.append(tensor)
        ratios_list.append(ratios)
        padding_list.append(padding)
    return np.concatenate(tensors, axis=0), ratios_list, padding_list


GPU_CLASSES = [
    pytest.param(CUDAPreprocessor, id="cuda"),
    pytest.param(TRTPreprocessor, id="trt"),
]

# (preprocessor class, extra constructor kwargs) for all three backends,
# used by the Buffer-input tests below.
ALL_PREPROC_CLASSES = [
    pytest.param(CPUPreprocessor, {}, id="cpu"),
    pytest.param(CUDAPreprocessor, {}, id="cuda"),
    pytest.param(TRTPreprocessor, {"batch_size": 3}, id="trt"),
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


@pytest.mark.parametrize("count", [1, 3, 8])
def test_trt_dynamic_batch_matches_cuda_and_self(random_images, count) -> None:
    """
    TRTPreprocessor(batch_size=8) at 1/3/8 images matches itself and CUDAPreprocessor at batch 1.

    Runs the preprocessing engine at the submitted batch (not the configured
    max), so batch position i must reproduce the single-image TRT output
    bit-for-bit, and stay within the CUDA-vs-TRT tolerance from
    test_gpu_matches_cpu (the TRT engine computes in fp16 internally).
    """
    trt_preproc = TRTPreprocessor(SIZE, RANGE, DTYPE, batch_size=8)
    cuda_preproc = CUDAPreprocessor(SIZE, RANGE, DTYPE)
    imgs = random_images(count)
    batch, batch_ratios, batch_padding = trt_preproc.preprocess(imgs)
    assert batch.shape == (count, 3, 640, 640)
    for i, img in enumerate(imgs):
        single_trt, trt_ratios, trt_padding = trt_preproc.preprocess([img])
        single_cuda, cuda_ratios, cuda_padding = cuda_preproc.preprocess([img])
        assert batch_ratios[i] == trt_ratios[0] == cuda_ratios[0]
        assert batch_padding[i] == trt_padding[0] == cuda_padding[0]
        np.testing.assert_array_equal(batch[i], single_trt[0])
        assert np.abs(batch[i] - single_cuda[0]).mean() < 0.02


def test_trt_batch_size_exceeded_raises(random_images) -> None:
    """Submitting more images than the configured batch size raises ValueError."""
    trt_preproc = TRTPreprocessor(SIZE, RANGE, DTYPE, batch_size=4)
    imgs = random_images(5)
    with pytest.raises(ValueError, match="exceeds configured batch size"):
        trt_preproc.preprocess(imgs)


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


@pytest.fixture
def preproc_images(images) -> list[np.ndarray]:
    """Horse, people, and two odd-sized random images (seeded for reproducibility)."""
    rng = np.random.default_rng(0)
    odd1 = rng.integers(0, 255, (137, 251, 3), dtype=np.uint8)
    odd2 = rng.integers(0, 255, (400, 89, 3), dtype=np.uint8)
    return [images["horse"].array, images["people"].array, odd1, odd2]


# (method, input_range, mean, std, dtype, exact)
# "exact" cases are bit-identical to the pre-PR float64 pipeline: the
# identity range takes a pure-transpose fast path, and uint8/float16 outputs
# quantize away the float32-arithmetic rounding noise. The float32-output
# scale and mean/std cases are algebraically identical but not bit-identical:
# the fused implementation does the affine transform in float32 throughout,
# while the old pipeline computed it in float64 and rounded once at the end.
# Values near zero (e.g. after mean/std centering) make a ULP-based bound
# meaningless -- a tiny absolute difference is a huge ULP count there -- so
# those cases are checked with a tight absolute/relative tolerance instead.
_PREPROC_CASES = [
    pytest.param("letterbox", (0.0, 1.0), None, None, np.float32, False, id="letterbox-scale01"),
    pytest.param("letterbox", (-1.0, 1.0), None, None, np.float32, False, id="letterbox-scale-1-1"),
    pytest.param(
        "letterbox",
        (0.0, 1.0),
        IMAGENET["mean"],
        IMAGENET["std"],
        np.float32,
        False,
        id="letterbox-imagenet",
    ),
    pytest.param(
        "letterbox", (0.0, 255.0), None, None, np.uint8, True, id="letterbox-uint8-identity"
    ),
    pytest.param("letterbox", (0.0, 1.0), None, None, np.float16, True, id="letterbox-fp16"),
    pytest.param("linear", (0.0, 1.0), None, None, np.float32, False, id="linear-scale01"),
    pytest.param("linear", (-1.0, 1.0), None, None, np.float32, False, id="linear-scale-1-1"),
    pytest.param(
        "linear",
        (0.0, 1.0),
        IMAGENET["mean"],
        IMAGENET["std"],
        np.float32,
        False,
        id="linear-imagenet",
    ),
    pytest.param("linear", (0.0, 255.0), None, None, np.uint8, True, id="linear-uint8-identity"),
    pytest.param("linear", (0.0, 1.0), None, None, np.float16, True, id="linear-fp16"),
]


@pytest.mark.cpu
@pytest.mark.parametrize(("method", "input_range", "mean", "std", "dtype", "exact"), _PREPROC_CASES)
def test_preprocess_matches_pre_pr_algorithm(
    preproc_images, method, input_range, mean, std, dtype, exact
) -> None:
    """preprocess() output matches the pre-PR per-image float64 pipeline."""
    old_tensor, old_ratios, old_padding = _old_preprocess(
        preproc_images, SIZE, np.dtype(dtype), input_range, method, mean, std
    )
    new_tensor, new_ratios, new_padding = preprocess(
        preproc_images, SIZE, np.dtype(dtype), input_range, method, mean, std
    )
    assert new_ratios == old_ratios
    assert new_padding == old_padding
    if exact:
        np.testing.assert_array_equal(new_tensor, old_tensor)
    else:
        np.testing.assert_allclose(new_tensor, old_tensor, rtol=1e-5, atol=1e-6)


@pytest.mark.cpu
def test_preprocess_empty_list_raises() -> None:
    """An empty image list raises ValueError instead of failing inside np.concatenate."""
    with pytest.raises(ValueError, match="No images"):
        preprocess([], SIZE, DTYPE, RANGE, "letterbox")


@pytest.mark.cpu
def test_preprocess_out_used_when_shape_and_dtype_match(random_images) -> None:
    """out= is written in place when its shape and dtype match, otherwise ignored."""
    imgs = random_images(2)
    matching = np.empty((2, 3, 640, 640), dtype=np.float32)
    result, _, _ = preprocess(imgs, SIZE, DTYPE, RANGE, "letterbox", out=matching)
    assert result is matching

    wrong_shape = np.empty((3, 3, 640, 640), dtype=np.float32)
    result, _, _ = preprocess(imgs, SIZE, DTYPE, RANGE, "letterbox", out=wrong_shape)
    assert result is not wrong_shape

    wrong_dtype = np.empty((2, 3, 640, 640), dtype=np.float16)
    result, _, _ = preprocess(imgs, SIZE, DTYPE, RANGE, "letterbox", out=wrong_dtype)
    assert result is not wrong_dtype


@pytest.mark.cpu
def test_cpu_preprocessor_matches_preprocess_function(preproc_images) -> None:
    """CPUPreprocessor produces the same output as calling preprocess() directly."""
    cpu = CPUPreprocessor(SIZE, RANGE, DTYPE, resize="letterbox")
    cpu_tensor, cpu_ratios, cpu_padding = cpu.preprocess(preproc_images)
    direct_tensor, direct_ratios, direct_padding = preprocess(
        preproc_images, SIZE, DTYPE, RANGE, "letterbox"
    )
    np.testing.assert_array_equal(cpu_tensor, direct_tensor)
    assert cpu_ratios == direct_ratios
    assert cpu_padding == direct_padding


@pytest.mark.cpu
def test_cpu_preprocessor_no_copy_view_overwritten_by_next_call(random_images) -> None:
    """no_copy=True above the reuse threshold returns a view the next call overwrites."""
    cpu = CPUPreprocessor(SIZE, RANGE, DTYPE)
    imgs_a = random_images(_REUSE_BATCH_SIZE)
    imgs_b = random_images(_REUSE_BATCH_SIZE)

    view, _, _ = cpu.preprocess(imgs_a, no_copy=True)
    assert view.nbytes > 16 * 1024 * 1024
    snapshot = view.copy()

    cpu.preprocess(imgs_b, no_copy=True)

    assert not np.array_equal(view, snapshot)


@pytest.mark.cpu
def test_cpu_preprocessor_copy_survives_next_call(random_images) -> None:
    """no_copy=False (the default) returns a private array unaffected by later calls."""
    cpu = CPUPreprocessor(SIZE, RANGE, DTYPE)
    imgs_a = random_images(_REUSE_BATCH_SIZE)
    imgs_b = random_images(_REUSE_BATCH_SIZE)

    result, _, _ = cpu.preprocess(imgs_a, no_copy=False)
    assert result.nbytes > 16 * 1024 * 1024
    snapshot = result.copy()

    cpu.preprocess(imgs_b, no_copy=False)

    np.testing.assert_array_equal(result, snapshot)


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


# ----------------------------------------------------------------------
# Buffer inputs (PR 9)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(("preproc_cls", "kwargs"), ALL_PREPROC_CLASSES)
def test_preprocess_device_buffer_matches_ndarray(random_images, preproc_cls, kwargs) -> None:
    """preprocess([device Buffer]) matches preprocess([ndarray]) exactly."""
    preproc = preproc_cls(SIZE, RANGE, DTYPE, **kwargs)
    img = random_images(1)[0]
    expected, exp_ratios, exp_padding = preproc.preprocess([img])

    buf = Buffer.from_array(img, MemoryLocation.DEVICE)
    try:
        result, ratios, padding = preproc.preprocess([buf])
    finally:
        buf.free()
    np.testing.assert_array_equal(result, expected)
    assert ratios == exp_ratios
    assert padding == exp_padding


@pytest.mark.parametrize(("preproc_cls", "kwargs"), ALL_PREPROC_CLASSES)
def test_preprocess_host_buffer_matches_ndarray(random_images, preproc_cls, kwargs) -> None:
    """preprocess([host Buffer]) matches preprocess([ndarray]) exactly."""
    preproc = preproc_cls(SIZE, RANGE, DTYPE, **kwargs)
    img = random_images(1)[0]
    expected, exp_ratios, exp_padding = preproc.preprocess([img])

    buf = Buffer.from_array(img, MemoryLocation.HOST)
    try:
        result, ratios, padding = preproc.preprocess([buf])
    finally:
        buf.free()
    np.testing.assert_array_equal(result, expected)
    assert ratios == exp_ratios
    assert padding == exp_padding


@pytest.mark.parametrize(("preproc_cls", "kwargs"), ALL_PREPROC_CLASSES)
def test_preprocess_single_buffer_not_wrapped_in_list(random_images, preproc_cls, kwargs) -> None:
    """A single Buffer (not wrapped in a list) is treated as one image, like a bare ndarray."""
    preproc = preproc_cls(SIZE, RANGE, DTYPE, **kwargs)
    img = random_images(1)[0]
    expected, exp_ratios, exp_padding = preproc.preprocess(img)

    buf = Buffer.from_array(img, MemoryLocation.DEVICE)
    try:
        result, ratios, padding = preproc.preprocess(buf)
    finally:
        buf.free()
    np.testing.assert_array_equal(result, expected)
    assert ratios == exp_ratios
    assert padding == exp_padding


@pytest.mark.parametrize(("preproc_cls", "kwargs"), ALL_PREPROC_CLASSES)
def test_preprocess_mixed_batch_ndarray_host_device_buffers(
    random_images, preproc_cls, kwargs
) -> None:
    """A batch mixing ndarray / host Buffer / device Buffer of different sizes matches singles."""
    preproc = preproc_cls(SIZE, RANGE, DTYPE, **kwargs)
    sizes = [(480, 640), (720, 1280), (600, 800)]
    imgs = [random_images(1, h, w)[0] for h, w in sizes]
    singles = [preproc.preprocess([img]) for img in imgs]

    host_buf = Buffer.from_array(imgs[1], MemoryLocation.HOST)
    device_buf = Buffer.from_array(imgs[2], MemoryLocation.DEVICE)
    try:
        batch, ratios, padding = preproc.preprocess([imgs[0], host_buf, device_buf])
    finally:
        host_buf.free()
        device_buf.free()

    assert batch.shape[0] == 3
    for i in range(3):
        tensor, single_ratios, single_padding = singles[i]
        np.testing.assert_array_equal(batch[i], tensor[0])
        assert ratios[i] == single_ratios[0]
        assert padding[i] == single_padding[0]


@pytest.mark.parametrize(("preproc_cls", "kwargs"), ALL_PREPROC_CLASSES)
@pytest.mark.parametrize(
    ("bad_array", "match"),
    [
        pytest.param(
            np.zeros((4, 4), dtype=np.uint8),
            r"(?i)image must be|preprocess color",
            id="wrong-ndim",
        ),
        pytest.param(
            np.zeros((4, 4, 4), dtype=np.uint8),
            r"(?i)image must be|preprocess color",
            id="wrong-channels",
        ),
        pytest.param(np.zeros((4, 4, 3), dtype=np.float32), "uint8", id="wrong-dtype"),
    ],
)
def test_preprocess_bad_device_buffer_raises_value_error(
    preproc_cls, kwargs, bad_array, match
) -> None:
    """A device Buffer with wrong ndim/channels/dtype raises ValueError."""
    preproc = preproc_cls(SIZE, RANGE, DTYPE, **kwargs)
    buf = Buffer.from_array(bad_array, MemoryLocation.DEVICE)
    try:
        with pytest.raises(ValueError, match=match):
            preproc.preprocess([buf])
    finally:
        buf.free()


def test_preprocess_cuda_array_interface_device_buffer(random_images) -> None:
    """A from_cuda_array()-wrapped device Buffer works exactly like a plain device Buffer."""
    preproc = CUDAPreprocessor(SIZE, RANGE, DTYPE)
    img = random_images(1)[0]
    expected, exp_ratios, exp_padding = preproc.preprocess([img])

    owner = Buffer.from_array(img, MemoryLocation.DEVICE)
    view = Buffer.from_cuda_array(owner)
    try:
        result, ratios, padding = preproc.preprocess([view])
    finally:
        owner.free()

    np.testing.assert_array_equal(result, expected)
    assert ratios == exp_ratios
    assert padding == exp_padding
