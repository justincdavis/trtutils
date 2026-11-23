# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/preprocessors/_tent.py and _cpu.py -- the antialiased resize."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from trtutils._flags import FLAGS
from trtutils.image.preprocessors import CPUPreprocessor
from trtutils.image.preprocessors._base import RESIZE_METHODS, crop_geometry
from trtutils.image.preprocessors._cpu import resize, resize_image
from trtutils.image.preprocessors._tent import _tent_loops, _tent_numpy, tent_resize, tent_taps

# (height, width) -> (width, height), named by the downscale factor of each axis
DOWNSCALES = [
    pytest.param((107, 100), (100, 100), id="1.07x-height"),
    pytest.param((300, 321), (300, 300), id="1.07x-width"),
    pytest.param((90, 160), (80, 45), id="2x"),
    pytest.param((299, 401), (130, 97), id="3.1x"),
    pytest.param((720, 1280), (320, 180), id="4x"),
    pytest.param((1080, 1920), (256, 144), id="7.5x"),
    pytest.param((768, 512), (32, 48), id="16x"),
    pytest.param((200, 640), (40, 400), id="16x-width-2x-height-up"),
    pytest.param((50, 300), (100, 25), id="3x-width-2x-height"),
]
NOT_DOWNSCALED = [
    pytest.param((64, 48), (48, 64), id="1x"),
    pytest.param((45, 80), (160, 90), id="2x-up"),
    pytest.param((60, 61), (183, 60), id="3x-up-width-1x-height"),
]


def _image(height: int, width: int, seed: int = 0) -> np.ndarray:
    """A random image smoothed a little, so resizes see edges and gradients."""
    noise = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (3, 3), 0)


def _tent_matrix(in_size: int, out_size: int) -> np.ndarray:
    """The (out, in) tent weights, written from PIL's precompute_coeffs one pixel at a time."""
    scale = in_size / out_size
    support = max(scale, 1.0)
    matrix = np.zeros((out_size, in_size))
    for i in range(out_size):
        center = (i + 0.5) * scale
        first = max(int(center - support + 0.5), 0)
        stop = min(int(center + support + 0.5), in_size)
        for j in range(first, stop):
            matrix[i, j] = max(0.0, 1.0 - abs((j + 0.5 - center) / support))
        matrix[i] /= matrix[i].sum()
    return matrix


def _reference(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The tent resize in float64 with dense weight matrices."""
    rows = _tent_matrix(image.shape[0], size[1])
    columns = _tent_matrix(image.shape[1], size[0])
    height, width, channels = image.shape
    out = (rows @ image.reshape(height, width * channels)).reshape(-1, width, channels)
    out = np.einsum("pw,owc->opc", columns, out, optimize=True)
    return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)


@pytest.fixture(params=[True, False], ids=["numba", "numpy"])
def jit(request):
    """Run the resize through the Numba kernel, or the numpy one."""
    if request.param and not FLAGS.FOUND_NUMBA:
        pytest.skip("numba is not installed")
    was_on = FLAGS.JIT
    FLAGS.JIT = request.param
    yield request.param
    FLAGS.JIT = was_on


@pytest.mark.cpu
@pytest.mark.parametrize(("image_hw", "size"), DOWNSCALES)
def test_tent_resize_matches_reference(image_hw, size, jit) -> None:
    """The tent resize is within a level of the float64 tent, and of PIL's BILINEAR."""
    _ = jit
    image = _image(*image_hw)
    out = resize(image, size, antialias=True)
    assert out.shape == (size[1], size[0], 3)
    assert out.dtype == np.uint8
    diff = np.abs(out.astype(np.int16) - _reference(image, size))
    assert diff.max() <= 1
    assert diff.mean() < 0.01

    pil = pytest.importorskip("PIL.Image")
    expected = np.asarray(pil.fromarray(image).resize(size, pil.BILINEAR))
    diff = np.abs(out.astype(np.int16) - expected)
    assert diff.max() <= 1
    assert diff.mean() < 0.25


@pytest.mark.cpu
@pytest.mark.parametrize(("image_hw", "size"), NOT_DOWNSCALED)
def test_not_downscaled_is_bilinear(image_hw, size) -> None:
    """Without a downscale on either axis, antialias resizes exactly as cv2 INTER_LINEAR."""
    image = _image(*image_hw)
    expected = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(resize(image, size, antialias=True), expected)
    np.testing.assert_array_equal(resize(image, size), expected)


@pytest.mark.cpu
def test_numba_matches_numpy() -> None:
    """The Numba kernel, compiled or as Python, writes exactly the numpy implementation's pixels."""
    image = _image(97, 131, seed=1)
    for size in ((40, 30), (131, 24), (17, 97)):
        y_indices, y_weights = tent_taps(image.shape[0], size[1])
        x_indices, x_weights = tent_taps(image.shape[1], size[0])
        expected = _tent_numpy(image, y_indices, y_weights, x_indices, x_weights)
        loops = [_tent_loops.py_func, *([_tent_loops.jitted] if FLAGS.FOUND_NUMBA else [])]
        for kernel in loops:
            out = np.empty_like(expected)
            kernel(image, y_indices, y_weights, x_indices, x_weights, out)
            np.testing.assert_array_equal(out, expected)
    # non-contiguous input
    view = image[::2, ::3]
    np.testing.assert_array_equal(tent_resize(view, (20, 15)), tent_resize(view.copy(), (20, 15)))


@pytest.mark.cpu
def test_tent_taps() -> None:
    """Taps sum to one, stay inside the image, and are cached read-only."""
    for in_size, out_size in ((1920, 256), (100, 107), (7, 3), (3, 7)):
        indices, weights = tent_taps(in_size, out_size)
        assert indices.shape == weights.shape
        assert indices.shape[0] == out_size
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=1e-6)
        assert indices.min() >= 0
        assert indices.max() < in_size
        np.testing.assert_allclose(
            weights,
            _tent_matrix(in_size, out_size)[np.arange(out_size)[:, None], indices] * (weights > 0),
            atol=1e-6,
        )
        assert tent_taps(in_size, out_size)[0] is indices
        assert not weights.flags.writeable


def _bilinear_reference(
    image: np.ndarray, size: tuple[int, int], method: str
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Each resize method written with cv2 INTER_LINEAR, as before antialias existed."""
    h, w = image.shape[:2]
    width, height = size
    if method == "linear":
        return (
            cv2.resize(image, size, interpolation=cv2.INTER_LINEAR),
            (width / w, height / h),
            (
                0.0,
                0.0,
            ),
        )
    if method == "center_crop":
        (rw, rh, left, top), ratios, padding = crop_geometry(w, h, size)
        full = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
        return full[top : top + height, left : left + width], ratios, padding
    r = min(height / h, width / w)
    nw, nh = round(w * r), round(h * r)
    inner = (
        image if (nw, nh) == (w, h) else cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    )
    if method == "pad":
        top = left = 0
        bottom, right = height - nh, width - nw
        padding = (0.0, 0.0)
    else:
        dw, dh = (width - nw) / 2, (height - nh) / 2
        top, bottom, left, right = round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1)
        padding = (dw, dh)
    out = cv2.copyMakeBorder(inner, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,) * 3)
    return out, (r, r), padding


@pytest.mark.cpu
@pytest.mark.parametrize("method", RESIZE_METHODS)
@pytest.mark.parametrize(
    ("image_hw", "size"),
    [
        pytest.param((1080, 1920), (640, 640), id="1920x1080-640"),
        pytest.param((750, 1000), (224, 224), id="1000x750-224"),
        pytest.param((1999, 3001), (384, 640), id="3001x1999-384x640"),
        pytest.param((288, 447), (640, 640), id="447x288-640-up"),
    ],
)
def test_resize_methods(method, image_hw, size, jit) -> None:
    """Off is bit-identical to bilinear; on keeps the geometry and tent-resizes the image region."""
    _ = jit
    image = _image(*image_hw)
    expected, ratios, padding = _bilinear_reference(image, size, method)
    off = resize_image(image, size, method)
    np.testing.assert_array_equal(off[0], expected)
    assert off[1:] == (ratios, padding)
    np.testing.assert_array_equal(resize_image(image, size, method, antialias=False)[0], expected)

    on, on_ratios, on_padding = resize_image(image, size, method, antialias=True)
    assert on.shape == expected.shape
    assert (on_ratios, on_padding) == (ratios, padding)
    h, w = image_hw
    width, height = size
    if method == "center_crop":
        (rw, rh, left, top), _, _ = crop_geometry(w, h, size)
        region, inner = (slice(None), slice(None)), (rw, rh)
        crop = (slice(top, top + height), slice(left, left + width))
    elif method == "linear":
        region, inner, crop = (slice(None), slice(None)), size, (slice(None), slice(None))
    else:
        r = min(height / h, width / w)
        nw, nh = round(w * r), round(h * r)
        top, left = (0, 0) if method == "pad" else (round(padding[1] - 0.1), round(padding[0] - 0.1))
        region, inner = (slice(top, top + nh), slice(left, left + nw)), (nw, nh)
        crop = (slice(None), slice(None))
        border = np.ones(on.shape[:2], dtype=bool)
        border[region] = False
        np.testing.assert_array_equal(on[border], expected[border])
    downscaled = inner[0] < w or inner[1] < h
    reference = _reference(image, inner) if downscaled else cv2.resize(image, inner)
    diff = np.abs(on[region].astype(np.int16) - reference[crop])
    assert diff.max() <= 1
    if downscaled:
        assert not np.array_equal(on, expected)
    else:
        np.testing.assert_array_equal(on, expected)


@pytest.mark.cpu
@pytest.mark.parametrize("method", RESIZE_METHODS)
def test_preprocessor_antialias(method) -> None:
    """CPUPreprocessor antialiases from its constructor or per call, and per call overrides it."""
    images = [_image(1080, 1920), _image(750, 1000, seed=2)]
    plain = CPUPreprocessor((224, 224), (0.0, 1.0), np.float32, resize=method)
    smooth = CPUPreprocessor((224, 224), (0.0, 1.0), np.float32, resize=method, antialias=True)
    on = smooth.preprocess(images)
    off = plain.preprocess(images)
    np.testing.assert_array_equal(plain.preprocess(images, antialias=True)[0], on[0])
    np.testing.assert_array_equal(plain(images, antialias=True)[0], on[0])
    np.testing.assert_array_equal(smooth.preprocess(images, antialias=False)[0], off[0])
    np.testing.assert_array_equal(smooth.preprocess(images, no_copy=True)[0], on[0])
    assert on[1:] == off[1:]
    assert not np.array_equal(on[0], off[0])


@pytest.mark.cpu
def test_preprocess_function_rejects_antialias() -> None:
    """A preprocessor built with a function rejects a per-call antialias."""

    def _fn(images, *, verbose=None):
        _ = verbose
        return np.zeros((len(images), 3, 8, 8), np.float32), [(1.0, 1.0)], [(0.0, 0.0)]

    preprocessor = CPUPreprocessor((8, 8), (0.0, 1.0), np.float32, resize=_fn)
    with pytest.raises(ValueError, match="antialias"):
        preprocessor.preprocess(_image(16, 16), antialias=True)
