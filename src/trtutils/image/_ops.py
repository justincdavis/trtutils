# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: RUF046
"""Image ops (resize, letterbox, patch, rescale, nms, warp) with a cv2 fast path and numpy fallback."""

from __future__ import annotations

import math
import operator

import numpy as np

from trtutils._flags import FLAGS
from trtutils._jit import register_jit
from trtutils.compat._libs import cv2

_COEF = 2048  # cv2 INTER_RESIZE_COEF_SCALE
_WARP_BITS = 5  # cv2 INTER_BITS
_WARP_TAB = 1 << _WARP_BITS
_HWC = 3


def _coords(n_out: int, n_in: int) -> tuple[np.ndarray, np.ndarray]:
    """Half-pixel source coordinates as (floor index, fraction), computed like cv2 (float32)."""
    pos = ((np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5).astype(np.float32)
    idx = np.floor(pos).astype(np.intp)
    return idx, pos - idx.astype(np.float32)


def _resize_bilinear(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Bilinear resize in numpy, matching cv2.INTER_LINEAR.

    uint8 inputs use cv2's fixed-point arithmetic and match cv2 exactly.
    Other dtypes are interpolated in floating point.

    Parameters
    ----------
    image : np.ndarray
        The image, HxW or HxWxC.
    width : int
        The output width.
    height : int
        The output height.

    Returns
    -------
    np.ndarray
        The resized image with the same dtype and number of dims.

    """
    src = image if image.ndim == _HWC else image[:, :, None]
    in_h, in_w = src.shape[:2]

    xi, xf = _coords(width, in_w)
    xf = np.where((xi < 0) | (xi >= in_w - 1), 0, xf).astype(np.float32)
    x0 = np.clip(xi, 0, in_w - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    yi, yf = _coords(height, in_h)
    y0 = np.clip(yi, 0, in_h - 1)
    y1 = np.clip(yi + 1, 0, in_h - 1)

    if src.dtype == np.uint8:
        wx1 = np.rint(xf * _COEF).astype(np.int32)[None, :, None]
        wx0 = np.rint((np.float32(1) - xf) * _COEF).astype(np.int32)[None, :, None]
        wy1 = np.rint(yf * _COEF).astype(np.int32)[:, None, None]
        wy0 = np.rint((np.float32(1) - yf) * _COEF).astype(np.int32)[:, None, None]
        wide = src.astype(np.int32)
        cols = wide[:, x0] * wx0 + wide[:, x1] * wx1
        top = ((cols[y0] >> 4) * wy0) >> 16
        bottom = ((cols[y1] >> 4) * wy1) >> 16
        out = np.clip((top + bottom + 2) >> 2, 0, 255).astype(np.uint8)
    else:
        ftype = np.float64 if src.dtype == np.float64 else np.float32
        fx = xf.astype(ftype)[None, :, None]
        fy = yf.astype(ftype)[:, None, None]
        wide = src.astype(ftype)
        cols = wide[:, x0] * (1 - fx) + wide[:, x1] * fx
        out = (cols[y0] * (1 - fy) + cols[y1] * fy).astype(src.dtype)

    return out[:, :, 0] if out.shape[2] == 1 else out  # cv2 drops a single channel axis


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if FLAGS.FOUND_CV2:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return _resize_bilinear(image, width, height)


def resize_linear(
    image: np.ndarray,
    new_shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Resize an image using linear interpolation.

    Parameters
    ----------
    image : np.ndarray
        The image to resize.
    new_shape : tuple[int, int]
        The shape to resize the image to, in form (width, height).

    Returns
    -------
    tuple[np.ndarray, tuple[float, float]]
        The resized image and the (x, y) scaling ratios.

    """
    resized = _resize(image, new_shape[0], new_shape[1])
    h, w = image.shape[:2]
    return resized, (new_shape[1] / w, new_shape[0] / h)


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """
    Resize an image while keeping its aspect ratio, padding the rest with a color.

    Parameters
    ----------
    image : np.ndarray
        The image to resize.
    new_shape : tuple[int, int]
        The output shape in form (width, height).
    color : tuple[int, int, int]
        The padding color.

    Returns
    -------
    tuple[np.ndarray, tuple[float, float], tuple[float, float]]
        The letterboxed image, the (x, y) ratio of new to old size,
        and the (dw, dh) padding applied to each side.

    """
    shape = image.shape[:2]
    r = min(new_shape[1] / shape[0], new_shape[0] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[0] - new_unpad[0]) / 2
    dh = (new_shape[1] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        image = _resize(image, new_unpad[0], new_unpad[1])
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    h, w = image.shape[:2]
    canvas = np.empty((h + top + bottom, w + left + right, *image.shape[2:]), dtype=image.dtype)
    # fill one row and broadcast it, a tuple over the channel axis is ~70x slower
    canvas[...] = (
        np.array(color, image.dtype)[None].repeat(canvas.shape[1], 0)
        if image.ndim == _HWC
        else color[0]
    )
    canvas[top : top + h, left : left + w] = image
    return canvas, (r, r), (dw, dh)


def patch(
    image: np.ndarray,
    patch_size: tuple[int, int],
    padding: int | tuple[int, int] | None = None,
    overlap: float | tuple[float, float] | None = None,
) -> tuple[list[np.ndarray], list[tuple[int, int]], tuple[int, int]]:
    """
    Divide an image into patches of a specific size.

    If the image is not evenly divisible by the patch size, it is
    scaled to the closest size that is, based on the padding/overlap.

    Parameters
    ----------
    image : np.ndarray
        The image to patch.
    patch_size : tuple[int, int]
        The patch size in (width, height).
    padding : int, tuple[int, int], optional
        The number of pixels (or pixel x/y) the patches overlap by.
    overlap : float, tuple[float, float], optional
        The ratio of the patch dimension the patches overlap by.

    Returns
    -------
    tuple[list[np.ndarray], list[tuple[int, int]], tuple[int, int]]
        The patches, their (x, y) offsets, and the (width, height)
        the patches and offsets are based on.

    Raises
    ------
    ValueError
        If the overlap in x or y is not smaller than the patch width or height.

    """

    def patch_dimension(patch_dim: int, image_dim: int, pad: int) -> tuple[int, int]:
        if image_dim < patch_dim:
            return patch_dim, 1
        stride = patch_dim - pad
        num_patches = math.ceil((image_dim - patch_dim) / stride) + 1
        return stride * (num_patches - 1) + patch_dim, num_patches

    height, width = image.shape[:2]
    patch_width, patch_height = patch_size

    pad = (0, 0)
    if padding:
        pad = padding if isinstance(padding, tuple) else (padding, padding)
    if overlap:
        ox, oy = overlap if isinstance(overlap, tuple) else (overlap, overlap)
        pad = (int(ox * patch_width), int(oy * patch_height))

    if pad[0] >= patch_width:
        err_msg = f"Pad x cannot be greater than or equal to patch width: {pad[0]} !>= {patch_width}"
        raise ValueError(err_msg)
    if pad[1] >= patch_height:
        err_msg = (
            f"Pad y cannot be greater than or equal to patch height: {pad[1]} !>= {patch_height}"
        )
        raise ValueError(err_msg)

    new_height, num_h_patches = patch_dimension(patch_height, height, pad[1])
    new_width, num_w_patches = patch_dimension(patch_width, width, pad[0])
    resized = _resize(image, new_width, new_height)

    patches: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    for i in range(num_h_patches):
        for j in range(num_w_patches):
            y_start = i * (patch_height - pad[1])
            x_start = j * (patch_width - pad[0])
            patches.append(
                resized[y_start : y_start + patch_height, x_start : x_start + patch_width]
            )
            offsets.append((x_start, y_start))

    return patches, offsets, (new_width, new_height)


def rescale(image: np.ndarray, value_range: tuple[float, float]) -> np.ndarray:
    """
    Rescale a uint8 image to be within the given value range.

    Parameters
    ----------
    image : np.ndarray
        The image to scale, of type np.uint8.
    value_range : tuple[float, float]
        The (low, high) values to scale to.

    Returns
    -------
    np.ndarray
        The rescaled image.

    Raises
    ------
    ValueError
        If value_range[0] is not less than value_range[1].

    """
    low, high = value_range
    if low >= high:
        err_msg = "Low value of range cannot be higher than or equal to high value."
        raise ValueError(err_msg)

    img_float = image.astype(float)
    if low == 0.0:
        return img_float * (high / 255.0)
    return (img_float / 255.0) * (high - low) + low


@register_jit()
def _nms_kernel(
    bboxes: list[tuple[tuple[int, int, int, int], float, int]],
    iou_threshold: float,
    agnostic: bool,  # noqa: FBT001
) -> list[bool]:
    keep = [True] * len(bboxes)

    for i in range(len(bboxes)):
        if not keep[i]:
            continue

        box1 = bboxes[i]
        x1a, y1a, x2a, y2a = box1[0]
        area1 = (x2a - x1a) * (y2a - y1a)
        for j in range(i + 1, len(bboxes)):
            if not keep[j]:
                continue

            box2 = bboxes[j]
            if not agnostic and box1[2] != box2[2]:
                continue

            x1b, y1b, x2b, y2b = box2[0]
            inter = max(0, min(x2a, x2b) - max(x1a, x1b)) * max(0, min(y2a, y2b) - max(y1a, y1b))
            if inter == 0:
                continue
            union = area1 + (x2b - x1b) * (y2b - y1b) - inter
            if union != 0 and inter / union > iou_threshold:
                keep[j] = False

    return keep


def nms(
    bboxes: list[tuple[tuple[int, int, int, int], float, int]],
    iou_threshold: float = 0.5,
    *,
    agnostic: bool | None = None,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    """
    Perform non-maximum suppression on a list of bounding boxes.

    Parameters
    ----------
    bboxes : list[tuple[tuple[int, int, int, int], float, int]]
        The boxes as ((x1, y1, x2, y2), confidence, class).
    iou_threshold : float
        The IoU above which the lower scoring box is suppressed.
    agnostic : bool, optional
        If True, boxes of different classes also suppress each other.

    Returns
    -------
    list[tuple[tuple[int, int, int, int], float, int]]
        The kept boxes, sorted by descending confidence.

    """
    if not bboxes:
        return []
    ordered = sorted(bboxes, key=operator.itemgetter(1), reverse=True)
    keep = _nms_kernel(ordered, iou_threshold, bool(agnostic))
    return [box for box, kept in zip(ordered, keep) if kept]


def _warp_axis(
    n_out: int,
    n_in: int,
    scale: float,
    offset: float,
    *,
    rows: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # cv2 warpAffine fixed point: 10 bit coords, rounded down to a 1/32 grid
    s = float(np.float32(scale))
    t = float(np.float32(offset))
    k = np.arange(n_out)
    fixed = np.rint((s * k + t) * 1024) if rows else np.rint(t * 1024) + np.rint(s * k * 1024)
    pos = (fixed.astype(np.int64) + 16) >> 5
    idx = pos >> _WARP_BITS
    frac = (pos & (_WARP_TAB - 1)).astype(np.float32) / np.float32(_WARP_TAB)
    return np.clip(idx, 0, n_in - 1), np.clip(idx + 1, 0, n_in - 1), frac


def warp_scale(
    src: np.ndarray,
    scale: tuple[float, float],
    offset: tuple[float, float],
    size: tuple[int, int],
) -> np.ndarray:
    """
    Sample a float32 2D array with a scale-and-offset inverse map.

    Output pixel (x, y) reads src at (scale_x * x + offset_x, scale_y * y + offset_y)
    with bilinear interpolation and replicated borders.
    Equivalent to cv2.warpAffine with WARP_INVERSE_MAP, INTER_LINEAR, BORDER_REPLICATE.

    Parameters
    ----------
    src : np.ndarray
        The float32 2D source array.
    scale : tuple[float, float]
        The (x, y) scale from output to source coordinates.
    offset : tuple[float, float]
        The (x, y) offset in source coordinates.
    size : tuple[int, int]
        The output size in form (width, height).

    Returns
    -------
    np.ndarray
        The float32 output of shape (height, width).

    """
    width, height = size
    if FLAGS.FOUND_CV2:
        affine = np.array(
            [[scale[0], 0.0, offset[0]], [0.0, scale[1], offset[1]]],
            dtype=np.float32,
        )
        return cv2.warpAffine(
            src,
            affine,
            (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )

    y0, y1, fy = _warp_axis(height, src.shape[0], scale[1], offset[1], rows=True)
    x0, x1, fx = _warp_axis(width, src.shape[1], scale[0], offset[0], rows=False)
    rows = src[y0] * (1 - fy)[:, None] + src[y1] * fy[:, None]
    return rows[:, x0] * (1 - fx) + rows[:, x1] * fx
