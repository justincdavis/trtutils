# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
#
# perspective_crop is a Python/numba port of OpenCV's getPerspectiveTransform and warpPerspective (INTER_CUBIC)
# (Copyright (C) OpenCV contributors, Apache License 2.0, https://www.apache.org/licenses/LICENSE-2.0),
# translated from C++ and adapted to numba; the fixed-point arithmetic is kept to match its results.
"""Perspective crop (getPerspectiveTransform + warpPerspective cubic) with a cv2 fast path."""

from __future__ import annotations

import numpy as np

from trtutils._flags import FLAGS
from trtutils._jit import register_jit
from trtutils.compat._libs import cv2

try:
    from numba import prange
except ImportError:
    prange = range

_INTER_BITS = 5
_TAB_SIZE = 1 << _INTER_BITS
_COEF_BITS = 15
_COEF_SCALE = 1 << _COEF_BITS
_BLOCK_SZ = 32
_INT_MAX = 2147483647.0
_INT_MIN = -2147483648.0


def _cubic_taps(x: np.float32) -> np.ndarray:
    a = np.float32(-0.75)
    one = np.float32(1)
    c = np.zeros(4, dtype=np.float32)
    x1 = x + one
    c[0] = ((a * x1 - np.float32(5) * a) * x1 + np.float32(8) * a) * x1 - np.float32(4) * a
    c[1] = ((a + np.float32(2)) * x - (a + np.float32(3))) * x * x + one
    y = one - x
    c[2] = ((a + np.float32(2)) * y - (a + np.float32(3))) * y * y + one
    c[3] = one - c[0] - c[1] - c[2]
    return c


def _build_cubic_table() -> np.ndarray:
    """Build OpenCV's 32x32x(4x4) fixed-point cubic weights (initInterTab2D)."""
    taps = np.stack([_cubic_taps(np.float32(i) * np.float32(1.0 / _TAB_SIZE)) for i in range(32)])
    tab = np.zeros((_TAB_SIZE, _TAB_SIZE, 4, 4), dtype=np.int32)
    for i in range(_TAB_SIZE):
        for j in range(_TAB_SIZE):
            v = taps[i][:, None] * taps[j][None, :]
            t = np.clip(np.rint(v * np.float32(_COEF_SCALE)), -32768, 32767).astype(np.int32)
            diff = int(t.sum()) - _COEF_SCALE
            if diff != 0:
                # nudge the smallest/largest of the centre 2x2 taps so the weights sum to 1
                lo = hi = (2, 2)
                for k1 in (2, 3):
                    for k2 in (2, 3):
                        if t[k1, k2] < t[lo]:
                            lo = (k1, k2)
                        elif t[k1, k2] > t[hi]:
                            hi = (k1, k2)
                t[hi if diff < 0 else lo] -= diff
            tab[i, j] = t
    return tab.ravel()


_CUBIC_TAB = _build_cubic_table()


@register_jit()
def _inverse_matrix(  # noqa: C901, PLR0915
    points: np.ndarray, width: int, height: int, out: np.ndarray
) -> None:
    """Write inv(getPerspectiveTransform(points, rect)) as float64 3x3 (LU, like OpenCV)."""
    dst = np.zeros((4, 2), dtype=np.float32)
    dst[1, 0] = width
    dst[2, 0] = width
    dst[2, 1] = height
    dst[3, 1] = height
    a = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    for i in range(4):
        sx = points[i, 0]
        sy = points[i, 1]
        a[i, 0] = sx
        a[i + 4, 3] = sx
        a[i, 1] = sy
        a[i + 4, 4] = sy
        a[i, 2] = 1.0
        a[i + 4, 5] = 1.0
        a[i, 6] = -sx * dst[i, 0]
        a[i, 7] = -sy * dst[i, 0]
        a[i + 4, 6] = -sx * dst[i, 1]
        a[i + 4, 7] = -sy * dst[i, 1]
        b[i] = dst[i, 0]
        b[i + 4] = dst[i, 1]
    n = 8
    for i in range(n):
        k = i
        for j in range(i + 1, n):
            if abs(a[j, i]) > abs(a[k, i]):
                k = j
        if k != i:
            for c in range(i, n):
                t = a[i, c]
                a[i, c] = a[k, c]
                a[k, c] = t
            t = b[i]
            b[i] = b[k]
            b[k] = t
        d = -1.0 / a[i, i]
        for j in range(i + 1, n):
            alpha = a[j, i] * d
            for c in range(i + 1, n):
                a[j, c] += alpha * a[i, c]
            b[j] += alpha * b[i]
    for i in range(n - 1, -1, -1):
        s = b[i]
        for c in range(i + 1, n):
            s -= a[i, c] * b[c]
        b[i] = s / a[i, i]

    m = np.ones(9, dtype=np.float64)
    m[:8] = b
    det = (
        m[0] * (m[4] * m[8] - m[5] * m[7])
        - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
    )
    d = 1.0 / det
    out[0] = (m[4] * m[8] - m[5] * m[7]) * d
    out[1] = (m[2] * m[7] - m[1] * m[8]) * d
    out[2] = (m[1] * m[5] - m[2] * m[4]) * d
    out[3] = (m[5] * m[6] - m[3] * m[8]) * d
    out[4] = (m[0] * m[8] - m[2] * m[6]) * d
    out[5] = (m[2] * m[3] - m[0] * m[5]) * d
    out[6] = (m[3] * m[7] - m[4] * m[6]) * d
    out[7] = (m[1] * m[6] - m[0] * m[7]) * d
    out[8] = (m[0] * m[4] - m[1] * m[3]) * d


@register_jit(parallel=True)
def _warp_cubic(image: np.ndarray, m: np.ndarray, tab: np.ndarray, out: np.ndarray) -> None:
    """Inverse-map every output pixel and sample with the 4x4 fixed-point cubic, replicating edges."""
    src_h, src_w, chans = image.shape
    dst_h, dst_w, _ = out.shape
    if dst_h == 0 or dst_w == 0:
        return
    bh0 = min(_BLOCK_SZ // 2, dst_h)
    bw0 = min(_BLOCK_SZ * _BLOCK_SZ // bh0, dst_w)
    delta = 1 << (_COEF_BITS - 1)
    mask = _TAB_SIZE - 1
    for y in prange(dst_h):
        for x in range(dst_w):
            # opencv evaluates the row origin at each block start, then adds the in-block offset
            xb = (x // bw0) * bw0
            x1 = x - xb
            x0 = m[0] * xb + m[1] * y + m[2]
            y0 = m[3] * xb + m[4] * y + m[5]
            w = m[6] * xb + m[7] * y + m[8] + m[6] * x1
            w = _TAB_SIZE / w if w != 0.0 else 0.0
            fx = max(_INT_MIN, min(_INT_MAX, (x0 + m[0] * x1) * w))
            fy = max(_INT_MIN, min(_INT_MAX, (y0 + m[3] * x1) * w))
            xi = int(np.rint(fx))
            yi = int(np.rint(fy))
            sx = max(-32768, min(32767, xi >> _INTER_BITS)) - 1
            sy = max(-32768, min(32767, yi >> _INTER_BITS)) - 1
            base = ((yi & mask) * _TAB_SIZE + (xi & mask)) * 16
            for c in range(chans):
                acc = 0
                for i in range(4):
                    yy = min(max(sy + i, 0), src_h - 1)
                    for j in range(4):
                        xx = min(max(sx + j, 0), src_w - 1)
                        acc += int(image[yy, xx, c]) * tab[base + i * 4 + j]
                out[y, x, c] = min(max((acc + delta) >> _COEF_BITS, 0), 255)


def _warp_cubic_numpy(image: np.ndarray, m: np.ndarray, tab: np.ndarray, out: np.ndarray) -> None:
    """Sample with the same fixed-point cubic as ``_warp_cubic``."""
    src_h, src_w = image.shape[:2]
    dst_h, dst_w, _ = out.shape
    if dst_h == 0 or dst_w == 0:
        return
    bh0 = min(_BLOCK_SZ // 2, dst_h)
    bw0 = min(_BLOCK_SZ * _BLOCK_SZ // bh0, dst_w)
    delta = 1 << (_COEF_BITS - 1)
    bits = _TAB_SIZE - 1
    ys, xs = np.indices((dst_h, dst_w))
    xb = (xs // bw0) * bw0
    x1 = xs - xb
    x0 = m[0] * xb + m[1] * ys + m[2]
    y0 = m[3] * xb + m[4] * ys + m[5]
    denom = m[6] * xb + m[7] * ys + m[8] + m[6] * x1
    scale = np.zeros(denom.shape, dtype=np.float64)
    np.divide(float(_TAB_SIZE), denom, out=scale, where=denom != 0.0)
    fx = np.clip((x0 + m[0] * x1) * scale, _INT_MIN, _INT_MAX)
    fy = np.clip((y0 + m[3] * x1) * scale, _INT_MIN, _INT_MAX)
    xi = np.rint(fx).astype(np.int32)
    yi = np.rint(fy).astype(np.int32)
    sx = np.clip(xi >> _INTER_BITS, -32768, 32767) - 1
    sy = np.clip(yi >> _INTER_BITS, -32768, 32767) - 1
    base = ((yi & bits) * _TAB_SIZE + (xi & bits)) * 16
    acc = np.zeros(out.shape, dtype=np.int32)
    for i in range(4):
        yy = np.clip(sy + i, 0, src_h - 1)
        for j in range(4):
            xx = np.clip(sx + j, 0, src_w - 1)
            acc += image[yy, xx].astype(np.int32) * tab[base + i * 4 + j, None]
    np.copyto(out, np.clip((acc + delta) >> _COEF_BITS, 0, 255), casting="unsafe")


def perspective_crop(image: np.ndarray, points: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Rectify a quadrilateral of an image into a (height, width) crop.

    Equivalent to cv2.getPerspectiveTransform onto the axis-aligned rectangle followed by
    cv2.warpPerspective with INTER_CUBIC and BORDER_REPLICATE, bit-exact for uint8 images.

    Parameters
    ----------
    image : np.ndarray
        The uint8 image, HWC or HW.
    points : np.ndarray
        The quad corners (top-left, top-right, bottom-right, bottom-left), shape (4, 2).
    width : int
        The output width.
    height : int
        The output height.

    Returns
    -------
    np.ndarray
        The uint8 crop with shape (height, width) plus the channel axis of the image.

    """
    quad = np.ascontiguousarray(points, dtype=np.float32)
    if FLAGS.FOUND_CV2:
        target = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        return cv2.warpPerspective(
            image,
            cv2.getPerspectiveTransform(quad, target),
            (width, height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
    src = np.ascontiguousarray(image)
    planar = src.ndim == 2  # noqa: PLR2004
    if planar:
        src = src[:, :, None]
    inv = np.zeros(9, dtype=np.float64)
    _inverse_matrix(quad, width, height, inv)
    out = np.empty((height, width, src.shape[2]), dtype=np.uint8)
    if FLAGS.FOUND_NUMBA and FLAGS.JIT:
        _warp_cubic(src, inv, _CUBIC_TAB, out)
    else:
        _warp_cubic_numpy(src, inv, _CUBIC_TAB, out)
    return out[:, :, 0] if planar else out
