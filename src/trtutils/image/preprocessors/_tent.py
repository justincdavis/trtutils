# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Antialiased resize on the CPU: a tent filter whose support grows with the downscale factor.

Per axis this is what PIL's ``BILINEAR`` and torchvision's ``antialias=True`` compute: output
pixel ``i`` averages the source pixels within ``max(scale, 1)`` of its center
``(i + 0.5) * scale`` with triangle weights, renormalized at the borders. On an axis that is
not downscaled the weights are bilinear interpolation's.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from trtutils._flags import FLAGS
from trtutils._jit import register_jit


@functools.lru_cache(maxsize=32)
def tent_taps(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the source indices and weights of one axis of a tent resize.

    Parameters
    ----------
    in_size : int
        The source size of the axis.
    out_size : int
        The resized size of the axis.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The (out_size, taps) int32 source indices and float32 weights; unused taps have
        zero weight and a valid index.

    """
    scale = in_size / out_size
    support = max(scale, 1.0)
    centers = (np.arange(out_size) + 0.5) * scale
    # int() truncates toward zero, as PIL's precompute_coeffs does
    first = np.maximum((centers - support + 0.5).astype(np.int64), 0)
    stop = np.minimum((centers + support + 0.5).astype(np.int64), in_size)
    indices = first[:, None] + np.arange(int((stop - first).max()))
    weights = np.maximum(1.0 - np.abs(indices + 0.5 - centers[:, None]) / support, 0.0)
    weights[indices >= stop[:, None]] = 0.0
    weights /= weights.sum(axis=1, keepdims=True)
    indices = np.minimum(indices, in_size - 1).astype(np.int32)
    weights = weights.astype(np.float32)
    indices.flags.writeable = False
    weights.flags.writeable = False
    return indices, weights


@register_jit(nogil=True)
def _tent_loops(
    image: np.ndarray,
    y_indices: np.ndarray,
    y_weights: np.ndarray,
    x_indices: np.ndarray,
    x_weights: np.ndarray,
    out: np.ndarray,
) -> None:
    """Rows then columns, one output row at a time, float32 in between."""
    # serial: numba's thread pool costs more than the whole resize at these sizes
    height, width, channels = image.shape
    pixels = image.reshape(height, width * channels)
    out_height, y_taps = y_weights.shape
    out_width, x_taps = x_weights.shape
    row = np.empty(width * channels, dtype=np.float32)
    for oy in range(out_height):
        row[:] = 0.0
        for k in range(y_taps):
            weight = y_weights[oy, k]
            if weight != 0.0:
                source = pixels[y_indices[oy, k]]
                for i in range(width * channels):
                    row[i] += weight * np.float32(source[i])
        for ox in range(out_width):
            for c in range(channels):
                total = np.float32(0.0)
                for k in range(x_taps):
                    total += x_weights[ox, k] * row[x_indices[ox, k] * channels + c]
                out[oy, ox, c] = min(max(math.floor(total + 0.5), 0), 255)


def _tent_numpy(
    image: np.ndarray,
    y_indices: np.ndarray,
    y_weights: np.ndarray,
    x_indices: np.ndarray,
    x_weights: np.ndarray,
) -> np.ndarray:
    """Rows then columns, one gather and multiply-add per tap, float32 in between."""
    rows = _weighted_rows(image, y_indices, y_weights)
    # columns as rows: gathering whole rows is much faster than gathering columns
    out = _weighted_rows(np.ascontiguousarray(rows.transpose(1, 0, 2)), x_indices, x_weights)
    out += 0.5
    np.floor(out, out=out)
    return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8).transpose(1, 0, 2))


def _weighted_rows(image: np.ndarray, indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Sum the rows ``indices`` of ``image`` with ``weights``, one output row per weights row."""
    term = np.empty((len(weights), *image.shape[1:]), dtype=np.float32)
    total = np.zeros_like(term)
    for k in range(weights.shape[1]):
        np.multiply(weights[:, k, None, None], image[indices[:, k]], out=term)
        total += term
    return total


def tent_resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """
    Resize an HWC uint8 image with the scale-aware tent filter.

    Runs Numba-compiled while JIT is enabled, else in numpy.

    Parameters
    ----------
    image : np.ndarray
        The HWC uint8 image.
    size : tuple[int, int]
        The (width, height) to resize to.

    Returns
    -------
    np.ndarray
        The resized HWC uint8 image.

    """
    width, height = size
    y_indices, y_weights = tent_taps(image.shape[0], height)
    x_indices, x_weights = tent_taps(image.shape[1], width)
    image = np.ascontiguousarray(image)
    if FLAGS.JIT and FLAGS.FOUND_NUMBA:
        out = np.empty((height, width, image.shape[2]), dtype=np.uint8)
        _tent_loops(image, y_indices, y_weights, x_indices, x_weights, out)
        return out
    return _tent_numpy(image, y_indices, y_weights, x_indices, x_weights)
