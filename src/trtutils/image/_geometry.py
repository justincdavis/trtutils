# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Geometry ops (find_contours) with a cv2 fast path and a numba/numpy fallback."""

from __future__ import annotations

import numpy as np

from trtutils._flags import FLAGS
from trtutils._jit import register_jit
from trtutils.compat._libs import cv2

# 8 neighbours in clockwise order (image coords, y down), starting east
_DX = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
_DY = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
_EAST = 0
_WEST = 4
_NDIR = 8


@register_jit()
def _trace(  # noqa: C901, PLR0912, PLR0915
    grid: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    pts: np.ndarray,
    tmp: np.ndarray,
    offsets: np.ndarray,
) -> int:
    """
    Suzuki-Abe border following on a zero padded grid, filling compressed points.

    Returns the contour count, or -1 if the point buffers were too small.
    """
    height, width = grid.shape
    cap = pts.shape[0]
    n_out = 0
    n_contours = 0
    nbd = 1
    offsets[0] = 0
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            val = grid[y, x]
            if val < 1:
                continue
            if val == 1 and grid[y, x - 1] == 0:
                start_dir = 4
            elif grid[y, x + 1] == 0:
                start_dir = 0
            else:
                continue
            nbd += 1

            # first neighbour found clockwise from the start direction
            first = -1
            for k in range(8):
                d = (start_dir + k) % 8
                if grid[y + dy[d], x + dx[d]] != 0:
                    first = d
                    break
            if first < 0:
                grid[y, x] = -nbd
                if n_out + 1 > cap:
                    return -1
                pts[n_out, 0] = x
                pts[n_out, 1] = y
                n_out += 1
                n_contours += 1
                offsets[n_contours] = n_out
                continue

            n_tmp = 0
            x1 = x + dx[first]
            y1 = y + dy[first]
            x2 = x1
            y2 = y1
            x3 = x
            y3 = y
            while True:
                # counterclockwise from the previous point around the current one
                d2 = 0
                for d in range(8):
                    if x2 - x3 == dx[d] and y2 - y3 == dy[d]:
                        d2 = d
                        break
                east_zero = False
                x4 = x3
                y4 = y3
                for k in range(1, 9):
                    d = (d2 - k) % 8
                    if grid[y3 + dy[d], x3 + dx[d]] != 0:
                        x4 = x3 + dx[d]
                        y4 = y3 + dy[d]
                        break
                    if d == 0:
                        east_zero = True
                if n_tmp >= cap:
                    return -1
                tmp[n_tmp, 0] = x3
                tmp[n_tmp, 1] = y3
                n_tmp += 1
                if east_zero:
                    grid[y3, x3] = -nbd
                elif grid[y3, x3] == 1:
                    grid[y3, x3] = nbd
                if x4 == x and y4 == y and x3 == x1 and y3 == y1:
                    break
                x2 = x3
                y2 = y3
                x3 = x4
                y3 = y4

            # keep points where the incoming and outgoing step differ (cyclic)
            for k in range(n_tmp):
                p = (k - 1) % n_tmp
                q = (k + 1) % n_tmp
                in_x = np.sign(tmp[k, 0] - tmp[p, 0])
                in_y = np.sign(tmp[k, 1] - tmp[p, 1])
                out_x = np.sign(tmp[q, 0] - tmp[k, 0])
                out_y = np.sign(tmp[q, 1] - tmp[k, 1])
                if in_x != out_x or in_y != out_y or n_tmp == 1:
                    if n_out >= cap:
                        return -1
                    pts[n_out, 0] = tmp[k, 0]
                    pts[n_out, 1] = tmp[k, 1]
                    n_out += 1
            n_contours += 1
            offsets[n_contours] = n_out
    return n_contours


def _find_contours_numpy(binary: np.ndarray) -> list[np.ndarray]:
    height, width = binary.shape
    grid = np.zeros((height + 2, width + 2), dtype=np.int32)
    grid[1:-1, 1:-1] = binary != 0
    cap = 4 * int(np.count_nonzero(binary)) + 16
    while True:
        pts = np.empty((cap, 2), dtype=np.int32)
        tmp = np.empty((cap, 2), dtype=np.int32)
        offsets = np.zeros(cap + 1, dtype=np.int64)
        work = grid.copy()
        count = _trace(work, _DX, _DY, pts, tmp, offsets)
        if count >= 0:
            break
        cap *= 2
    # padding shifts coordinates by one; opencv lists the last found contour first
    return [pts[offsets[i] : offsets[i + 1]] - 1 for i in range(count - 1, -1, -1)]


def find_contours(binary: np.ndarray) -> list[np.ndarray]:
    """
    Find all contours of a binary map, like cv2.RETR_LIST with CHAIN_APPROX_SIMPLE.

    Outer borders and hole borders are both returned, foreground is 8-connected and
    points in the middle of horizontal, vertical and diagonal runs are dropped.

    Parameters
    ----------
    binary : np.ndarray
        The 2D map, any nonzero value is foreground.

    Returns
    -------
    list[np.ndarray]
        One (N, 2) int32 array of (x, y) points per contour, in OpenCV's order.

    """
    if FLAGS.FOUND_CV2:
        mask = (binary != 0).astype(np.uint8) * 255
        found = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        return [c.reshape(-1, 2) for c in found]
    return _find_contours_numpy(binary)
