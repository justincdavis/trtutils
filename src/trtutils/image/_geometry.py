# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
#
# min_area_rect is a Python/numba port of OpenCV's rotating-calipers minAreaRect and boxPoints
# (Copyright (C) OpenCV contributors, Apache License 2.0, https://www.apache.org/licenses/LICENSE-2.0),
# translated from C++ and adapted to numba; the float32 arithmetic is kept to match its results.
# _fill_poly is a port of the fixed point scanline fill and Bresenham line of OpenCV
# fillPoly (drawing.cpp, Apache License 2.0), kept integer exact to match its pixels.
"""Geometry ops (find_contours, min_area_rect, ...) with a cv2 fast path and a numpy fallback."""

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


@register_jit()
def _min_area_rect(pts: np.ndarray, order: np.ndarray) -> tuple[np.ndarray, np.float32]:  # noqa: C901, PLR0912, PLR0915
    """
    Port of OpenCV's rotating calipers on the hull of sorted (N, 2) float32 points.

    Follows the float32 arithmetic of cv2.minAreaRect and cv2.boxPoints so the corners match.
    Returns the 4 corners (4, 2) float32 and the short side.
    """
    pi = np.pi
    half = np.float32(0.5)
    zero = np.float32(0)
    total = pts.shape[0]
    buf = np.empty((2 * total + 1, 2), dtype=np.float32)
    ids = np.empty(2 * total + 1, dtype=np.int64)
    h = 0
    lower = 2
    last = 0
    for i in range(2 * total - 1):
        idx = i if i < total else 2 * total - 2 - i
        if i == total:
            lower = h + 1
        while h >= lower:
            ax = np.float64(buf[h - 1, 0]) - np.float64(buf[h - 2, 0])
            ay = np.float64(buf[h - 1, 1]) - np.float64(buf[h - 2, 1])
            bx = np.float64(pts[idx, 0]) - np.float64(buf[h - 2, 0])
            by = np.float64(pts[idx, 1]) - np.float64(buf[h - 2, 1])
            if ax * by - ay * bx > 0:
                break
            h -= 1
        buf[h] = pts[idx]
        ids[h] = order[idx]
        h += 1
        if i == total - 1:
            last = h - 1
    h = max(h - 1, 1)
    hull = np.empty((h, 2), dtype=np.float32)
    hids = np.empty(h, dtype=np.int64)
    for i in range(h):
        hull[i] = buf[(last + i) % h]
        hids[i] = ids[(last + i) % h]

    # opencv rotates its hull so the original point indices run monotonic
    start = 0
    min_i = 0
    max_i = 0
    lt = 0
    for i in range(1, h):
        lt += hids[i - 1] < hids[i]
        if lt > 1 and lt <= i - 2:
            break
        if hids[i] < hids[min_i]:
            min_i = i
        if hids[i] > hids[max_i]:
            max_i = i
    dist = abs(max_i - min_i)
    if h >= 3 and (dist in (1, h - 1)) and (lt <= 1 or lt >= h - 2):  # noqa: PLR2004
        ascending = (max_i + 1) % h == min_i
        i0 = min_i if ascending else max_i
        j = i0
        ok = i0 > 0
        for _ in range(h - 1):
            nxt = (j + 1) % h
            if ascending != (hids[j] < hids[nxt]):
                ok = False
                break
            j = nxt
        if ok:
            start = i0
    hull = np.concatenate((hull[start:], hull[:start]))
    n = hull.shape[0]
    cx = hull[0, 0]
    cy = hull[0, 1]
    width = zero
    height = zero
    angle = zero

    if n == 2:  # noqa: PLR2004
        cx = (hull[0, 0] + hull[1, 0]) * half
        cy = (hull[0, 1] + hull[1, 1]) * half
        dx = np.float64(hull[1, 0] - hull[0, 0])
        dy = np.float64(hull[1, 1] - hull[0, 1])
        width = np.float32(np.sqrt(dx * dx + dy * dy))
        angle = np.float32(np.arctan2(dy, dx))
    elif n > 2:  # noqa: PLR2004
        vect = np.empty((n, 2), dtype=np.float32)
        inv_len = np.empty(n, dtype=np.float32)
        left = 0
        right = 0
        top = 0
        bottom = 0
        for i in range(n):
            if hull[i, 0] < hull[left, 0]:
                left = i
            if hull[i, 0] > hull[right, 0]:
                right = i
            if hull[i, 1] > hull[top, 1]:
                top = i
            if hull[i, 1] < hull[bottom, 1]:
                bottom = i
            nxt = (i + 1) % n
            dx = np.float64(hull[nxt, 0] - hull[i, 0])
            dy = np.float64(hull[nxt, 1] - hull[i, 1])
            vect[i, 0] = dx
            vect[i, 1] = dy
            inv_len[i] = 1.0 / np.sqrt(dx * dx + dy * dy)

        orientation = np.float32(1)
        ax = np.float64(vect[n - 1, 0])
        ay = np.float64(vect[n - 1, 1])
        for i in range(n):
            bx = np.float64(vect[i, 0])
            by = np.float64(vect[i, 1])
            convexity = ax * by - ay * bx
            if convexity != 0:
                orientation = np.float32(1 if convexity > 0 else -1)
                break
            ax = bx
            ay = by

        base_a = orientation
        base_b = zero
        seq = np.array([bottom, right, top, left], dtype=np.int64)
        rot = np.empty((4, 2), dtype=np.float32)
        min_area = np.float32(np.finfo(np.float32).max)
        best_a = zero
        best_b = zero
        best_w = zero
        best_h = zero
        best_left = 0
        best_bottom = 0
        for _ in range(n):
            rot[0, 0] = vect[seq[0], 0]
            rot[0, 1] = vect[seq[0], 1]
            rot[1, 0] = vect[seq[1], 1]
            rot[1, 1] = -vect[seq[1], 0]
            rot[2, 0] = -vect[seq[2], 0]
            rot[2, 1] = -vect[seq[2], 1]
            rot[3, 0] = -vect[seq[3], 1]
            rot[3, 1] = vect[seq[3], 0]
            main = 0
            for i in range(1, 4):
                # first vector is to the right (clockwise) of the current main one
                if rot[i, 1] * rot[main, 0] + (-rot[i, 0]) * rot[main, 1] < 0:
                    main = i

            pi_ = seq[main]
            lead_x = vect[pi_, 0] * inv_len[pi_]
            lead_y = vect[pi_, 1] * inv_len[pi_]
            if main == 0:
                base_a = lead_x
                base_b = lead_y
            elif main == 1:
                base_a = lead_y
                base_b = -lead_x
            elif main == 2:  # noqa: PLR2004
                base_a = -lead_x
                base_b = -lead_y
            else:
                base_a = -lead_y
                base_b = lead_x
            seq[main] = (seq[main] + 1) % n

            dx = hull[seq[1], 0] - hull[seq[3], 0]
            dy = hull[seq[1], 1] - hull[seq[3], 1]
            w = dx * base_a + dy * base_b
            dx = hull[seq[2], 0] - hull[seq[0], 0]
            dy = hull[seq[2], 1] - hull[seq[0], 1]
            hgt = -dx * base_b + dy * base_a
            area = w * hgt
            if area <= min_area:
                min_area = area
                best_left = seq[3]
                best_bottom = seq[0]
                best_a = base_a
                best_b = base_b
                best_w = w
                best_h = hgt

        a1 = best_a
        b1 = best_b
        a2 = -best_b
        b2 = best_a
        c1 = a1 * hull[best_left, 0] + hull[best_left, 1] * b1
        c2 = a2 * hull[best_bottom, 0] + hull[best_bottom, 1] * b2
        idet = np.float32(1) / (a1 * b2 - a2 * b1)
        px = (c1 * b2 - c2 * b1) * idet
        py = (a1 * c2 - a2 * c1) * idet
        v1x = a1 * best_w
        v1y = b1 * best_w
        v2x = a2 * best_h
        v2y = b2 * best_h
        cx = px + (v1x + v2x) * half
        cy = py + (v1y + v2y) * half
        width = np.float32(np.sqrt(np.float64(v1x) * v1x + np.float64(v1y) * v1y))
        height = np.float32(np.sqrt(np.float64(v2x) * v2x + np.float64(v2y) * v2y))
        angle = np.float32(np.arctan2(np.float64(v1y), np.float64(v1x)))

    angle = np.float32(np.float64(angle * np.float32(180)) / pi)
    rad = np.float64(angle) * pi / 180.0
    b = np.float32(np.cos(rad)) * half
    a = np.float32(np.sin(rad)) * half
    corners = np.empty((4, 2), dtype=np.float32)
    corners[0, 0] = cx - a * height - b * width
    corners[0, 1] = cy + b * height - a * width
    corners[1, 0] = cx + a * height - b * width
    corners[1, 1] = cy - b * height - a * width
    corners[2, 0] = np.float32(2) * cx - corners[0, 0]
    corners[2, 1] = np.float32(2) * cy - corners[0, 1]
    corners[3, 0] = np.float32(2) * cx - corners[1, 0]
    corners[3, 1] = np.float32(2) * cy - corners[1, 1]
    return corners, min(width, height)


def min_area_rect(points: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Minimum-area rotated rectangle of 2D points, like cv2.minAreaRect plus cv2.boxPoints.

    Degenerate input gives a degenerate rectangle (coincident corners, short side 0)
    and no points gives zeros, as in OpenCV.

    Parameters
    ----------
    points : np.ndarray
        The (N, 2) points, any numeric dtype.

    Returns
    -------
    tuple[np.ndarray, float]
        The 4 corners as (4, 2) float32 in no particular order, and the short side.

    """
    if len(points) == 0:
        return np.zeros((4, 2), dtype=np.float32), 0.0
    if FLAGS.FOUND_CV2:
        pts = points if points.dtype in (np.int32, np.float32) else points.astype(np.float32)
        rect = cv2.minAreaRect(np.ascontiguousarray(pts))
        return cv2.boxPoints(rect), float(min(rect[1]))
    pts = np.asarray(points, dtype=np.float32)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    corners, short = _min_area_rect(pts[order], order)
    return corners, float(short)


def dilate_2x2(binary: np.ndarray) -> np.ndarray:
    """
    Dilate a 2D uint8 map with a 2x2 kernel, like cv2.dilate with np.ones((2, 2)).

    Each output pixel is the max over the 2x2 window ending at it (anchor (1, 1)),
    and pixels outside the map are ignored.

    Parameters
    ----------
    binary : np.ndarray
        The (H, W) uint8 map.

    Returns
    -------
    np.ndarray
        The dilated (H, W) uint8 map.

    """
    if FLAGS.FOUND_CV2:
        return cv2.dilate(binary, np.ones((2, 2), dtype=np.uint8))
    out = binary.copy()
    np.maximum(out[:, 1:], binary[:, :-1], out=out[:, 1:])
    np.maximum(out[1:, :], out[:-1, :], out=out[1:, :])
    return out


@register_jit()
def _fill_poly(mask: np.ndarray, pts: np.ndarray) -> None:  # noqa: C901, PLR0912, PLR0915
    """Fill an int polygon like cv2.fillPoly with LINE_8, edges included."""
    height, width = mask.shape
    n = pts.shape[0]
    right = width - 1
    bottom = height - 1
    ey0 = np.empty(n, dtype=np.int64)
    ey1 = np.empty(n, dtype=np.int64)
    ex = np.empty(n, dtype=np.int64)
    edx = np.empty(n, dtype=np.int64)
    n_edges = 0
    for i in range(n):
        ax = int(pts[i - 1, 0])
        ay = int(pts[i - 1, 1])
        bx = int(pts[i, 0])
        by = int(pts[i, 1])

        # clip like cv2.clipLine, the edge is drawn clipped and filled from the clipped line
        x1, y1, x2, y2 = ax, ay, bx, by
        c1 = (x1 < 0) + (x1 > right) * 2 + (y1 < 0) * 4 + (y1 > bottom) * 8
        c2 = (x2 < 0) + (x2 > right) * 2 + (y2 < 0) * 4 + (y2 > bottom) * 8
        if (c1 & c2) == 0 and (c1 | c2) != 0:
            if c1 & 12:
                a = 0 if c1 < 8 else bottom  # noqa: PLR2004
                x1 += int(float(a - y1) * (x2 - x1) / (y2 - y1))
                y1 = a
                c1 = (x1 < 0) + (x1 > right) * 2
            if c2 & 12:
                a = 0 if c2 < 8 else bottom  # noqa: PLR2004
                x2 += int(float(a - y2) * (x2 - x1) / (y2 - y1))
                y2 = a
                c2 = (x2 < 0) + (x2 > right) * 2
            if (c1 & c2) == 0 and (c1 | c2) != 0:
                if c1:
                    a = 0 if c1 == 1 else right
                    y1 += int(float(a - x1) * (y2 - y1) / (x2 - x1))
                    x1 = a
                    c1 = 0
                if c2:
                    a = 0 if c2 == 1 else right
                    y2 += int(float(a - x2) * (y2 - y1) / (x2 - x1))
                    x2 = a
                    c2 = 0
        if (c1 | c2) == 0:
            # bresenham from the left end, err < 0 steps the minor axis too
            if x2 < x1:
                lx, rx, ly, ry = x2, x1, y2, y1
            else:
                lx, rx, ly, ry = x1, x2, y1, y2
            ddx = rx - lx
            ddy = abs(ry - ly)
            ystep = 1 if ry > ly else -1
            if ddy > ddx:
                major = ddy
                minor = ddx
            else:
                major = ddx
                minor = ddy
            err = major - 2 * minor
            px = lx
            py = ly
            for _ in range(major + 1):
                mask[py, px] = 1
                if ddy > ddx:
                    py += ystep
                    if err < 0:
                        px += 1
                        err += 2 * major
                else:
                    px += 1
                    if err < 0:
                        py += ystep
                        err += 2 * major
                err -= 2 * minor

        # scanline edge with the original rows and x extrapolated from the clipped line
        if ay == by:
            continue
        cy0 = ay
        cy1 = by
        if y1 != y2:
            cy0 = y1
            cy1 = y2
        num = (x2 - x1) << 16
        den = cy1 - cy0
        q = abs(num) // abs(den)
        step = q if (num < 0) == (den < 0) else -q
        if ay < by:
            ey0[n_edges] = ay
            ey1[n_edges] = by
            ex[n_edges] = (x1 << 16) + (ay - cy0) * step
        else:
            ey0[n_edges] = by
            ey1[n_edges] = ay
            ex[n_edges] = (x2 << 16) + (by - cy1) * step
        edx[n_edges] = step
        n_edges += 1

    if n_edges < 2:  # noqa: PLR2004
        return
    y_min = ey0[:n_edges].min()
    y_max = ey1[:n_edges].max()
    x_end = ex[:n_edges] + (ey1[:n_edges] - ey0[:n_edges]) * edx[:n_edges]
    x_lo = min(ex[:n_edges].min(), x_end.min())
    x_hi = max(ex[:n_edges].max(), x_end.max())
    if y_max < 0 or y_min >= height or x_hi < 0 or x_lo >= (width << 16):
        return
    y_max = min(y_max, height)
    xs = np.empty(n_edges, dtype=np.int64)
    for y in range(max(y_min, 0), y_max):
        m = 0
        for e in range(n_edges):
            if ey0[e] <= y < ey1[e]:
                xs[m] = ex[e] + edx[e] * (y - ey0[e])
                m += 1
        for i in range(1, m):
            v = xs[i]
            j = i - 1
            while j >= 0 and xs[j] > v:
                xs[j + 1] = xs[j]
                j -= 1
            xs[j + 1] = v
        for k in range(0, m - 1, 2):
            x1 = (xs[k] + 65535) >> 16
            x2 = xs[k + 1] >> 16
            if x1 < width and x2 >= 0:
                x1 = max(x1, 0)
                x2 = min(x2, width - 1)
                for x in range(x1, x2 + 1):
                    mask[y, x] = 1


def _fill_convex_poly(shape: tuple[int, int], pts: np.ndarray) -> np.ndarray:
    """Mask of an int polygon, bit identical to cv2.fillPoly(mask, [pts], (1,))."""
    mask = np.zeros(shape, dtype=np.uint8)
    _fill_poly(mask, np.ascontiguousarray(pts, dtype=np.int64))
    return mask


def polygon_mean(pred: np.ndarray, box: np.ndarray) -> float:
    """
    Mean of a probability map inside a polygon, as DB text detection scores a box.

    The polygon's bounding window is clipped to the map, the polygon is truncated to
    integers and filled like cv2.fillPoly, and the mean is taken over the filled pixels
    (0.0 when none).

    Parameters
    ----------
    pred : np.ndarray
        The (H, W) float32 probability map.
    box : np.ndarray
        The (4, 2) float polygon in (x, y) pixels.

    Returns
    -------
    float
        The mean of ``pred`` over the polygon.

    """
    h, w = pred.shape
    xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
    xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
    ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
    ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))
    window = pred[ymin : ymax + 1, xmin : xmax + 1]
    poly = (box - (xmin, ymin)).astype(np.int32)
    if FLAGS.FOUND_CV2:
        mask = np.zeros(window.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [poly], (1,))
        return cv2.mean(window, mask)[0]
    mask = _fill_convex_poly(window.shape, poly).astype(bool)
    if not mask.any():
        return 0.0
    return float(window[mask].mean(dtype=np.float64))
