# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Postprocessing for PaddleOCR (PP-OCR) style text detection and recognition models.

DB box extraction and CTC greedy decoding follow PaddleOCR's ``DBPostProcess`` and
``CTCLabelDecode`` (including the Clipper round-join unclip), with PaddleOCR's defaults.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

_MIN_BOX_SIDE = 3
_SAME_LINE_PX = 10


def _mini_box(rect: tuple) -> tuple[np.ndarray, float]:
    """Corner points of a cv2.minAreaRect result ordered TL, TR, BR, BL, and its short side."""
    p = sorted(cv2.boxPoints(rect).tolist(), key=lambda pt: pt[0])
    tl, bl = (p[0], p[1]) if p[1][1] > p[0][1] else (p[1], p[0])
    tr, br = (p[2], p[3]) if p[3][1] > p[2][1] else (p[3], p[2])
    return np.array([tl, tr, br, bl], dtype=np.float32), min(rect[1])


def _box_score(pred: np.ndarray, box: np.ndarray) -> float:
    """Mean probability of ``pred`` inside the polygon ``box``."""
    h, w = pred.shape
    xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
    xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
    ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
    ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    shifted = box - np.array([xmin, ymin], dtype=box.dtype)
    cv2.fillPoly(mask, [shifted.reshape(-1, 2).astype(np.int32)], (1,))
    return cv2.mean(pred[ymin : ymax + 1, xmin : xmax + 1], mask)[0]


def _round_half_away(value: float) -> int:
    return int(value - 0.5) if value < 0 else int(value + 0.5)


def _unclip(box: np.ndarray, unclip_ratio: float) -> tuple:
    """
    Expand a polygon outward by DB's unclip distance (area * ratio / perimeter).

    Reproduces the integer round-join polygon offset (Clipper library semantics,
    arc tolerance 0.25) used by the reference implementation, so boxes match it
    pixel-for-pixel. Returns the cv2.minAreaRect of the expanded polygon.
    """
    pts = box.astype(np.float64)
    nxt = np.roll(pts, -1, axis=0)
    area = abs(float(np.sum(pts[:, 0] * nxt[:, 1] - nxt[:, 0] * pts[:, 1]))) / 2
    delta = area * unclip_ratio / float(np.sum(np.linalg.norm(nxt - pts, axis=1)))

    # integer polygon, consecutive duplicates removed, counter-clockwise
    ipts = [(int(x), int(y)) for x, y in box.tolist()]
    ipts = [p for i, p in enumerate(ipts) if p != ipts[i - 1]] or ipts[:1]
    n = len(ipts)
    signed = sum(
        ipts[j][0] * ipts[(j + 1) % n][1] - ipts[(j + 1) % n][0] * ipts[j][1] for j in range(n)
    )
    if signed < 0:
        ipts = ipts[::-1]
    if delta <= 0 or n < 2:  # noqa: PLR2004
        return cv2.minAreaRect(np.array(ipts, dtype=np.int32))

    steps = min(math.pi / math.acos(1 - min(0.25, delta * 0.25) / delta), delta * math.pi)
    sin_s, cos_s = math.sin(2 * math.pi / steps), math.cos(2 * math.pi / steps)
    normals = []
    for j in range(n):
        dx, dy = ipts[(j + 1) % n][0] - ipts[j][0], ipts[(j + 1) % n][1] - ipts[j][1]
        length = math.hypot(dx, dy)
        normals.append((dy / length, -dx / length) if length else (0.0, 0.0))

    def at(src: tuple[int, int], normal: tuple[float, float]) -> tuple[int, int]:
        return (
            _round_half_away(src[0] + normal[0] * delta),
            _round_half_away(src[1] + normal[1] * delta),
        )

    out: list[tuple[int, int]] = []
    for j in range(n):
        nk, nj = normals[j - 1], normals[j]
        sin_a = nk[0] * nj[1] - nj[0] * nk[1]
        cos_a = nk[0] * nj[0] + nk[1] * nj[1]
        if abs(sin_a * delta) < 1.0 and cos_a > 0:
            out.append(at(ipts[j], nk))
            continue
        sin_a = max(-1.0, min(1.0, sin_a))
        if sin_a * delta < 0:
            out += [at(ipts[j], nk), ipts[j], at(ipts[j], nj)]
            continue
        x, y = nk
        for _ in range(
            max(_round_half_away(steps / (2 * math.pi) * abs(math.atan2(sin_a, cos_a))), 1)
        ):
            out.append(at(ipts[j], (x, y)))
            x, y = x * cos_s - sin_s * y, x * sin_s + y * cos_s
        out.append(at(ipts[j], nj))
    return cv2.minAreaRect(np.array(out, dtype=np.int32))


def _order_clockwise(points: np.ndarray) -> np.ndarray:
    x_sorted = points[np.argsort(points[:, 0]), :]
    left = x_sorted[:2][np.argsort(x_sorted[:2, 1])]
    right = x_sorted[2:][np.argsort(x_sorted[2:, 1])]
    return np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)


def sort_text_boxes(
    boxes: np.ndarray,
    scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort text boxes in reading order: top to bottom, then left to right.

    Boxes whose top-left corners are within 10 pixels vertically are treated
    as the same line.

    Parameters
    ----------
    boxes : np.ndarray
        The boxes, shape (N, 4, 2).
    scores : np.ndarray, optional
        Per-box scores, reordered alongside the boxes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The sorted boxes and scores.

    """
    if scores is None:
        scores = np.zeros(len(boxes), dtype=np.float32)
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][0][1], boxes[i][0][0]))
    for i in range(len(order) - 1):
        for j in range(i, -1, -1):
            a, b = boxes[order[j]][0], boxes[order[j + 1]][0]
            if abs(b[1] - a[1]) < _SAME_LINE_PX and b[0] < a[0]:
                order[j], order[j + 1] = order[j + 1], order[j]
            else:
                break
    idx = np.array(order, dtype=np.int64)
    return boxes[idx].reshape(-1, 4, 2), np.asarray(scores)[idx]


def postprocess_text_detection(
    output: np.ndarray,
    image_shape: tuple[int, int],
    thresh: float = 0.3,
    box_thresh: float = 0.6,
    max_candidates: int = 1000,
    unclip_ratio: float = 1.5,
    *,
    use_dilation: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract text boxes from the probability map of a DB text detection model.

    Parameters
    ----------
    output : np.ndarray
        The model output probability map, shape (1, 1, H, W) or (H, W).
    image_shape : tuple[int, int]
        The (height, width) of the original image the boxes are scaled to.
    thresh : float
        Binarization threshold for the probability map. Default is 0.3.
    box_thresh : float
        Minimum mean probability inside a box to keep it. Default is 0.6.
    max_candidates : int
        Maximum number of contours to consider. Default is 1000.
    unclip_ratio : float
        How far to expand each shrunk text kernel. Default is 1.5.
    use_dilation : bool
        Whether to dilate the binary map with a 2x2 kernel. Default is False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The boxes, shape (N, 4, 2) float32 corner points (TL, TR, BR, BL) in
        reading order, and their scores, shape (N,).

    """
    height, width = int(output.shape[-2]), int(output.shape[-1])
    pred = output.reshape(height, width)
    src_h, src_w = image_shape
    bitmap = (pred > thresh).astype(np.uint8)
    if use_dilation:
        bitmap = cv2.dilate(bitmap, np.ones((2, 2), dtype=np.uint8))
    contours = cv2.findContours(bitmap * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    boxes: list[np.ndarray] = []
    scores: list[float] = []
    for contour in contours[:max_candidates]:
        points, short_side = _mini_box(cv2.minAreaRect(contour))
        if short_side < _MIN_BOX_SIDE:
            continue
        score = _box_score(pred, points)
        if score < box_thresh:
            continue
        box, short_side = _mini_box(_unclip(points, unclip_ratio))
        if short_side < _MIN_BOX_SIDE + 2:
            continue
        box[:, 0] = np.clip(np.round(box[:, 0] / width * src_w), 0, src_w)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * src_h), 0, src_h)
        box = _order_clockwise(box.astype(np.int32).astype(np.float32))
        box[:, 0] = np.clip(box[:, 0], 0, src_w - 1).astype(np.int32)
        box[:, 1] = np.clip(box[:, 1], 0, src_h - 1).astype(np.int32)
        if (
            int(np.linalg.norm(box[0] - box[1])) <= _MIN_BOX_SIDE
            or int(np.linalg.norm(box[0] - box[3])) <= _MIN_BOX_SIDE
        ):
            continue
        boxes.append(box)
        scores.append(score)

    if not boxes:
        return np.zeros((0, 4, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)
    return sort_text_boxes(np.stack(boxes), np.array(scores, dtype=np.float32))


def postprocess_text_recognition(
    output: np.ndarray,
    characters: Sequence[str],
) -> list[tuple[str, float]]:
    """
    Greedy CTC decode the output of a text recognition model.

    Parameters
    ----------
    output : np.ndarray
        The per-timestep class probabilities, shape (N, T, C).
    characters : Sequence[str]
        The class labels, where index 0 is the CTC blank.

    Returns
    -------
    list[tuple[str, float]]
        The decoded text and its mean character confidence for each batch item.
        An empty prediction has confidence 0.0.

    """
    indices = output.argmax(axis=2)
    probs = output.max(axis=2)
    results: list[tuple[str, float]] = []
    for idx, prob in zip(indices, probs):
        keep = np.ones(len(idx), dtype=bool)
        keep[1:] = idx[1:] != idx[:-1]
        keep &= idx != 0
        text = "".join(characters[i] for i in idx[keep])
        score = float(np.mean(prob[keep])) if keep.any() else 0.0
        results.append((text, score))
    return results
