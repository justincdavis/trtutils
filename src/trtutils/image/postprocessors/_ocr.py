# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np

from trtutils._log import LOG

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from trtutils.image.types import Text, TextBox

_MIN_BOX_SIDE = 3
_SIDE_SQ_LIMIT = (_MIN_BOX_SIDE + 1) ** 2
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
    # plain floats: numpy calls on a 4x2 box cost more than the math
    xs, ys = box.T.tolist()
    xmin = min(max(math.floor(min(xs)), 0), w - 1)
    xmax = min(max(math.ceil(max(xs)), 0), w - 1)
    ymin = min(max(math.floor(min(ys)), 0), h - 1)
    ymax = min(max(math.ceil(max(ys)), 0), h - 1)
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    cv2.fillPoly(mask, [(box - (xmin, ymin)).astype(np.int32)], (1,))
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
    # float64 shoelace area and perimeter, summed in numpy's order for small arrays
    pts = box.tolist()
    nxt = pts[1:] + pts[:1]
    area = abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, nxt))) / 2
    perimeter = sum(
        math.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))
        for (x0, y0), (x1, y1) in zip(pts, nxt)
    )
    delta = area * unclip_ratio / perimeter

    # integer polygon, consecutive duplicates removed, counter-clockwise
    ipts = [(int(x), int(y)) for x, y in pts]
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
        # the arc: the hot loop, with at() and _round_half_away() inlined
        px, py = ipts[j]
        x, y = nk
        for _ in range(
            max(_round_half_away(steps / (2 * math.pi) * abs(math.atan2(sin_a, cos_a))), 1)
        ):
            vx, vy = px + x * delta, py + y * delta
            out.append(
                (
                    int(vx - 0.5) if vx < 0 else int(vx + 0.5),
                    int(vy - 0.5) if vy < 0 else int(vy + 0.5),
                )
            )
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


def _scaled_boxes(
    kept: list[np.ndarray],
    kept_scores: list[float],
    size: np.ndarray,
    limit: np.ndarray,
) -> tuple[list[np.ndarray], list[float]]:
    """Scale unclipped boxes to the source image, dropping those with a side under 4 px."""
    boxes: list[np.ndarray] = []
    scores: list[float] = []
    if not kept:
        return boxes, scores
    # rounded and clipped, so already integral like the reference's int32 boxes; float32
    # elementwise ops, so scaling all boxes at once gives the per-box bits
    scaled = np.clip(np.round(np.stack(kept) / size * limit), 0, limit)
    for raw, score in zip(scaled, kept_scores):
        box = np.clip(_order_clockwise(raw), 0, limit - 1)
        (x0, y0), (x1, y1), _, (x3, y3) = box.tolist()
        # integral corners: int(norm) <= 3 is exactly a squared length under 16
        if (x1 - x0) ** 2 + (y1 - y0) ** 2 < _SIDE_SQ_LIMIT or (x3 - x0) ** 2 + (
            y3 - y0
        ) ** 2 < _SIDE_SQ_LIMIT:
            continue
        boxes.append(box)
        scores.append(score)
    return boxes, scores


def _text_boxes(
    pred: np.ndarray,
    image_shape: tuple[int, int],
    thresh: float,
    box_thresh: float,
    max_candidates: int,
    unclip_ratio: float,
    *,
    use_dilation: bool,
    bitmap: np.ndarray | None = None,
    wait: Callable[[], None] | None = None,
) -> list[np.ndarray]:
    """
    Boxes (N, 4, 2) and scores (N,) in reading order from one (H, W) probability map.

    ``bitmap`` is ``pred > thresh`` when already computed, and ``wait`` blocks
    until ``pred`` is readable (it may still be copying from the device).
    """
    height, width = pred.shape
    src_h, src_w = image_shape
    if bitmap is None:
        bitmap = (pred > thresh).astype(np.uint8)
    if use_dilation:
        bitmap = cv2.dilate(bitmap, np.ones((2, 2), dtype=np.uint8))
    contours = cv2.findContours(bitmap * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    size = np.array([width, height], np.float32)
    limit = np.array([src_w, src_h], np.float32)
    kept: list[np.ndarray] = []
    kept_scores: list[float] = []
    for contour in contours[:max_candidates]:
        points, short_side = _mini_box(cv2.minAreaRect(contour))
        if short_side < _MIN_BOX_SIDE:
            continue
        if wait is not None:
            wait()
            wait = None
        score = _box_score(pred, points)
        if score < box_thresh:
            continue
        box, short_side = _mini_box(_unclip(points, unclip_ratio))
        if short_side < _MIN_BOX_SIDE + 2:
            continue
        kept.append(box)
        kept_scores.append(score)

    boxes, scores = _scaled_boxes(kept, kept_scores, size, limit)
    if not boxes:
        return [np.zeros((0, 4, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)]
    return list(sort_text_boxes(np.stack(boxes), np.array(scores, dtype=np.float32)))


def postprocess_text_detection(
    outputs: list[np.ndarray],
    ratios: list[tuple[float, float]],
    padding: list[tuple[float, float]],
    thresh: float = 0.3,
    box_thresh: float = 0.6,
    max_candidates: int = 1000,
    unclip_ratio: float = 1.5,
    *,
    use_dilation: bool = False,
    bitmaps: np.ndarray | None = None,
    wait: Callable[[], None] | None = None,
    no_copy: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Postprocess outputs from a DB text detection network.

    Follows PaddleOCR's ``DBPostProcess`` (including the Clipper round-join
    unclip, reproduced pixel-for-pixel) with PaddleOCR's defaults.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from a DB text detection network, the first being the
        probability map of shape (B, 1, H, W).
    ratios : list[tuple[float, float]]
        The (width, height) resize ratio of each image from preprocessing.
    padding : list[tuple[float, float]]
        The (right, bottom) padding of each image from preprocessing.
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
    bitmaps : np.ndarray, optional
        ``outputs[0] > thresh`` as 0/1 bytes when already computed (by the CUDA postprocessor).
    wait : Callable[[], None], optional
        Blocks until ``outputs[0]`` is readable, called before it is first read for scoring.
    no_copy : bool, optional
        Unused, the boxes are always new arrays.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[np.ndarray]]
        The postprocessed outputs per image, each containing [boxes, scores]:
        boxes of shape (N, 4, 2) float32 corner points (TL, TR, BR, BL) in
        original image coordinates and reading order, and scores of shape (N,).

    """
    if verbose:
        LOG.debug(f"Text detection postprocess, output shape: {outputs[0].shape}")

    maps = outputs[0]
    height, width = int(maps.shape[-2]), int(maps.shape[-1])
    results = []
    for i, ((ratio_w, ratio_h), (pad_x, pad_y)) in enumerate(zip(ratios, padding)):
        valid_h, valid_w = height - int(pad_y), width - int(pad_x)
        pred = maps[i].reshape(height, width)[:valid_h, :valid_w]
        image_shape = (round(valid_h / ratio_h), round(valid_w / ratio_w))
        results.append(
            _text_boxes(
                pred,
                image_shape,
                thresh,
                box_thresh,
                max_candidates,
                unclip_ratio,
                use_dilation=use_dilation,
                bitmap=None
                if bitmaps is None
                else bitmaps[i].reshape(height, width)[:valid_h, :valid_w],
                wait=wait,
            )
        )
        # the first image's scoring waited already
        wait = None
    return results


def get_text_boxes(
    outputs: list[list[np.ndarray]],
    *,
    verbose: bool | None = None,
) -> list[list[TextBox]]:
    """
    Get the text boxes from postprocessed text detection outputs.

    Parameters
    ----------
    outputs : list[list[np.ndarray]]
        The postprocessed outputs per image from a text detection network.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[TextBox]]
        The text boxes per image in reading order, one TextBox (box, score) per
        region, where box is the (4, 2) float32 corner points (TL, TR, BR, BL)
        in original image coordinates.

    """
    if verbose:
        LOG.debug("Getting text boxes")

    return [[(box, float(score)) for box, score in zip(boxes, scores)] for boxes, scores in outputs]


def postprocess_text_recognition(
    outputs: list[np.ndarray],
    *,
    no_copy: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Postprocess outputs from a CTC text recognition network.

    Takes the greedy (best path) class and its probability at every timestep,
    as PaddleOCR's ``CTCLabelDecode`` does.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from a CTC text recognition network, the first being the
        per-timestep class probabilities of shape (B, T, C).
    no_copy : bool, optional
        Unused, the results are always new arrays.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[np.ndarray]]
        The postprocessed outputs per image, each containing [indices, probs]:
        the best class index and its probability per timestep, of shape (T,).

    """
    if verbose:
        LOG.debug(f"Text recognition postprocess, output shape: {outputs[0].shape}")

    indices = outputs[0].argmax(axis=2)
    probs = np.take_along_axis(outputs[0], indices[..., None], 2)[..., 0]
    return [[np.asarray(idx), np.asarray(prob)] for idx, prob in zip(indices, probs)]


def get_texts(
    outputs: list[list[np.ndarray]],
    characters: Sequence[str],
    *,
    verbose: bool | None = None,
) -> list[Text]:
    """
    Get the texts from postprocessed text recognition outputs by CTC decoding.

    Repeated classes are collapsed and blanks dropped.

    Parameters
    ----------
    outputs : list[list[np.ndarray]]
        The postprocessed outputs per image from a text recognition network.
    characters : Sequence[str]
        The class labels, where index 0 is the CTC blank.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[Text]
        One Text (text, confidence) per image, where the confidence is the
        mean probability of the kept characters
        (0.0 for an empty text).

    """
    if verbose:
        LOG.debug("Getting texts")

    results = []
    for idx, prob in outputs:
        keep = idx != 0
        keep[1:] &= idx[1:] != idx[:-1]
        text = "".join(characters[i] for i in idx[keep])
        results.append((text, float(prob[keep].mean()) if keep.any() else 0.0))
    return results
