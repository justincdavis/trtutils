# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Box helpers shared by the box-based postprocessors."""

from __future__ import annotations

import numpy as np

from trtutils._jit import register_jit, register_op
from trtutils.compat._libs import cv2


def nms_indices(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    conf_thres: float | None,
    nms_iou_thres: float,
    max_det: int | None = None,
) -> np.ndarray:
    """Class-aware NMS over xyxy boxes, returns the kept indices by score, at most max_det."""
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=int)

    xywh = np.empty_like(boxes, dtype=np.float32)
    xywh[:, :2] = boxes[:, :2]
    xywh[:, 2:] = boxes[:, 2:] - boxes[:, :2]
    return _nms_pick(xywh, scores, labels, conf_thres, nms_iou_thres, max_det)


def _nms_order(
    xywh: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    conf_thres: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boxes shifted by class, areas, and the stable score order, matching NMSBoxesBatched."""
    xywh = xywh.astype(np.float64)
    offset = labels.astype(np.float64) * (np.max(xywh[:, :2] + xywh[:, 2:]) + 1)
    xywh[:, :2] += offset[:, None]
    boxes = np.concatenate((xywh[:, :2], xywh[:, :2] + xywh[:, 2:]), axis=1)
    scores = scores.astype(np.float32).astype(np.float64)
    thres = 0.0 if conf_thres is None else conf_thres
    cand = np.flatnonzero(scores > thres)
    order = cand[np.argsort(-scores[cand], kind="stable")]
    boxes = boxes[order]
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    return boxes, areas, order


def _nms_indices_cv2(
    xywh: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    conf_thres: float | None,
    nms_iou_thres: float,
    max_det: int | None,
) -> np.ndarray:
    raw_idx = cv2.dnn.NMSBoxesBatched(
        xywh.tolist(),
        scores.astype(np.float32).tolist(),
        labels.astype(np.int32).tolist(),
        0.0 if conf_thres is None else conf_thres,
        nms_iou_thres,
    )
    # opencv returns the kept indices sorted by score
    return np.asarray(raw_idx, dtype=int).ravel()[:max_det]


@register_jit()
def _nms_suppress_indices(
    boxes: np.ndarray,
    areas: np.ndarray,
    iou_thres: float,
    limit: int,
) -> np.ndarray:
    """Return kept positions. Drop a later box when its iou is above the threshold."""
    n = boxes.shape[0]
    alive = np.ones(n, dtype=np.bool_)
    keep = np.empty(n, dtype=np.int64)
    kept = 0
    for i in range(n):
        if not alive[i]:
            continue
        keep[kept] = i
        kept += 1
        if kept == limit:
            break
        x1 = boxes[i, 0]
        y1 = boxes[i, 1]
        x2 = boxes[i, 2]
        y2 = boxes[i, 3]
        area_i = areas[i]
        for j in range(i + 1, n):
            if not alive[j]:
                continue
            iw = min(x2, boxes[j, 2]) - max(x1, boxes[j, 0])
            ih = min(y2, boxes[j, 3]) - max(y1, boxes[j, 1])
            if iw < 0.0:
                iw = 0.0
            if ih < 0.0:
                ih = 0.0
            inter = iw * ih
            union = area_i + areas[j] - inter
            iou = inter / union if union > 0.0 else 1.0
            if iou > iou_thres:
                alive[j] = False
    return keep[:kept]


def _nms_indices_numba(
    xywh: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    conf_thres: float | None,
    nms_iou_thres: float,
    max_det: int | None,
) -> np.ndarray:
    boxes, areas, order = _nms_order(xywh, scores, labels, conf_thres)
    limit = order.shape[0] + 1 if max_det is None else max_det
    chosen = _nms_suppress_indices.jitted(boxes, areas, float(nms_iou_thres), int(limit))
    return order[chosen].astype(int)


def _nms_indices_numpy(
    xywh: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    conf_thres: float | None,
    nms_iou_thres: float,
    max_det: int | None,
) -> np.ndarray:
    # same result as cv2.dnn.NMSBoxesBatched
    boxes, areas, order = _nms_order(xywh, scores, labels, conf_thres)
    alive = np.ones(order.shape[0], dtype=bool)
    keep: list[int] = []
    for i in range(order.shape[0]):
        if not alive[i]:
            continue
        keep.append(i)
        if len(keep) == max_det:
            break
        rest = slice(i + 1, None)
        wh = np.minimum(boxes[i, 2:], boxes[rest, 2:]) - np.maximum(boxes[i, :2], boxes[rest, :2])
        inter = np.clip(wh, 0, None).prod(axis=1)
        union = areas[i] + areas[rest] - inter
        # opencv scores a zero-area pair as iou 1
        iou = np.divide(inter, union, out=np.ones_like(inter), where=union > 0)
        alive[rest] &= iou <= nms_iou_thres
    return order[keep].astype(int)


_nms_pick = register_op(
    globals(),
    "_nms_pick",
    cv2=_nms_indices_cv2,
    numba=_nms_indices_numba,
    numpy=_nms_indices_numpy,
)


# not jitted: numba rejects axis= on max/argmax
def filter_class_scores(
    output: np.ndarray,
    cls: slice,
    conf_thres: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the (C, N) head columns whose best class score passes, as (pred (n, C), scores, ids)."""
    cls_scores = output[cls]
    scores = cls_scores.max(axis=0)
    if conf_thres is not None:
        keep = scores >= conf_thres
        output, cls_scores, scores = output[:, keep], cls_scores[:, keep], scores[keep]
    return output.T, scores, cls_scores.argmax(axis=0).astype(np.int64)


def cxcywh_to_xyxy(cxcywh: np.ndarray) -> np.ndarray:
    """Convert (N, >=4) cxcywh rows to (N, 4) xyxy."""
    half = cxcywh[:, 2:4] / 2
    return np.concatenate((cxcywh[:, :2] - half, cxcywh[:, :2] + half), axis=1)


def clip_boxes(
    boxes: np.ndarray,
    ratios: tuple[float, float],
    padding: tuple[float, float],
    input_size: tuple[int, int] | None,
) -> np.ndarray:
    """
    Clip xyxy boxes in original image coordinates to [0, width] / [0, height].

    The original size is recovered from the preprocessing, (input - 2 * padding) / ratio, so it
    can be off by under half a model pixel when the image was letterboxed. Without
    ``input_size`` boxes are only clipped at 0.
    """
    if input_size is None:
        return np.clip(boxes, 0, None)
    width = round((input_size[0] - 2 * padding[0]) / ratios[0])
    height = round((input_size[1] - 2 * padding[1]) / ratios[1])
    return np.clip(boxes, 0, [width, height, width, height])


def unletterbox_boxes(
    boxes: np.ndarray,
    ratios: tuple[float, float],
    padding: tuple[float, float],
    input_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Map xyxy boxes out of letterboxed coordinates, clipped to the original image."""
    ratio_width, ratio_height = ratios
    pad_x, pad_y = padding

    out = boxes.copy()
    out[:, 0] = (out[:, 0] - pad_x) / ratio_width
    out[:, 1] = (out[:, 1] - pad_y) / ratio_height
    out[:, 2] = (out[:, 2] - pad_x) / ratio_width
    out[:, 3] = (out[:, 3] - pad_y) / ratio_height
    return clip_boxes(out, ratios, padding, input_size)
