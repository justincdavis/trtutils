# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Box helpers shared by the box-based postprocessors."""

from __future__ import annotations

import numpy as np

from trtutils._flags import FLAGS
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
    if not FLAGS.FOUND_CV2:
        return _nms_indices_numpy(xywh, scores, labels, conf_thres, nms_iou_thres, max_det)

    raw_idx = cv2.dnn.NMSBoxesBatched(
        xywh.tolist(),
        scores.astype(np.float32).tolist(),
        labels.astype(np.int32).tolist(),
        0.0 if conf_thres is None else conf_thres,
        nms_iou_thres,
    )
    # opencv returns the kept indices sorted by score
    return np.asarray(raw_idx, dtype=int).ravel()[:max_det]


def _nms_indices_numpy(
    xywh: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    conf_thres: float | None,
    nms_iou_thres: float,
    max_det: int | None,
) -> np.ndarray:
    # same result as cv2.dnn.NMSBoxesBatched, ~10x slower
    xywh = xywh.astype(np.float64)
    offset = labels.astype(np.float64) * (np.max(xywh[:, :2] + xywh[:, 2:]) + 1)
    xywh[:, :2] += offset[:, None]
    boxes = np.concatenate((xywh[:, :2], xywh[:, :2] + xywh[:, 2:]), axis=1)
    scores = scores.astype(np.float32).astype(np.float64)
    thres = 0.0 if conf_thres is None else conf_thres

    # stable descending sort over the strictly passing scores, like opencv
    cand = np.flatnonzero(scores > thres)
    order = cand[np.argsort(-scores[cand], kind="stable")]
    boxes = boxes[order]
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)

    alive = np.ones(order.shape[0], dtype=bool)
    keep: list[int] = []
    for i in range(order.shape[0]):
        if not alive[i]:
            continue
        keep.append(i)
        if len(keep) == max_det:
            break
        rest = slice(i + 1, None)
        inter = np.prod(
            np.clip(
                np.minimum(boxes[i, 2:], boxes[rest, 2:])
                - np.maximum(boxes[i, :2], boxes[rest, :2]),
                0,
                None,
            ),
            axis=1,
        )
        union = areas[i] + areas[rest] - inter
        # opencv scores a zero-area pair as iou 1
        zero = union <= 0
        iou = np.where(zero, 1.0, inter / np.where(zero, 1.0, union))
        alive[rest] &= iou <= nms_iou_thres
    return order[keep].astype(int)


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
