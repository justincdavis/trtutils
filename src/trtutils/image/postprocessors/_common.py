# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Box helpers shared by the box-based postprocessors."""

from __future__ import annotations

import cv2
import numpy as np


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

    raw_idx = cv2.dnn.NMSBoxesBatched(
        xywh.tolist(),
        scores.astype(np.float32).tolist(),
        labels.astype(np.int32).tolist(),
        0.0 if conf_thres is None else conf_thres,
        nms_iou_thres,
    )
    # opencv returns the kept indices sorted by score
    return np.asarray(raw_idx, dtype=int).ravel()[:max_det]


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


def unletterbox_boxes(
    boxes: np.ndarray,
    ratios: tuple[float, float],
    padding: tuple[float, float],
) -> np.ndarray:
    """Map xyxy boxes out of letterboxed coordinates, clipped at 0."""
    ratio_width, ratio_height = ratios
    pad_x, pad_y = padding

    out = boxes.copy()
    out[:, 0] = (out[:, 0] - pad_x) / ratio_width
    out[:, 1] = (out[:, 1] - pad_y) / ratio_height
    out[:, 2] = (out[:, 2] - pad_x) / ratio_width
    out[:, 3] = (out[:, 3] - pad_y) / ratio_height
    return np.clip(out, 0, None)
