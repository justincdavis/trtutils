# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Postprocessing for YOLO-style oriented bounding box (OBB) detector heads."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from trtutils._jit import register_jit
from trtutils._log import LOG

# rotated box (cx, cy, w, h, angle) in original image pixels, angle in radians
RBox = Tuple[float, float, float, float, float]
OBBDetection = Tuple[RBox, float, int]


def postprocess_yolo_obb(
    outputs: list[np.ndarray],
    ratios: list[tuple[float, float]],
    padding: list[tuple[float, float]],
    conf_thres: float | None = None,
    input_size: tuple[int, int] | None = None,  # noqa: ARG001
    nms_iou_thres: float = 0.5,
    *,
    no_copy: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Postprocess YOLO-style OBB engine output.

    Expects a single output ``output0`` of shape ``(batch, 4 + nc + 1, N)`` where the
    last channel is the rotation angle in radians. Decodes, unletterboxes, and
    deduplicates with per-class rotated NMS, then derives an axis-aligned enclosing
    box for each rotated rect.

    Parameters
    ----------
    outputs : list[np.ndarray]
        Raw YOLO-style OBB engine outputs (single tensor ``output0``).
    ratios : list[tuple[float, float]]
        Preprocessing resize ratios per image.
    padding : list[tuple[float, float]]
        Preprocessing padding per image.
    conf_thres : float, optional
        Confidence threshold used both to filter candidates and as the NMS
        score threshold. If not passed, no confidence filtering is applied.
    input_size : tuple[int, int] | None
        Unused.
    nms_iou_thres : float
        IoU threshold for the per-class rotated NMS.
    no_copy : bool, optional
        Kept for interface symmetry with the other postprocessors. NMS-based
        indexing always allocates new arrays, so this has no effect.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[np.ndarray]]
        One list per image: [bboxes (N,4), scores (N,), class_ids (N,), rboxes
        (N,5)] in original image coordinates. ``rboxes`` is the authoritative
        ``(cx, cy, w, h, angle)``; ``bboxes`` is the derived enclosing
        axis-aligned xyxy box of each rotated rect.

    """
    if verbose:
        LOG.debug(f"OBB postprocess, output shape: {outputs[0].shape}")

    batch_size = outputs[0].shape[0]
    results = []
    for i in range(batch_size):
        rboxes, scores, class_ids = _decode_obb_core(
            outputs[0][i],
            ratios[i],
            padding[i],
            conf_thres,
        )
        keep_idx = _rotated_nms_indices(rboxes, scores, class_ids, conf_thres, nms_iou_thres)
        rboxes = rboxes[keep_idx]
        scores = scores[keep_idx]
        class_ids = class_ids[keep_idx]
        bboxes = _enclosing_bboxes(rboxes)
        results.append([bboxes, scores, class_ids, rboxes])
    return results


# not jitted: numba rejects axis= on max/argmax
def _decode_obb_core(
    output: np.ndarray,
    ratios: tuple[float, float],
    padding: tuple[float, float],
    conf_thres: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ratio_width, ratio_height = ratios
    pad_x, pad_y = padding

    pred = output.T  # (N, 4 + nc + 1)
    nc = pred.shape[1] - 5
    cxcywh = pred[:, :4]
    cls_scores = pred[:, 4 : 4 + nc]
    scores = cls_scores.max(axis=1)
    class_ids = cls_scores.argmax(axis=1)
    angle = pred[:, -1]

    if conf_thres is not None:
        keep = scores >= conf_thres
        cxcywh = cxcywh[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        angle = angle[keep]

    # letterbox uses a uniform ratio (ratio_width == ratio_height), so the angle
    # is preserved exactly; only cx/cy/w/h need to be remapped out of network coords
    cx = (cxcywh[:, 0] - pad_x) / ratio_width
    cy = (cxcywh[:, 1] - pad_y) / ratio_height
    w = cxcywh[:, 2] / ratio_width
    h = cxcywh[:, 3] / ratio_height

    rboxes = np.stack((cx, cy, w, h, angle), axis=1)
    return rboxes, scores, class_ids.astype(np.int64)


def _rotated_nms_indices(
    rboxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    conf_thres: float | None,
    nms_iou_thres: float,
) -> np.ndarray:
    if rboxes.shape[0] == 0:
        return np.zeros((0,), dtype=int)
    thres = 0.0 if conf_thres is None else conf_thres

    keep: list[int] = []
    # opencv has no batched rotated NMS, so loop per class
    for cls_id in np.unique(class_ids):
        idx = np.where(class_ids == cls_id)[0]
        rects = [
            (
                (float(rboxes[j, 0]), float(rboxes[j, 1])),
                (float(rboxes[j, 2]), float(rboxes[j, 3])),
                float(np.degrees(rboxes[j, 4])),
            )
            for j in idx
        ]
        cls_scores = scores[idx].astype(np.float32).tolist()
        raw_idx = cv2.dnn.NMSBoxesRotated(rects, cls_scores, thres, nms_iou_thres)
        cls_keep = np.asarray(raw_idx, dtype=int).ravel()
        keep.extend(idx[cls_keep].tolist())
    return np.array(keep, dtype=int)


@register_jit(nogil=True)
def _enclosing_bboxes(rboxes: np.ndarray) -> np.ndarray:
    if rboxes.shape[0] == 0:
        return np.zeros((0, 4), dtype=rboxes.dtype)
    cx, cy, w, h, angle = rboxes[:, 0], rboxes[:, 1], rboxes[:, 2], rboxes[:, 3], rboxes[:, 4]
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    half_w_ext = (np.abs(w * cos_a) + np.abs(h * sin_a)) / 2
    half_h_ext = (np.abs(w * sin_a) + np.abs(h * cos_a)) / 2
    bboxes = np.stack(
        (cx - half_w_ext, cy - half_h_ext, cx + half_w_ext, cy + half_h_ext),
        axis=1,
    )
    return np.clip(bboxes, 0, None)


def get_obb_detections(
    outputs: list[list[np.ndarray]],
    conf_thres: float | None = None,
    *,
    verbose: bool | None = None,
) -> list[list[OBBDetection]]:
    """
    Convert postprocessed OBB outputs to human-friendly detections.

    Parameters
    ----------
    outputs : list[list[np.ndarray]]
        Postprocessed outputs per image, as returned by postprocess_yolo_obb.
    conf_thres : float, optional
        Confidence threshold; detections below are dropped.
    verbose : bool, optional
        If True, log extra debug information.

    Returns
    -------
    list[list[OBBDetection]]
        One list per image; each detection is ``((cx, cy, w, h, angle), score, class_id)``.

    """
    all_results = []
    for image_outputs in outputs:
        if verbose:
            LOG.debug(f"Decoding obb detections, num candidates: {len(image_outputs[0])}")
        all_results.append(_get_obb_detections_core(image_outputs, conf_thres))
    return all_results


# not jitted: numba can't type a list mixing 2D and 1D arrays
def _get_obb_detections_core(
    outputs: list[np.ndarray],
    conf_thres: float | None = None,
) -> list[OBBDetection]:
    if conf_thres is None:
        conf_thres = 0.0

    scores = outputs[1]
    class_ids = outputs[2]
    rboxes = outputs[3]

    results: list[OBBDetection] = []
    for idx in range(len(rboxes)):
        if scores[idx] >= conf_thres:
            entry = (
                (
                    float(rboxes[idx, 0]),
                    float(rboxes[idx, 1]),
                    float(rboxes[idx, 2]),
                    float(rboxes[idx, 3]),
                    float(rboxes[idx, 4]),
                ),
                float(scores[idx]),
                int(class_ids[idx]),
            )
            results.append(entry)

    return results
