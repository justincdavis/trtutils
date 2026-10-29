# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Postprocessing for YOLO-style oriented bounding box (OBB) detector heads."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from trtutils._log import LOG

from ._common import filter_class_scores

if TYPE_CHECKING:
    from trtutils.image.types import OBBDetection


def postprocess_yolo_obb(
    outputs: list[np.ndarray],
    ratios: list[tuple[float, float]],
    padding: list[tuple[float, float]],
    conf_thres: float | None = None,
    nms_iou_thres: float = 0.5,
    *,
    max_det: int = 300,
    verbose: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Postprocess YOLO-style OBB engine output.

    Expects one ``output0`` of shape ``(batch, 4 + nc + 1, N)``, last channel the
    angle in radians. Per-class probiou NMS (as ultralytics), results sorted by
    score in original image coordinates.

    Parameters
    ----------
    outputs : list[np.ndarray]
        Raw YOLO-style OBB engine outputs (single tensor ``output0``).
    ratios : list[tuple[float, float]]
        Preprocessing resize ratios per image.
    padding : list[tuple[float, float]]
        Preprocessing padding per image.
    conf_thres : float, optional
        Confidence threshold used to filter candidates.
        If not passed, no confidence filtering is applied.
    nms_iou_thres : float
        Probabilistic IoU threshold for the per-class rotated NMS.
    max_det : int
        Maximum number of detections kept per image, highest scores first.
        By default 300.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[np.ndarray]]
        One list per image: [bboxes (N,4), scores (N,), class_ids (N,), rboxes
        (N,5)] in original image coordinates. ``rboxes`` is the authoritative
        ``(cx, cy, w, h, angle)``; ``bboxes`` is the derived enclosing
        axis-aligned xyxy box of each rotated rect.

    Raises
    ------
    ValueError
        If an image was resized non-uniformly (e.g. ``resize_method="linear"``),
        which rotated boxes cannot be mapped back through.

    """
    if verbose:
        LOG.debug(f"OBB postprocess, output shape: {outputs[0].shape}")

    batch_size = outputs[0].shape[0]
    results = []
    for i in range(batch_size):
        if not np.isclose(ratios[i][0], ratios[i][1]):
            err_msg = "OBB decoding needs uniform scaling, use resize_method='letterbox'"
            raise ValueError(err_msg)
        rboxes, scores, class_ids = _decode_obb_core(
            outputs[0][i],
            ratios[i],
            padding[i],
            conf_thres,
        )
        keep_idx = _rotated_nms_indices(rboxes, scores, class_ids, nms_iou_thres, max_det)
        rboxes = rboxes[keep_idx]
        scores = scores[keep_idx]
        class_ids = class_ids[keep_idx]
        bboxes = _enclosing_bboxes(rboxes)
        results.append([bboxes, scores, class_ids, rboxes])
    return results


def _decode_obb_core(
    output: np.ndarray,
    ratios: tuple[float, float],
    padding: tuple[float, float],
    conf_thres: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ratio_width, ratio_height = ratios
    pad_x, pad_y = padding

    pred, scores, class_ids = filter_class_scores(output, slice(4, -1), conf_thres)
    cxcywh = pred[:, :4]
    angle = pred[:, -1]

    # letterbox scales uniformly, so the angle survives unchanged
    cx = (cxcywh[:, 0] - pad_x) / ratio_width
    cy = (cxcywh[:, 1] - pad_y) / ratio_height
    w = cxcywh[:, 2] / ratio_width
    h = cxcywh[:, 3] / ratio_height

    rboxes = np.stack((cx, cy, w, h, angle), axis=1)
    return rboxes, scores, class_ids


def _covariance(rboxes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # gaussian of each rotated box, covariance [[a, c], [c, b]]
    var_w = rboxes[:, 2] ** 2 / 12
    var_h = rboxes[:, 3] ** 2 / 12
    cos = np.cos(rboxes[:, 4])
    sin = np.sin(rboxes[:, 4])
    a = var_w * cos**2 + var_h * sin**2
    b = var_w * sin**2 + var_h * cos**2
    c = (var_w - var_h) * cos * sin
    return a, b, c


def _probiou(rboxes1: np.ndarray, rboxes2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Probabilistic IoU between two sets of rotated boxes (ultralytics ``batch_probiou``).

    Parameters
    ----------
    rboxes1 : np.ndarray
        The (N, 5) rotated boxes as (cx, cy, w, h, angle), angle in radians.
    rboxes2 : np.ndarray
        The (M, 5) rotated boxes as (cx, cy, w, h, angle), angle in radians.
    eps : float
        Numerical stability term.

    Returns
    -------
    np.ndarray
        The (N, M) similarity matrix in [0, 1].

    """
    a1, b1, c1 = (v[:, None] for v in _covariance(rboxes1))
    a2, b2, c2 = (v[None] for v in _covariance(rboxes2))
    dx = rboxes1[:, 0:1] - rboxes2[None, :, 0]
    dy = rboxes1[:, 1:2] - rboxes2[None, :, 1]
    return _probiou_terms(dx, dy, (a1, b1, c1), (a2, b2, c2), eps)


def _probiou_terms(
    dx: np.ndarray,
    dy: np.ndarray,
    cov1: tuple[np.ndarray, np.ndarray, np.ndarray],
    cov2: tuple[np.ndarray, np.ndarray, np.ndarray],
    eps: float,
) -> np.ndarray:
    # elementwise probiou from center offsets and covariances, any broadcastable shapes
    a1, b1, c1 = cov1
    a2, b2, c2 = cov2
    det = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2
    t1 = ((a1 + a2) * dy**2 + (b1 + b2) * dx**2) / (det + eps) * 0.25
    t2 = ((c1 + c2) * -dx * dy) / (det + eps) * 0.5
    dets = np.clip(a1 * b1 - c1**2, 0, None) * np.clip(a2 * b2 - c2**2, 0, None)
    t3 = np.log(det / (4 * np.sqrt(dets) + eps) + eps) * 0.5
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    return 1 - np.sqrt(1.0 - np.exp(-bd) + eps)


def _probiou_reach(nms_iou_thres: float, eps: float = 1e-7) -> float:
    # probiou >= thres needs bd <= bd_max, and bd >= |d|^2 / (4 * trace(cov1 + cov2)),
    # so pairs whose centers are farther apart than reach * trace cannot suppress
    lim = 1 + eps - (1 - nms_iou_thres) ** 2
    bd_max = -math.log(lim) if lim > 0 else math.inf
    return 4.04 * bd_max  # 1% margin for float32 rounding


def _rotated_nms_indices(
    rboxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    nms_iou_thres: float,
    max_det: int,
    block: int = 128,
    eps: float = 1e-7,
) -> np.ndarray:
    # ultralytics fast-nms: drop a box if any higher scored same-class box overlaps it
    order = np.argsort(-scores, kind="stable")
    # float32 as ultralytics computes it, about twice as fast as float64
    rboxes = rboxes[order].astype(np.float32)
    class_ids = class_ids[order]
    cov = _covariance(rboxes)
    x, y = rboxes[:, 0], rboxes[:, 1]
    # pairs too far apart to suppress skip the full formula
    trace = cov[0] + cov[1]
    reach = _probiou_reach(nms_iou_thres, eps)
    keep = [np.zeros(0, dtype=int)]
    kept = 0
    # small column blocks stop soon after max_det are kept, skipping most pairs
    for start in range(0, len(order), block):
        stop = min(start + block, len(order))
        dx = x[:stop, None] - x[None, start:stop]
        dy = y[:stop, None] - y[None, start:stop]
        near = dx * dx + dy * dy <= reach * (trace[:stop, None] + trace[None, start:stop])
        near &= np.arange(stop)[:, None] < np.arange(start, stop)[None]
        near &= class_ids[:stop, None] == class_ids[None, start:stop]
        rows, cols = np.nonzero(near)
        cols += start
        iou = _probiou_terms(
            x[rows] - x[cols],
            y[rows] - y[cols],
            tuple(v[rows] for v in cov),
            tuple(v[cols] for v in cov),
            eps,
        )
        hit = np.zeros(len(order), dtype=bool)
        hit[cols[iou >= nms_iou_thres]] = True
        cols_kept = start + np.flatnonzero(~hit[start:stop])
        keep.append(cols_kept)
        kept += len(cols_kept)
        if kept >= max_det:
            break
    return order[np.concatenate(keep)[:max_det]]


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
    keep = outputs[1] >= (conf_thres or 0.0)
    rboxes = outputs[3][keep].astype(float).tolist()
    return [
        ((cx, cy, w, h, angle), float(score), int(class_id))
        for (cx, cy, w, h, angle), score, class_id in zip(rboxes, outputs[1][keep], outputs[2][keep])
    ]
