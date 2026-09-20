# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Postprocessor for ultralytics pose estimation models.

Functions
---------
:func:`postprocess_yolo_pose`
    Postprocess the output of a YOLO-pose model.
:func:`get_poses`
    Get the poses from postprocessed pose estimation outputs.

"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from trtutils._jit import register_jit
from trtutils._log import LOG

# bbox as (x1, y1, x2, y2) in original image pixels
Bbox = Tuple[int, int, int, int]
# (bbox, score, keypoints (K, 3) as x, y, visibility in original image coords)
Pose = Tuple[Bbox, float, np.ndarray]


def postprocess_yolo_pose(
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
    Postprocess ultralytics YOLO-pose engine output.

    Expects a single output of shape (batch, 5 + K*3, N) with rows (cx, cy, w, h,
    score, kpt_x_0, kpt_y_0, kpt_vis_0, ...); the number of keypoints K is derived
    from the output shape, never hardcoded. Boxes and keypoint x/y are remapped out
    of letterboxed network-input coordinates, filtered by confidence, and deduplicated
    with NMS (single-class, since ultralytics pose only predicts "person").

    Parameters
    ----------
    outputs : list[np.ndarray]
        Raw YOLO-pose engine outputs, a single (batch, 5 + K*3, N) tensor.
    ratios : list[tuple[float, float]]
        Preprocessing resize ratios per image.
    padding : list[tuple[float, float]]
        Preprocessing padding per image.
    conf_thres : float, optional
        Confidence threshold used to filter candidate poses. If None, no filter
        is applied.
    input_size : tuple[int, int] | None
        Unused, kept for interface symmetry with the other postprocessors.
    nms_iou_thres : float
        IoU threshold for NMS.
    no_copy : bool, optional
        Kept for interface symmetry with the other postprocessors. NMS-based
        indexing always allocates new arrays, so this has no effect.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[np.ndarray]]
        One list per image: [bboxes (N,4), scores (N,), class_ids (N,) all zero,
        keypoints (N,K,3)], all in original image coordinates. Keypoint visibility
        (channel 2) is passed through unchanged.

    """
    if verbose:
        LOG.debug(f"Pose postprocess, output shape: {outputs[0].shape}")

    batch_size = outputs[0].shape[0]
    results = []
    for i in range(batch_size):
        pred = outputs[0][i].T  # (N, 5 + K*3)
        num_kpts = (pred.shape[1] - 5) // 3

        boxes = _decode_boxes_core(pred)
        scores = pred[:, 4].copy()
        kpts = pred[:, 5:].reshape(-1, num_kpts, 3).copy()

        if conf_thres is not None:
            mask = scores >= conf_thres
            boxes = boxes[mask]
            scores = scores[mask]
            kpts = kpts[mask]

        keep_idx = _nms_indices(boxes, scores, conf_thres, nms_iou_thres)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        kpts = kpts[keep_idx]

        boxes, kpt_xy = _unletterbox_pose_core(
            boxes,
            np.ascontiguousarray(kpts[:, :, :2]),
            ratios[i],
            padding[i],
        )
        kpts[:, :, :2] = kpt_xy
        class_ids = np.zeros(boxes.shape[0], dtype=int)

        results.append([boxes, scores, class_ids, kpts])
    return results


@register_jit(nogil=True)
def _decode_boxes_core(pred: np.ndarray) -> np.ndarray:
    """Convert cxcywh (network coords) to xyxy for every candidate row."""
    cx = pred[:, 0]
    cy = pred[:, 1]
    w = pred[:, 2]
    h = pred[:, 3]

    boxes = np.empty((pred.shape[0], 4), dtype=pred.dtype)
    boxes[:, 0] = cx - w / 2
    boxes[:, 1] = cy - h / 2
    boxes[:, 2] = cx + w / 2
    boxes[:, 3] = cy + h / 2
    return boxes


@register_jit(nogil=True)
def _unletterbox_pose_core(
    boxes: np.ndarray,
    kpt_xy: np.ndarray,
    ratios: tuple[float, float],
    padding: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    ratio_width, ratio_height = ratios
    pad_x, pad_y = padding

    adjusted_boxes = boxes.copy()
    adjusted_boxes[:, 0] = (adjusted_boxes[:, 0] - pad_x) / ratio_width
    adjusted_boxes[:, 1] = (adjusted_boxes[:, 1] - pad_y) / ratio_height
    adjusted_boxes[:, 2] = (adjusted_boxes[:, 2] - pad_x) / ratio_width
    adjusted_boxes[:, 3] = (adjusted_boxes[:, 3] - pad_y) / ratio_height
    adjusted_boxes = np.clip(adjusted_boxes, 0, None)

    # keypoints are NOT clipped - an occluded keypoint can legitimately land just
    # outside the frame, the visibility channel is what tells the caller to ignore it
    adjusted_kpt_xy = kpt_xy.copy()
    adjusted_kpt_xy[:, :, 0] = (adjusted_kpt_xy[:, :, 0] - pad_x) / ratio_width
    adjusted_kpt_xy[:, :, 1] = (adjusted_kpt_xy[:, :, 1] - pad_y) / ratio_height

    return adjusted_boxes, adjusted_kpt_xy


def _nms_indices(
    boxes: np.ndarray,
    scores: np.ndarray,
    conf_thres: float | None,
    nms_iou_thres: float,
) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=int)

    xywh = np.empty_like(boxes, dtype=np.float32)
    xywh[:, 0] = boxes[:, 0]
    xywh[:, 1] = boxes[:, 1]
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]

    # single-class (person) - batched NMS with all-zero labels is plain NMS
    labels = np.zeros(boxes.shape[0], dtype=np.int32)
    score_thres = conf_thres if conf_thres is not None else 0.0

    raw_idx = cv2.dnn.NMSBoxesBatched(
        xywh.tolist(),
        scores.astype(np.float32).tolist(),
        labels.tolist(),
        score_thres,
        nms_iou_thres,
    )
    return np.asarray(raw_idx, dtype=int).ravel()


def get_poses(
    outputs: list[list[np.ndarray]],
    conf_thres: float | None = None,
    *,
    verbose: bool | None = None,
) -> list[list[Pose]]:
    """
    Convert postprocessed pose outputs to human-friendly poses.

    Applies an optional confidence filter. Input format is one list per image of
    [bboxes (N,4), scores (N,), class_ids (N,), keypoints (N,K,3)].

    Parameters
    ----------
    outputs : list[list[np.ndarray]]
        Postprocessed outputs per image (unified format plus keypoints).
    conf_thres : float, optional
        Confidence threshold; poses below are dropped.
    verbose : bool, optional
        If True, log extra debug information.

    Returns
    -------
    list[list[Pose]]
        One list per image; each pose is (bbox, score, keypoints).

    """
    all_results = []
    for image_outputs in outputs:
        if verbose:
            LOG.debug(f"Decoding poses, num candidates: {len(image_outputs[0])}")
        all_results.append(_get_poses_core(image_outputs, conf_thres))
    return all_results


def _get_poses_core(
    outputs: list[np.ndarray],
    conf_thres: float | None = None,
) -> list[Pose]:
    # not jitted: the outputs list mixes 1D/2D/3D arrays (scores, bboxes, keypoints),
    # numba can't unify that into one reflected-list element type
    if conf_thres is None:
        conf_thres = 0.0

    bboxes = outputs[0]
    scores = outputs[1]
    kpts = outputs[3]

    results: list[Pose] = []
    for idx in range(len(bboxes)):
        if scores[idx] >= conf_thres:
            x1, y1, x2, y2 = bboxes[idx]
            entry = (
                (int(x1), int(y1), int(x2), int(y2)),
                float(scores[idx]),
                kpts[idx],
            )
            results.append(entry)

    return results
