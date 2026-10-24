# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Postprocessor for YOLO-style pose estimation models.

Functions
---------
:func:`postprocess_yolo_pose`
    Postprocess the output of a YOLO-pose model.
:func:`get_poses`
    Get the poses from postprocessed pose estimation outputs.

"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from trtutils._log import LOG

from ._common import cxcywh_to_xyxy, nms_indices, unletterbox_boxes

# bbox as (x1, y1, x2, y2) in original image pixels
Bbox = Tuple[int, int, int, int]
# (bbox, score, keypoints (K, 3) as x, y, visibility in original image coords)
Pose = Tuple[Bbox, float, np.ndarray]


def postprocess_yolo_pose(
    outputs: list[np.ndarray],
    ratios: list[tuple[float, float]],
    padding: list[tuple[float, float]],
    conf_thres: float | None = None,
    nms_iou_thres: float = 0.5,
    *,
    verbose: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Postprocess YOLO-style pose head engine output.

    Expects one output of shape (batch, 5 + K*3, N); K is read from the shape.
    Single-class NMS, results in original image coordinates.

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
    nms_iou_thres : float
        IoU threshold for NMS.
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

        scores = pred[:, 4]
        if conf_thres is not None:
            pred = pred[scores >= conf_thres]
            scores = pred[:, 4]
        boxes = cxcywh_to_xyxy(pred)

        keep_idx = nms_indices(
            boxes, scores, np.zeros(len(scores), dtype=int), conf_thres, nms_iou_thres
        )
        boxes = unletterbox_boxes(boxes[keep_idx], ratios[i], padding[i])
        scores = scores[keep_idx]
        kpts = pred[keep_idx, 5:].reshape(-1, num_kpts, 3)

        # keypoints are not clipped, occluded ones can land outside the frame
        ratio_width, ratio_height = ratios[i]
        kpts[:, :, 0] = (kpts[:, :, 0] - padding[i][0]) / ratio_width
        kpts[:, :, 1] = (kpts[:, :, 1] - padding[i][1]) / ratio_height
        class_ids = np.zeros(boxes.shape[0], dtype=int)

        results.append([boxes, scores, class_ids, kpts])
    return results


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
    # not jitted: mixed 1D/2D/3D arrays, numba can't unify types
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
