# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Postprocessors for pose estimation models.

Functions
---------
:func:`postprocess_yolo_pose`
    Postprocess the output of a YOLO-pose model.
:func:`postprocess_rtmo_pose`
    Postprocess the output of an RTMO model.
:func:`postprocess_yunet_pose`
    Postprocess the output of a YuNet face detector.
:func:`get_poses`
    Get the poses from postprocessed pose estimation outputs.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from trtutils._log import LOG

from ._common import clip_boxes, cxcywh_to_xyxy, nms_indices, unletterbox_boxes
from ._cpu_detection import _get_detections_core

if TYPE_CHECKING:
    from trtutils.image.types import Pose


def postprocess_yolo_pose(
    outputs: list[np.ndarray],
    ratios: list[tuple[float, float]],
    padding: list[tuple[float, float]],
    conf_thres: float | None = None,
    nms_iou_thres: float = 0.5,
    *,
    max_det: int = 300,
    input_size: tuple[int, int] | None = None,
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
    max_det : int
        Maximum number of detections kept per image, highest scores first.
        By default 300.
    input_size : tuple[int, int] | None
        Model input (width, height), used with ratios/padding to clip boxes and keypoints to
        the image. Without it boxes are only clipped at 0 and keypoints are not clipped.
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

        labels = np.zeros(len(scores), dtype=int)
        keep_idx = nms_indices(boxes, scores, labels, conf_thres, nms_iou_thres, max_det)
        boxes = unletterbox_boxes(boxes[keep_idx], ratios[i], padding[i], input_size)
        scores = scores[keep_idx]
        kpts = pred[keep_idx, 5:].reshape(-1, num_kpts, 3)
        kpts = _unletterbox_keypoints(kpts, ratios[i], padding[i], input_size)
        results.append([boxes, scores, labels[keep_idx], kpts])
    return results


def postprocess_rtmo_pose(
    outputs: list[np.ndarray],
    ratios: list[tuple[float, float]],
    padding: list[tuple[float, float]],
    conf_thres: float | None = None,
    nms_iou_thres: float = 0.5,
    *,
    max_det: int = 300,
    input_size: tuple[int, int] | None = None,
    verbose: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Postprocess RTMO engine output.

    Expects ``dets`` (batch, N, 5) as xyxy + score and ``keypoints`` (batch, N, K, 3) as
    x, y, score, both in input pixels. The engine already ran NMS and pads unused rows
    with zero scores; the NMS here only applies a stricter ``nms_iou_thres``.

    Parameters
    ----------
    outputs : list[np.ndarray]
        Raw RTMO engine outputs, [dets, keypoints].
    ratios : list[tuple[float, float]]
        Preprocessing resize ratios per image.
    padding : list[tuple[float, float]]
        Preprocessing padding per image.
    conf_thres : float, optional
        Confidence threshold used to filter poses. If None, no filter is applied.
    nms_iou_thres : float
        IoU threshold for NMS.
    max_det : int
        Maximum number of poses kept per image, highest scores first.
        By default 300.
    input_size : tuple[int, int] | None
        Model input (width, height), used with ratios/padding to clip boxes and keypoints to
        the image. Without it boxes are only clipped at 0 and keypoints are not clipped.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[np.ndarray]]
        One list per image: [bboxes (N,4), scores (N,), class_ids (N,) all zero,
        keypoints (N,K,3)], all in original image coordinates. Keypoint scores
        (channel 2) are passed through unchanged.

    """
    if verbose:
        LOG.debug(f"RTMO postprocess, output shapes: {outputs[0].shape}, {outputs[1].shape}")

    results = []
    for i in range(outputs[0].shape[0]):
        dets, kpts = outputs[0][i], outputs[1][i]
        if conf_thres is not None:
            mask = dets[:, 4] >= conf_thres
            dets, kpts = dets[mask], kpts[mask]
        scores = dets[:, 4]
        labels = np.zeros(len(scores), dtype=int)
        keep_idx = nms_indices(dets[:, :4], scores, labels, conf_thres, nms_iou_thres, max_det)
        boxes = unletterbox_boxes(dets[keep_idx, :4], ratios[i], padding[i], input_size)
        kpts = _unletterbox_keypoints(kpts[keep_idx], ratios[i], padding[i], input_size)
        results.append([boxes, scores[keep_idx], labels[keep_idx], kpts])
    return results


# the strides of YuNet's heads, in the order of its outputs
_YUNET_STRIDES = (8, 16, 32)


def postprocess_yunet_pose(
    outputs: list[np.ndarray],
    ratios: list[tuple[float, float]],
    padding: list[tuple[float, float]],
    conf_thres: float | None = None,
    nms_iou_thres: float = 0.5,
    *,
    max_det: int = 300,
    input_size: tuple[int, int] | None = None,
    verbose: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Postprocess YuNet face detector engine output, as OpenCV's ``FaceDetectorYN`` does.

    Expects the twelve raw heads in :attr:`PoseEstimatorOutputSchema.YUNET` order:
    ``cls`` and ``obj`` (batch, N, 1), ``bbox`` (batch, N, 4) and ``kps`` (batch, N, 10)
    for the strides 8, 16 and 32, with N the cells of each stride's row-major grid. The
    score is ``sqrt(cls * obj)`` with both clipped to [0, 1]; boxes and the five landmarks
    are decoded from their grid cell, then single-class NMS keeps the best faces.

    Parameters
    ----------
    outputs : list[np.ndarray]
        Raw YuNet engine outputs, the twelve heads in schema order.
    ratios : list[tuple[float, float]]
        Preprocessing resize ratios per image.
    padding : list[tuple[float, float]]
        Preprocessing padding per image.
    conf_thres : float, optional
        Confidence threshold used to filter faces. If None, no filter is applied.
    nms_iou_thres : float
        IoU threshold for NMS.
    max_det : int
        Maximum number of faces kept per image, highest scores first.
        By default 300.
    input_size : tuple[int, int] | None
        Model input (width, height), which sets the grid of each stride. Required.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[np.ndarray]]
        One list per image: [bboxes (N,4), scores (N,), class_ids (N,) all zero,
        keypoints (N,5,3)], all in original image coordinates. The keypoints are the right
        eye, left eye, nose tip, right and left mouth corner; YuNet gives them no scores of
        their own, so channel 2 repeats the face score.

    Raises
    ------
    ValueError
        If ``input_size`` is not given.

    """
    if input_size is None:
        err_msg = "YuNet postprocessing needs input_size to rebuild the grid of each stride"
        raise ValueError(err_msg)
    if verbose:
        LOG.debug(f"YuNet postprocess, input size: {input_size}")

    # (column, row) of every cell and its stride, in the order of the concatenated heads
    cells, strides = [], []
    for stride in _YUNET_STRIDES:
        cols, rows = input_size[0] // stride, input_size[1] // stride
        row, col = np.divmod(np.arange(rows * cols), cols)
        cells.append(np.stack([col, row], axis=1))
        strides.append(np.full(rows * cols, stride))
    all_cells = np.concatenate(cells).astype(np.float32)
    all_strides = np.concatenate(strides).astype(np.float32)[:, None]

    num = len(_YUNET_STRIDES)
    results = []
    for i in range(outputs[0].shape[0]):
        cls, obj, bbox, kps = (
            np.concatenate([out[i] for out in outputs[k * num : (k + 1) * num]]) for k in range(4)
        )
        scores = np.sqrt(np.clip(cls[:, 0], 0, 1) * np.clip(obj[:, 0], 0, 1))
        cell, stride = all_cells, all_strides
        if conf_thres is not None:
            mask = scores >= conf_thres
            scores, bbox, kps = scores[mask], bbox[mask], kps[mask]
            cell, stride = cell[mask], stride[mask]

        centers = (cell + bbox[:, :2]) * stride
        half_wh = np.exp(bbox[:, 2:]) * stride / 2
        boxes = np.concatenate([centers - half_wh, centers + half_wh], axis=1)
        kpts = np.empty((len(scores), 5, 3), dtype=np.float32)
        kpts[:, :, :2] = (kps.reshape(-1, 5, 2) + cell[:, None]) * stride[:, :, None]
        kpts[:, :, 2] = scores[:, None]

        labels = np.zeros(len(scores), dtype=int)
        keep_idx = nms_indices(boxes, scores, labels, conf_thres, nms_iou_thres, max_det)
        boxes = unletterbox_boxes(boxes[keep_idx], ratios[i], padding[i], input_size)
        kpts = _unletterbox_keypoints(kpts[keep_idx], ratios[i], padding[i], input_size)
        results.append([boxes, scores[keep_idx], labels[keep_idx], kpts])
    return results


def _unletterbox_keypoints(
    kpts: np.ndarray,
    ratios: tuple[float, float],
    padding: tuple[float, float],
    input_size: tuple[int, int] | None,
) -> np.ndarray:
    """Map (N, K, 3) keypoints out of letterboxed coordinates in place, clipped like boxes."""
    ratio_width, ratio_height = ratios
    kpts[:, :, 0] = (kpts[:, :, 0] - padding[0]) / ratio_width
    kpts[:, :, 1] = (kpts[:, :, 1] - padding[1]) / ratio_height
    if input_size is not None:
        # ultralytics clips keypoints to the image like boxes
        xy = clip_boxes(kpts[:, :, [0, 1, 0, 1]], ratios, padding, input_size)
        kpts[:, :, :2] = xy[:, :, :2]
    return kpts


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
    kpts = outputs[3][outputs[1] >= (conf_thres or 0.0)]
    dets = _get_detections_core(outputs[:3], conf_thres)
    return [(bbox, score, k) for (bbox, score, _), k in zip(dets, kpts)]
