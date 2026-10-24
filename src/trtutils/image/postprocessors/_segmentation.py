# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Postprocessing for YOLO-style instance segmentation heads."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from trtutils._log import LOG

from ._common import cxcywh_to_xyxy, nms_indices, unletterbox_boxes

# bbox as (x1, y1, x2, y2) in original image pixels
Bbox = Tuple[int, int, int, int]
# bbox, score, class_id, mask (uint8, original image resolution)
Segmentation = Tuple[Bbox, float, int, np.ndarray]


def postprocess_yolo_seg(
    outputs: list[np.ndarray],
    ratios: list[tuple[float, float]],
    padding: list[tuple[float, float]],
    conf_thres: float | None = None,
    nms_iou_thres: float = 0.5,
    mask_thres: float = 0.5,
    *,
    input_size: tuple[int, int],
    verbose: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Postprocess YOLO-style segmentation engine output.

    Expects two outputs: ``output0`` of shape ``(batch, 4 + nc + nm, N)`` and
    ``output1`` (the mask prototypes) of shape ``(batch, nm, mh, mw)``. Decodes,
    filters by confidence, deduplicates with class-aware NMS in letterboxed
    coordinates, then unletterboxes the surviving boxes and decodes per-detection
    masks at original image resolution.

    Parameters
    ----------
    outputs : list[np.ndarray]
        Raw YOLO-style segmentation engine outputs ``[output0, output1]``.
    ratios : list[tuple[float, float]]
        Preprocessing resize ratios per image.
    padding : list[tuple[float, float]]
        Preprocessing padding per image.
    conf_thres : float, optional
        Confidence threshold used both to filter candidates and as the NMS score
        threshold. If not passed, no confidence filtering is applied.
    nms_iou_thres : float
        IoU threshold for the class-aware NMS.
    mask_thres : float
        Threshold applied to the sigmoid mask logits to produce the binary mask.
    input_size : tuple[int, int]
        The network input (width, height), used to map the mask prototypes back
        to the letterboxed content region.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[np.ndarray]]
        One list per image: [bboxes (N,4), scores (N,), class_ids (N,), masks
        (N,H,W) uint8] in original image coordinates/resolution.

    """
    if verbose:
        LOG.debug(f"Seg postprocess, output shapes: {outputs[0].shape}, {outputs[1].shape}")

    batch_size = outputs[0].shape[0]
    results = []
    for i in range(batch_size):
        proto = outputs[1][i]  # (nm, mh, mw)
        boxes, scores, class_ids, coeffs = _decode_seg_core(
            outputs[0][i],
            proto.shape[0],
            conf_thres,
        )

        keep_idx = nms_indices(boxes, scores, class_ids, conf_thres, nms_iou_thres)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        class_ids = class_ids[keep_idx]
        coeffs = coeffs[keep_idx]

        boxes = unletterbox_boxes(boxes, ratios[i], padding[i])
        masks = _decode_masks(coeffs, proto, boxes, ratios[i], padding[i], input_size, mask_thres)

        results.append([boxes, scores, class_ids, masks])
    return results


# not jitted: numba rejects axis= on max/argmax
def _decode_seg_core(
    output: np.ndarray,
    nm: int,
    conf_thres: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred = output.T  # (N, 4 + nc + nm)
    nc = pred.shape[1] - 4 - nm
    cxcywh = pred[:, :4]
    cls_scores = pred[:, 4 : 4 + nc]
    scores = cls_scores.max(axis=1)
    class_ids = cls_scores.argmax(axis=1)
    coeffs = pred[:, 4 + nc :]

    if conf_thres is not None:
        keep = scores >= conf_thres
        cxcywh = cxcywh[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        coeffs = coeffs[keep]

    boxes = cxcywh_to_xyxy(cxcywh)
    return boxes, scores, class_ids.astype(np.int64), coeffs


def _decode_masks(
    coeffs: np.ndarray,
    proto: np.ndarray,
    boxes: np.ndarray,
    ratios: tuple[float, float],
    padding: tuple[float, float],
    input_size: tuple[int, int],
    mask_thres: float,
) -> np.ndarray:
    _, mh, mw = proto.shape
    ratio_width, ratio_height = ratios
    pad_x, pad_y = padding

    input_w, input_h = input_size

    orig_w = max(round((input_w - 2 * pad_x) / ratio_width), 1)
    orig_h = max(round((input_h - 2 * pad_y) / ratio_height), 1)

    if coeffs.shape[0] == 0:
        return np.zeros((0, orig_h, orig_w), dtype=np.uint8)

    # map the letterbox content region into proto-grid coordinates
    stride_x = input_w / mw
    stride_y = input_h / mh
    x0 = max(int(np.floor(pad_x / stride_x)), 0)
    y0 = max(int(np.floor(pad_y / stride_y)), 0)
    x1 = max(min(int(np.ceil((input_w - pad_x) / stride_x)), mw), x0 + 1)
    y1 = max(min(int(np.ceil((input_h - pad_y) / stride_y)), mh), y0 + 1)

    # original pixel -> proto cell index, as an exact resize of the content region
    scale_x = (x1 - x0) / orig_w
    scale_y = (y1 - y0) / orig_h
    off_x = x0 + 0.5 * scale_x - 0.5
    off_y = y0 + 0.5 * scale_y - 0.5

    masks = np.zeros((coeffs.shape[0], orig_h, orig_w), dtype=np.uint8)
    for idx in range(coeffs.shape[0]):
        bx1 = max(int(np.floor(boxes[idx, 0])), 0)
        by1 = max(int(np.floor(boxes[idx, 1])), 0)
        bx2 = min(int(np.ceil(boxes[idx, 2])), orig_w)
        by2 = min(int(np.ceil(boxes[idx, 3])), orig_h)
        if bx2 <= bx1 or by2 <= by1:
            continue

        # proto cells the box samples from, with one cell of bilinear margin
        cx1 = max(int(np.floor(scale_x * bx1 + off_x)), 0)
        cx2 = min(int(np.ceil(scale_x * (bx2 - 1) + off_x)) + 1, mw)
        cy1 = max(int(np.floor(scale_y * by1 + off_y)), 0)
        cy2 = min(int(np.ceil(scale_y * (by2 - 1) + off_y)) + 1, mh)
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        logits = np.tensordot(coeffs[idx], proto[:, cy1:cy2, cx1:cx2], axes=1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        affine = np.array(
            [
                [scale_x, 0.0, scale_x * bx1 + off_x - cx1],
                [0.0, scale_y, scale_y * by1 + off_y - cy1],
            ],
            dtype=np.float32,
        )
        resized = cv2.warpAffine(
            probs.astype(np.float32),
            affine,
            (bx2 - bx1, by2 - by1),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        masks[idx, by1:by2, bx1:bx2] = resized >= mask_thres

    return masks


def get_segmentations(
    outputs: list[list[np.ndarray]],
    conf_thres: float | None = None,
    *,
    verbose: bool | None = None,
) -> list[list[Segmentation]]:
    """
    Convert postprocessed segmentation outputs to human-friendly segmentations.

    Parameters
    ----------
    outputs : list[list[np.ndarray]]
        Postprocessed outputs per image, as returned by postprocess_yolo_seg.
    conf_thres : float, optional
        Confidence threshold; detections below are dropped.
    verbose : bool, optional
        If True, log extra debug information.

    Returns
    -------
    list[list[Segmentation]]
        One list per image; each segmentation is ((x1, y1, x2, y2), score, class_id, mask).

    """
    all_results = []
    for image_outputs in outputs:
        if verbose:
            LOG.debug(f"Decoding segmentations, num candidates: {len(image_outputs[0])}")
        all_results.append(_get_segmentations_core(image_outputs, conf_thres))
    return all_results


def _get_segmentations_core(
    outputs: list[np.ndarray],
    conf_thres: float | None = None,
) -> list[Segmentation]:
    if conf_thres is None:
        conf_thres = 0.0

    bboxes = outputs[0]
    scores = outputs[1]
    class_ids = outputs[2]
    masks = outputs[3]

    results: list[Segmentation] = []
    for idx in range(len(bboxes)):
        if scores[idx] >= conf_thres:
            x1, y1, x2, y2 = bboxes[idx]
            entry = (
                (int(x1), int(y1), int(x2), int(y2)),
                float(scores[idx]),
                int(class_ids[idx]),
                masks[idx],
            )
            results.append(entry)

    return results
