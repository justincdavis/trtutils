# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import numpy as np
from cv2ext.bboxes import nms

from trtutils._jit import register_jit
from trtutils._log import LOG


def postprocess_yolov10(
    outputs: list[np.ndarray],
    ratios: list[tuple[float, float]],
    padding: list[tuple[float, float]],
    conf_thres: float | None = None,
    input_size: tuple[int, int] | None = None,  # noqa: ARG001
    *,
    no_copy: bool | None = None,
    verbose: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Postprocess YOLO-v10 engine output.

    Expects a single output of shape (batch, N, 6) with rows (x1, y1, x2, y2, score,
    class_id); unletterboxes and returns the unified format per image.

    Inputs:
        outputs: Raw YOLO-v10 engine outputs.
        ratios: Preprocessing resize ratios per image.
        padding: Preprocessing padding per image.
        conf_thres: Optional extra confidence filter (use if model used a low cutoff).
        input_size: Unused.
        no_copy: If True, return buffers without copying (overwritten by later preprocessing).
        verbose: If True, log extra debug information.

    Outputs:
        One list per image, each [bboxes (N,4), scores (N,), class_ids (N,)] in original coords.

    """
    if verbose:
        LOG.debug(f"V10 postprocess, output shape: {outputs[0].shape}")

    batch_size = outputs[0].shape[0]
    results = []
    for i in range(batch_size):
        batch_outputs = [out[i : i + 1] for out in outputs]
        result = _postprocess_yolov10_core(
            batch_outputs,
            ratios[i],
            padding[i],
            conf_thres=conf_thres,
            no_copy=no_copy,
        )
        results.append(result)
    return results


@register_jit(nogil=True)
def _postprocess_yolov10_core(
    outputs: list[np.ndarray],
    ratios: tuple[float, float],
    padding: tuple[float, float],
    conf_thres: float | None = None,
    *,
    no_copy: bool | None = None,
) -> list[np.ndarray]:
    ratio_width, ratio_height = ratios
    pad_x, pad_y = padding
    output = outputs[0]

    bboxes = output[0, :, :4]
    scores = output[0, :, 4]
    class_ids = output[0, :, 5].astype(int)

    if conf_thres is not None:
        mask = scores >= conf_thres
        bboxes = bboxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

    adjusted_bboxes = bboxes
    adjusted_bboxes[:, 0] = (adjusted_bboxes[:, 0] - pad_x) / ratio_width
    adjusted_bboxes[:, 1] = (adjusted_bboxes[:, 1] - pad_y) / ratio_height
    adjusted_bboxes[:, 2] = (adjusted_bboxes[:, 2] - pad_x) / ratio_width
    adjusted_bboxes[:, 3] = (adjusted_bboxes[:, 3] - pad_y) / ratio_height
    adjusted_bboxes = np.clip(adjusted_bboxes, 0, None)

    if no_copy:
        return [adjusted_bboxes, scores, class_ids]
    return [adjusted_bboxes.copy(), scores.copy(), class_ids.copy()]


def get_detections(
    outputs: list[list[np.ndarray]],
    conf_thres: float | None = None,
    nms_iou_thres: float = 0.5,
    *,
    extra_nms: bool | None = None,
    agnostic_nms: bool | None = None,
    verbose: bool | None = None,
) -> list[list[tuple[tuple[int, int, int, int], float, int]]]:
    """
    Convert postprocessed unified outputs to human-friendly detections.

    Applies an optional confidence filter and optional CPU NMS. Input format is one
    list per image of [bboxes (N,4), scores (N,), class_ids (N,)].

    Inputs:
        outputs: Postprocessed outputs per image (unified format).
        conf_thres: Confidence threshold; detections below are dropped.
        nms_iou_thres: IoU threshold for optional extra NMS.
        extra_nms: If True, run CPU NMS on the final detections.
        agnostic_nms: If True, use class-agnostic NMS when extra_nms is True.
        verbose: If True, log extra debug information.

    Outputs:
        One list per image; each detection is ((x1, y1, x2, y2), score, class_id).

    """
    all_results = []
    for image_outputs in outputs:
        if verbose:
            LOG.debug(f"Decoding detections, num candidates: {len(image_outputs[0])}")

        results = _get_detections_core(image_outputs, conf_thres)

        if extra_nms:
            results = nms(results, iou_threshold=nms_iou_thres, agnostic=agnostic_nms)

        all_results.append(results)

    return all_results


@register_jit(nogil=True)
def _get_detections_core(
    outputs: list[np.ndarray],
    conf_thres: float | None = None,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    if conf_thres is None:
        conf_thres = 0.0

    bboxes = outputs[0]
    scores = outputs[1]
    class_ids = outputs[2]

    results: list[tuple[tuple[int, int, int, int], float, int]] = []
    for idx in range(len(bboxes)):
        if scores[idx] >= conf_thres:
            x1, y1, x2, y2 = bboxes[idx]
            entry = (
                (int(x1), int(y1), int(x2), int(y2)),
                float(scores[idx]),
                int(class_ids[idx]),
            )
            results.append(entry)

    return results


def postprocess_efficient_nms(
    outputs: list[np.ndarray],
    ratios: list[tuple[float, float]],
    padding: list[tuple[float, float]],
    conf_thres: float | None = None,
    input_size: tuple[int, int] | None = None,  # noqa: ARG001
    *,
    no_copy: bool | None = None,
    verbose: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Postprocess EfficientNMS plugin output.

    Raw outputs are [num_dets, bboxes, scores, class_ids], each with batch dim at 0;
    unletterboxes and returns the unified format per image.

    Inputs:
        outputs: Raw EfficientNMS outputs.
        ratios: Preprocessing resize ratios per image.
        padding: Preprocessing padding per image.
        conf_thres: Optional extra confidence filter (use if plugin used a low cutoff).
        input_size: Unused.
        no_copy: If True, return buffers without copying (overwritten by later preprocessing).
        verbose: If True, log extra debug information.

    Outputs:
        One list per image, each [bboxes (N,4), scores (N,), class_ids (N,)] in original coords.

    """
    if verbose:
        LOG.debug(f"EfficientNMS postprocess, raw bboxes shape: {outputs[1].shape}")

    batch_size = outputs[0].shape[0]
    results = []
    for i in range(batch_size):
        batch_outputs = [
            outputs[0][i : i + 1],
            outputs[1][i : i + 1],
            outputs[2][i : i + 1],
            outputs[3][i : i + 1],
        ]
        result = _postprocess_efficient_nms_core(
            batch_outputs,
            ratios[i],
            padding[i],
            conf_thres,
            no_copy=no_copy,
        )
        results.append(result)
    return results


@register_jit(nogil=True)
def _postprocess_efficient_nms_core(
    outputs: list[np.ndarray],
    ratios: tuple[float, float],
    padding: tuple[float, float],
    conf_thres: float | None = None,
    *,
    no_copy: bool | None = None,
) -> list[np.ndarray]:
    num_dets, bboxes, scores, class_ids = outputs
    ratio_width, ratio_height = ratios
    pad_x, pad_y = padding

    num_det_id = int(num_dets[0])
    bboxes = bboxes[0, :num_det_id]
    scores = scores[0, :num_det_id]
    class_ids = class_ids[0, :num_det_id]

    if conf_thres is not None:
        mask = scores >= conf_thres
        bboxes = bboxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

    adjusted_bboxes = bboxes
    adjusted_bboxes[:, 0] = (adjusted_bboxes[:, 0] - pad_x) / ratio_width
    adjusted_bboxes[:, 1] = (adjusted_bboxes[:, 1] - pad_y) / ratio_height
    adjusted_bboxes[:, 2] = (adjusted_bboxes[:, 2] - pad_x) / ratio_width
    adjusted_bboxes[:, 3] = (adjusted_bboxes[:, 3] - pad_y) / ratio_height

    adjusted_bboxes = np.clip(adjusted_bboxes, 0, None)

    if no_copy:
        return [adjusted_bboxes, scores, class_ids]
    return [adjusted_bboxes.copy(), scores.copy(), class_ids.copy()]


def postprocess_rfdetr(
    outputs: list[np.ndarray],
    ratios: list[tuple[float, float]],
    padding: list[tuple[float, float]],
    conf_thres: float | None = None,
    input_size: tuple[int, int] | None = None,
    *,
    no_copy: bool | None = None,
    verbose: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Postprocess RF-DETR output.

    Expects [dets, labels]: dets (batch, num_queries, 4) in normalized [cx, cy, w, h];
    labels (batch, num_queries, num_classes) as logits (class IDs 1-indexed). Converts
    to unified format in original coords.

    Inputs:
        outputs: Raw RF-DETR outputs [dets, labels].
        ratios: Preprocessing resize ratios per image.
        padding: Preprocessing padding per image.
        conf_thres: Confidence threshold to filter detections.
        input_size: Model input (width, height) to denormalize bboxes; default 640x640.
        no_copy: If True, return buffers without copying (overwritten by later preprocessing).
        verbose: If True, log extra debug information.

    Outputs:
        One list per image, each [bboxes (N,4), scores (N,), class_ids (N,)] in original coords.

    """
    if verbose:
        LOG.debug(f"RF-DETR postprocess, dets shape: {outputs[0].shape}")

    batch_size = outputs[0].shape[0]
    results = []
    for i in range(batch_size):
        batch_outputs = [out[i : i + 1] for out in outputs]
        result = _postprocess_rfdetr_core(
            batch_outputs,
            ratios[i],
            padding[i],
            conf_thres=conf_thres,
            input_size=input_size,
            no_copy=no_copy,
        )
        results.append(result)
    return results


@register_jit(nogil=True)
def _postprocess_rfdetr_core(
    outputs: list[np.ndarray],
    ratios: tuple[float, float],
    padding: tuple[float, float],
    conf_thres: float | None = None,
    input_size: tuple[int, int] | None = None,
    *,
    no_copy: bool | None = None,
) -> list[np.ndarray]:
    ratio_width, ratio_height = ratios
    pad_x, pad_y = padding
    bboxes = outputs[0][0, :, :]
    logits = outputs[1][0, :, :]

    probs = 1.0 / (1.0 + np.exp(-logits))
    num_queries, num_classes = probs.shape[0], probs.shape[1]
    probs_flat = probs.reshape(-1)
    num_select = min(num_queries, len(probs_flat))

    topk_indices = np.argsort(probs_flat)[-num_select:][::-1]
    topk_scores = probs_flat[topk_indices]
    topk_box_indices = topk_indices // num_classes
    topk_class_ids = topk_indices % num_classes
    selected_bboxes = bboxes[topk_box_indices]

    if conf_thres is not None:
        mask = topk_scores >= conf_thres
        selected_bboxes = selected_bboxes[mask]
        topk_scores = topk_scores[mask]
        topk_class_ids = topk_class_ids[mask]

    topk_class_ids = topk_class_ids - 1  # 1-indexed in model

    if input_size is not None:
        input_w, input_h = input_size
    else:
        input_w = input_h = 640

    cx = selected_bboxes[:, 0] * input_w
    cy = selected_bboxes[:, 1] * input_h
    w = selected_bboxes[:, 2] * input_w
    h = selected_bboxes[:, 3] * input_h
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    adjusted_x1 = (x1 - pad_x) / ratio_width
    adjusted_y1 = (y1 - pad_y) / ratio_height
    adjusted_x2 = (x2 - pad_x) / ratio_width
    adjusted_y2 = (y2 - pad_y) / ratio_height
    adjusted_bboxes = np.stack([adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2], axis=1)
    adjusted_bboxes = np.clip(adjusted_bboxes, 0, None)

    if no_copy:
        return [adjusted_bboxes, topk_scores, topk_class_ids.astype(int)]
    return [adjusted_bboxes.copy(), topk_scores.copy(), topk_class_ids.astype(int).copy()]


def postprocess_detr(
    outputs: list[np.ndarray],
    ratios: list[tuple[float, float]],
    padding: list[tuple[float, float]],
    conf_thres: float | None = None,
    input_size: tuple[int, int] | None = None,
    *,
    no_copy: bool | None = None,
    verbose: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Postprocess DETR-based output (DEIM, RT-DETR, D-FINE).

    Expects [scores, labels, boxes] with boxes (batch, num_queries, 4) as [x1, y1, x2,
    y2]. For models using orig_target_sizes, boxes are already in original image coords;
    no coordinate transform is applied.

    Inputs:
        outputs: Raw DETR outputs [scores, labels, boxes].
        ratios: Preprocessing resize ratios per image.
        padding: Preprocessing padding per image.
        conf_thres: Confidence threshold to filter detections.
        input_size: Unused for standard DETR.
        no_copy: If True, return buffers without copying (overwritten by later preprocessing).
        verbose: If True, log extra debug information.

    Outputs:
        One list per image, each [bboxes (N,4), scores (N,), class_ids (N,)].

    """
    if verbose:
        LOG.debug(f"DETR postprocess, labels shape: {outputs[0].shape}")

    batch_size = outputs[0].shape[0]
    results = []
    for i in range(batch_size):
        batch_outputs = [out[i : i + 1] for out in outputs]
        result = _postprocess_detr_core(
            batch_outputs,
            ratios[i],
            padding[i],
            conf_thres=conf_thres,
            input_size=input_size,
            no_copy=no_copy,
        )
        results.append(result)
    return results


@register_jit(nogil=True)
def _postprocess_detr_core(
    outputs: list[np.ndarray],
    ratios: tuple[float, float],  # noqa: ARG001
    padding: tuple[float, float],  # noqa: ARG001
    conf_thres: float | None = None,
    input_size: tuple[int, int] | None = None,  # noqa: ARG001
    *,
    no_copy: bool | None = None,
) -> list[np.ndarray]:
    scores = outputs[0]
    labels = outputs[1]
    boxes = outputs[2]
    class_ids = labels[0, :].astype(int)
    bboxes = boxes[0, :, :]
    scores_arr = scores[0, :]

    if conf_thres is not None:
        mask = scores_arr >= conf_thres
        class_ids = class_ids[mask]
        bboxes = bboxes[mask]
        scores_arr = scores_arr[mask]

    finite_mask = np.all(np.isfinite(bboxes), axis=1)
    if not np.all(finite_mask):
        bboxes = bboxes[finite_mask]
        scores_arr = scores_arr[finite_mask]
        class_ids = class_ids[finite_mask]

    adjusted_bboxes = np.clip(bboxes, 0, None)

    if no_copy:
        return [adjusted_bboxes, scores_arr, class_ids]
    return [adjusted_bboxes.copy(), scores_arr.copy(), class_ids.copy()]


def postprocess_detr_lbs(
    outputs: list[np.ndarray],
    ratios: list[tuple[float, float]],
    padding: list[tuple[float, float]],
    conf_thres: float | None = None,
    input_size: tuple[int, int] | None = None,
    *,
    no_copy: bool | None = None,
    verbose: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Postprocess DETR-style output when the engine returns (labels, boxes, scores).

    Used for DEIM and D-FINE; reorders and delegates to postprocess_detr.

    Inputs:
        outputs: Raw outputs in (labels, boxes, scores) order.
        ratios: Preprocessing resize ratios per image.
        padding: Preprocessing padding per image.
        conf_thres: Confidence threshold to filter detections.
        input_size: Unused.
        no_copy: If True, return buffers without copying (overwritten by later preprocessing).
        verbose: If True, log extra debug information.

    Outputs:
        One list per image, each [bboxes (N,4), scores (N,), class_ids (N,)].

    """
    reordered = [outputs[2], outputs[0], outputs[1]]
    return postprocess_detr(
        reordered,
        ratios=ratios,
        padding=padding,
        conf_thres=conf_thres,
        input_size=input_size,
        no_copy=no_copy,
        verbose=verbose,
    )


def postprocess_rtdetrv3(
    outputs: list[np.ndarray],
    ratios: list[tuple[float, float]],
    padding: list[tuple[float, float]],
    conf_thres: float | None = None,
    input_size: tuple[int, int] | None = None,  # noqa: ARG001
    *,
    no_copy: bool | None = None,
    verbose: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Postprocess RT-DETR v3 (PaddlePaddle export) output.

    Two tensors: combined_dets (total_N, 6) with rows (class_id, score, x1, y1, x2,
    y2) in original image coords, and num_dets_per_image [batch_size]. No coordinate
    transform is applied.

    Inputs:
        outputs: [combined_dets, num_dets_per_image].
        ratios: Unused (boxes already in image coords).
        padding: Unused.
        conf_thres: Confidence threshold to filter detections.
        input_size: Unused.
        no_copy: If True, return buffers without copying (overwritten by later preprocessing).
        verbose: If True, log extra debug information.

    Outputs:
        One list per image, each [bboxes (N,4), scores (N,), class_ids (N,)].

    """
    if verbose:
        LOG.debug(f"RT-DETR v3 postprocess, detections shape: {outputs[0].shape}")

    combined_dets = outputs[0]
    num_dets_raw = outputs[1]
    num_dets = np.array([int(num_dets_raw)]) if num_dets_raw.ndim == 0 else num_dets_raw

    batch_size = len(num_dets)
    results = []
    start_idx = 0

    for i in range(batch_size):
        num_det = int(num_dets[i])
        result = _postprocess_rtdetrv3_core(
            combined_dets,
            start_idx,
            num_det,
            ratios[i],
            padding[i],
            conf_thres=conf_thres,
            no_copy=no_copy,
        )
        results.append(result)
        start_idx += num_det

    return results


@register_jit(nogil=True)
def _postprocess_rtdetrv3_core(
    combined_dets: np.ndarray,
    start_idx: int,
    num_det: int,
    ratios: tuple[float, float],  # noqa: ARG001
    padding: tuple[float, float],  # noqa: ARG001
    conf_thres: float | None = None,
    *,
    no_copy: bool | None = None,
) -> list[np.ndarray]:
    dets = combined_dets[start_idx : start_idx + num_det, :]
    class_ids = dets[:, 0].astype(int)
    scores = dets[:, 1]
    bboxes = dets[:, 2:6]

    if conf_thres is not None:
        mask = scores >= conf_thres
        bboxes = bboxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

    finite_mask = np.all(np.isfinite(bboxes), axis=1)
    if not np.all(finite_mask):
        bboxes = bboxes[finite_mask]
        scores = scores[finite_mask]
        class_ids = class_ids[finite_mask]

    adjusted_bboxes = np.clip(bboxes, 0, None)

    if no_copy:
        return [adjusted_bboxes, scores, class_ids]
    return [adjusted_bboxes.copy(), scores.copy(), class_ids.copy()]
