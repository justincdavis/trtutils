# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
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
    Postprocess the output of a YOLO-v10 model.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from the TRTEngine using YOLO-v10 output.
    ratios : list[tuple[float, float]]
        The ratios used during preprocessing to resize each input image.
    padding : list[tuple[float, float]]
        The padding used during preprocessing to position each input image.
    conf_thres : float, optional
        Optional confidence threshold to further filter detections by.
        Detections are already filtered by YOLO-v10 parameters
        ahead of time. Should be used if YOLO-v10 was given low-confidence
        and want to filter higher variably.
    input_size : tuple[int, int] | None
        The input size used during preprocessing to resize the input.
    no_copy : bool, optional
        If True, the outputs will not be copied out
        from the cuda allocated host memory. Instead,
        the host memory will be returned directly.
        This memory WILL BE OVERWRITTEN INPLACE
        by future preprocessing calls.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[np.ndarray]]
        The postprocessed outputs per image, each containing
        [bboxes, scores, class_ids] reshaped and scaled based on ratios/padding.

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
    # V10 outputs (1, 300, 6)
    # each final entry is (bbox (4 parts), score, classid)
    ratio_width, ratio_height = ratios
    pad_x, pad_y = padding

    output = outputs[0]

    bboxes: np.ndarray = output[0, :, :4]
    scores: np.ndarray = output[0, :, 4]
    class_ids: np.ndarray = output[0, :, 5].astype(int)

    # pre-filter by the confidence threshold
    if conf_thres is not None:
        mask = scores >= conf_thres
        bboxes = bboxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

    # each bounding box is cx, cy, dx, dy
    adjusted_bboxes = bboxes
    adjusted_bboxes[:, 0] = (adjusted_bboxes[:, 0] - pad_x) / ratio_width  # x1
    adjusted_bboxes[:, 1] = (adjusted_bboxes[:, 1] - pad_y) / ratio_height  # y1
    adjusted_bboxes[:, 2] = (adjusted_bboxes[:, 2] - pad_x) / ratio_width  # x2
    adjusted_bboxes[:, 3] = (adjusted_bboxes[:, 3] - pad_y) / ratio_height  # y2

    # Clip the bounding boxes to ensure they're within valid ranges
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
    Get detections from postprocessed outputs of any supported detector.

    Parameters
    ----------
    outputs : list[list[np.ndarray]]
        Postprocessed outputs per image in the unified format:
        each entry is [bboxes (N,4), scores (N,), class_ids (N,)].
    conf_thres : float, optional
        Confidence threshold to filter detections by.
    nms_iou_thres : float
        IOU threshold for the optional additional NMS operation.
    extra_nms : bool, optional
        Whether to perform an additional CPU-side NMS on final detections.
    agnostic_nms : bool, optional
        Whether to perform class-agnostic NMS during the optional operation.
    verbose : bool, optional
        Whether to log additional information.

    Returns
    -------
    list[list[tuple[tuple[int, int, int, int], float, int]]]
        Detections per image, where each detection is ((x1, y1, x2, y2), score, class_id).

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
    # set conf_thres to zero if not provided (include all bboxes)
    if conf_thres is None:
        conf_thres = 0.0

    # unpack unified outputs
    bboxes = outputs[0]
    scores = outputs[1]
    class_ids = outputs[2]

    # convert to output format
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
    Postprocess the output of the EfficientNMS plugin.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The raw outputs from a model with EfficientNMS output
        in the form [num_dets, bboxes, scores, classes].
    ratios : list[tuple[float, float]]
        The ratios used during preprocessing to resize each input image.
    padding : list[tuple[float, float]]
        The padding used during preprocessing to position each input image.
    input_size : tuple[int, int] | None
        The input size used during preprocessing to resize the input.
    conf_thres : float, optional
        Optional confidence threshold to further filter detections by.
        Detections are already filtered by EfficientNMS parameters
        ahead of time. Should be used if EfficientNMS was given low-confidence
        and you want to filter higher variably.
    no_copy : bool, optional
        If True, the outputs will not be copied out from host memory and
        will be returned directly. This memory WILL BE OVERWRITTEN INPLACE
        by future preprocessing calls.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[np.ndarray]]
        Unified outputs per image, each containing [bboxes (N,4), scores (N,), class_ids (N,)],
        scaled to original image coordinates.

    """
    if verbose:
        LOG.debug(f"EfficientNMS postprocess, raw bboxes shape: {outputs[1].shape}")

    batch_size = outputs[0].shape[0]
    results = []
    for i in range(batch_size):
        # EfficientNMS outputs: [num_dets, bboxes, scores, class_ids]
        # Each has batch dim at position 0
        batch_outputs = [
            outputs[0][i : i + 1],  # num_dets
            outputs[1][i : i + 1],  # bboxes
            outputs[2][i : i + 1],  # scores
            outputs[3][i : i + 1],  # class_ids
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
    # EfficientNMS raw outputs are [num_dets, bboxes, scores, class_ids]
    num_dets, bboxes, scores, class_ids = outputs
    ratio_width, ratio_height = ratios
    pad_x, pad_y = padding

    # Number of valid detections in batch 0
    num_det_id = int(num_dets[0])

    # Slice batch dimension and valid detections only
    bboxes = bboxes[0, :num_det_id]
    scores = scores[0, :num_det_id]
    class_ids = class_ids[0, :num_det_id]

    # Optional confidence pre-filtering
    if conf_thres is not None:
        mask = scores >= conf_thres
        bboxes = bboxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

    # Adjust to original image coordinates
    adjusted_bboxes = bboxes
    adjusted_bboxes[:, 0] = (adjusted_bboxes[:, 0] - pad_x) / ratio_width  # x1
    adjusted_bboxes[:, 1] = (adjusted_bboxes[:, 1] - pad_y) / ratio_height  # y1
    adjusted_bboxes[:, 2] = (adjusted_bboxes[:, 2] - pad_x) / ratio_width  # x2
    adjusted_bboxes[:, 3] = (adjusted_bboxes[:, 3] - pad_y) / ratio_height  # y2

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
    Postprocess the output of an RF-DETR model.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from the TRTEngine using RF-DETR output.
    ratios : list[tuple[float, float]]
        The ratios used during preprocessing to resize each input image.
    padding : list[tuple[float, float]]
        The padding used during preprocessing to position each input image.
    conf_thres : float, optional
        Optional confidence threshold to further filter detections by.
    input_size : tuple[int, int] | None
        The input size used during preprocessing to resize the input.
    no_copy : bool, optional
        If True, the outputs will not be copied out
        from the cuda allocated host memory. Instead,
        the host memory will be returned directly.
        This memory WILL BE OVERWRITTEN INPLACE
        by future preprocessing calls.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[np.ndarray]]
        The postprocessed outputs per image, each containing
        [bboxes, scores, class_ids] reshaped and scaled based on ratios/padding.

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
    # RF-DETR outputs ["dets", "labels"]
    # dets: (batch, num_queries, 4), each is [cx, cy, w, h] in normalized coords (0-1)
    # labels: (batch, num_queries, num_classes) - raw logits
    ratio_width, ratio_height = ratios
    pad_x, pad_y = padding

    dets = outputs[0]
    labels = outputs[1]

    # bboxes are (1, 300, 4)
    # labels are (1, 300, num_classes) - contains raw logits
    bboxes: np.ndarray = dets[0, :, :]  # (num_queries, 4)
    logits: np.ndarray = labels[0, :, :]  # (num_queries, num_classes)

    # Convert logits to probabilities using sigmoid
    probs = 1.0 / (1.0 + np.exp(-logits))  # (num_queries, num_classes)

    # Flatten probabilities to (num_queries * num_classes,) for global top-K selection
    num_queries = probs.shape[0]
    num_classes = probs.shape[1]
    probs_flat = probs.reshape(-1)

    # Get top-K indices (default to num_queries as num_select)
    num_select = num_queries
    if len(probs_flat) < num_select:
        num_select = len(probs_flat)

    # Get top-K values and their flat indices
    topk_indices = np.argsort(probs_flat)[-num_select:][::-1]  # descending order
    topk_scores = probs_flat[topk_indices]

    # Convert flat indices back to (query_idx, class_idx)
    topk_box_indices = topk_indices // num_classes  # which query
    topk_class_ids = topk_indices % num_classes  # which class

    # Gather the corresponding boxes using topk_box_indices
    selected_bboxes = bboxes[topk_box_indices]  # (num_select, 4)

    # Apply confidence threshold filter if provided
    if conf_thres is not None:
        mask = topk_scores >= conf_thres
        selected_bboxes = selected_bboxes[mask]
        topk_scores = topk_scores[mask]
        topk_class_ids = topk_class_ids[mask]

    # class IDS are 1 indexed, so we need to subtract 1
    topk_class_ids = topk_class_ids - 1

    # Bboxes are in normalized coordinates (0-1), scale to model input size first
    if input_size is not None:
        input_w, input_h = input_size
    else:
        # Fallback: assume square input (e.g., 640x640)
        input_w = input_h = 640

    # Convert from normalized center format (cxcywh) to corner format (xyxy)
    cx = selected_bboxes[:, 0] * input_w
    cy = selected_bboxes[:, 1] * input_h
    w = selected_bboxes[:, 2] * input_w
    h = selected_bboxes[:, 3] * input_h

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Adjust bounding boxes based on padding and ratios to get original image coords
    adjusted_x1 = (x1 - pad_x) / ratio_width
    adjusted_y1 = (y1 - pad_y) / ratio_height
    adjusted_x2 = (x2 - pad_x) / ratio_width
    adjusted_y2 = (y2 - pad_y) / ratio_height

    # Stack back into bbox array
    adjusted_bboxes = np.stack([adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2], axis=1)

    # Clip the bounding boxes to ensure they're within valid ranges
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
    Postprocess the output of a DETR-based model (DEIM, RT-DETR, D-FINE).

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from the TRTEngine using DETR output.
    ratios : list[tuple[float, float]]
        The ratios used during preprocessing to resize each input image.
    padding : list[tuple[float, float]]
        The padding used during preprocessing to position each input image.
    conf_thres : float, optional
        Optional confidence threshold to further filter detections by.
    input_size : tuple[int, int] | None
        The input size used during preprocessing to resize the input.
    no_copy : bool, optional
        If True, the outputs will not be copied out
        from the cuda allocated host memory. Instead,
        the host memory will be returned directly.
        This memory WILL BE OVERWRITTEN INPLACE
        by future preprocessing calls.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[np.ndarray]]
        The postprocessed outputs per image, each containing
        [bboxes, scores, class_ids] reshaped and scaled based on ratios/padding.

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
    # DETR outputs ["scores", "labels", "boxes"]
    # scores: (batch, num_queries) containing confidence scores (already probabilities)
    # labels: (batch, num_queries) containing class IDs
    # boxes: (batch, num_queries, 4) where 4 is [x1, y1, x2, y2]
    #
    # IMPORTANT: For RT-DETR/D-FINE models that take "orig_target_sizes" as input,
    # the boxes are ALREADY in original image pixel coordinates!
    # No coordinate transformation is needed!

    scores = outputs[0]
    labels = outputs[1]
    boxes = outputs[2]

    # Extract class IDs, bounding boxes, and scores
    class_ids: np.ndarray = labels[0, :].astype(int)
    bboxes: np.ndarray = boxes[0, :, :]
    scores_arr: np.ndarray = scores[0, :]

    # pre-filter by the confidence threshold
    if conf_thres is not None:
        mask = scores_arr >= conf_thres
        class_ids = class_ids[mask]
        bboxes = bboxes[mask]
        scores_arr = scores_arr[mask]

    # Filter out non-finite bboxes (e.g. infinity values from invalid detections)
    finite_mask = np.all(np.isfinite(bboxes), axis=1)
    if not np.all(finite_mask):
        bboxes = bboxes[finite_mask]
        scores_arr = scores_arr[finite_mask]
        class_ids = class_ids[finite_mask]

    # Bboxes are already in original image pixel coordinates (no transformation needed)
    # Just clip to ensure they're within valid ranges
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
    Postprocess the output of a DETR-based model with LBS output order.

    Models like DEIM and D-FINE output tensors in (labels, boxes, scores) order
    instead of the standard DETR (scores, labels, boxes) order. This function
    reorders the outputs and delegates to postprocess_detr.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from the TRTEngine in (labels, boxes, scores) order.
    ratios : list[tuple[float, float]]
        The ratios used during preprocessing to resize each input image.
    padding : list[tuple[float, float]]
        The padding used during preprocessing to position each input image.
    conf_thres : float, optional
        Optional confidence threshold to further filter detections by.
    input_size : tuple[int, int] | None
        The input size used during preprocessing to resize the input.
    no_copy : bool, optional
        If True, the outputs will not be copied out
        from the cuda allocated host memory. Instead,
        the host memory will be returned directly.
        This memory WILL BE OVERWRITTEN INPLACE
        by future preprocessing calls.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[np.ndarray]]
        The postprocessed outputs per image, each containing
        [bboxes, scores, class_ids] reshaped and scaled based on ratios/padding.

    """
    # Reorder from (labels, boxes, scores) to (scores, labels, boxes)
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
    Postprocess the output of an RT-DETR v3 model (PaddlePaddle export format).

    RT-DETR v3 outputs two tensors:
    - Output 0: Combined detections [total_N, 6] where 6 = (x1, y1, x2, y2, score, class_id)
    - Output 1: Number of detections per image [batch_size]

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from the TRTEngine using RT-DETR v3 output.
    ratios : list[tuple[float, float]]
        The ratios used during preprocessing to resize each input image.
    padding : list[tuple[float, float]]
        The padding used during preprocessing to position each input image.
    conf_thres : float, optional
        Optional confidence threshold to further filter detections by.
    input_size : tuple[int, int] | None
        The input size used during preprocessing to resize the input.
        Not used for RT-DETR v3 since boxes are in original pixel coordinates.
    no_copy : bool, optional
        If True, the outputs will not be copied out
        from the cuda allocated host memory. Instead,
        the host memory will be returned directly.
        This memory WILL BE OVERWRITTEN INPLACE
        by future preprocessing calls.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[np.ndarray]]
        The postprocessed outputs per image, each containing
        [bboxes, scores, class_ids].

    """
    if verbose:
        LOG.debug(f"RT-DETR v3 postprocess, detections shape: {outputs[0].shape}")

    # outputs[0] = combined detections [total_N, 6] or [batch, max_dets, 6]
    # outputs[1] = num_dets per image - can be scalar (batch=1) or [batch_size]
    combined_dets = outputs[0]
    num_dets_raw = outputs[1]

    # Handle scalar (0-dimensional) num_dets for batch size 1
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
    # RT-DETR v3 (PaddlePaddle export) outputs combined tensor [N, 6]
    # Format: (class_id, score, x1, y1, x2, y2)
    # With im_shape/scale_factor inputs, bboxes are already in original pixel coords
    # No coordinate transformation needed (similar to RT-DETR v1/v2 with orig_target_sizes)

    # Extract this image's detections
    dets = combined_dets[start_idx : start_idx + num_det, :]  # [num_det, 6]

    class_ids: np.ndarray = dets[:, 0].astype(int)
    scores: np.ndarray = dets[:, 1]
    bboxes: np.ndarray = dets[:, 2:6]  # x1, y1, x2, y2

    # Optional confidence pre-filtering
    if conf_thres is not None:
        mask = scores >= conf_thres
        bboxes = bboxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

    # Filter out non-finite bboxes (e.g. infinity values from invalid detections)
    finite_mask = np.all(np.isfinite(bboxes), axis=1)
    if not np.all(finite_mask):
        bboxes = bboxes[finite_mask]
        scores = scores[finite_mask]
        class_ids = class_ids[finite_mask]

    # Clip the bounding boxes to ensure they're within valid ranges
    adjusted_bboxes = np.clip(bboxes, 0, None)

    if no_copy:
        return [adjusted_bboxes, scores, class_ids]
    return [adjusted_bboxes.copy(), scores.copy(), class_ids.copy()]
