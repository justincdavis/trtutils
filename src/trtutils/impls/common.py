# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Common implementations for TensorRT engines.

Functions
---------
:func:`decode_efficient_nms`
    Processes the output of a model with EfficientNMS plugin outputs.
:func:`postprocess_efficient_nms`
    Postprocesses the output of a model to reshape and scale based on preprocessing.
"""

from __future__ import annotations

import logging

import numpy as np
from cv2ext.bboxes import nms

_log = logging.getLogger(__name__)


def postprocess_efficient_nms(
    outputs: list[np.ndarray],
    ratios: tuple[float, float] = (1.0, 1.0),
    padding: tuple[float, float] = (0.0, 0.0),
    conf_thres: float | None = None,
    *,
    no_copy: bool | None = None,
    verbose: bool | None = None,
) -> list[np.ndarray]:
    """
    Postprocess the output of the EfficientNMS plugin.

    Must be used before passing outputs to decode_efficient_nms
    since this will reshape the outputs.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from the TRTEngine using EfficientNMS output.
    ratios : tuple[float, float]
        The ratios used during preprocessing to resize the input.
    padding : tuple[float, float]
        The padding used during preprocessing to position the input.
    conf_thres : float, optional
        Optional confidence threshold to further filter detections by.
        Detections are already filtered by EfficientNMS parameters
        ahead of time. Should be used if EfficientNMS was given low-confidence
        and want to filter higher variably.
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
    list[np.ndarray]
        The postprocessed outputs, reshaped and scaled based on ratios/padding.

    """
    # efficient NMS postprocessor essentially
    # inputs are list[num_dets, bboxes, scores, classes]
    num_dets, bboxes, scores, class_ids = outputs
    ratio_width, ratio_height = ratios
    pad_x, pad_y = padding

    if verbose:
        _log.debug(f"EfficientNMS postprocess, bboxes shape: {bboxes.shape}")

    # throw out all detections not included in the num_dets
    num_det_id = int(outputs[0][0])  # needs to be integer
    bboxes = bboxes[:, :num_det_id]
    scores = scores[:, :num_det_id]
    class_ids = class_ids[:, :num_det_id]

    if conf_thres is not None:
        mask = scores >= conf_thres
        bboxes = np.where(mask[..., np.newaxis], bboxes, 0)
        scores = np.where(mask, scores, 0)
        class_ids = np.where(mask, class_ids, 0)

    adjusted_bboxes = bboxes
    adjusted_bboxes[:, :, 0] = (adjusted_bboxes[:, :, 0] - pad_x) / ratio_width  # x1
    adjusted_bboxes[:, :, 1] = (adjusted_bboxes[:, :, 1] - pad_y) / ratio_height  # y1
    adjusted_bboxes[:, :, 2] = (adjusted_bboxes[:, :, 2] - pad_x) / ratio_width  # x2
    adjusted_bboxes[:, :, 3] = (adjusted_bboxes[:, :, 3] - pad_y) / ratio_height  # y2

    adjusted_bboxes = np.clip(adjusted_bboxes, 0, None)

    if no_copy:
        return [num_dets, adjusted_bboxes, scores, class_ids]
    return [num_dets.copy(), adjusted_bboxes.copy(), scores.copy(), class_ids.copy()]


def decode_efficient_nms(
    outputs: list[np.ndarray],
    conf_thres: float | None = None,
    nms_iou_thres: float = 0.5,
    *,
    extra_nms: bool | None = None,
    agnostic_nms: bool | None = None,
    verbose: bool | None = None,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    """
    Decode EfficientNMS plugin output.

    Must have called postprocess_efficient_nms before calling
    this function, due to the reshape stage needing to occur.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from a model with EfficientNMS output
    conf_thres : float
        A confidence value to threshold detctions by.
        By default None.
    nms_iou_thres : float
        The IOU threshold to use during the optional additional
        NMS operation. By default, 0.5
    extra_nms : bool, optional
        Whether or not an additional CPU-side NMS operation
        should be conducted on final detections.
    agnostic_nms : bool, optional
        Whether or not to perform class-agnostic NMS during the
        optional additional operation.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[tuple[tuple[int, int, int, int], float, int]]
        The decoded outputs.
        Bounding box (x1, y1, x2, y2), confidence score, classid

    """
    num_dects: int = int(outputs[0][0])
    bboxes: np.ndarray = outputs[1][0]
    scores: np.ndarray = outputs[2][0]
    classes: np.ndarray = outputs[3][0]

    conf_thres = conf_thres or 0.0

    if verbose:
        _log.debug(f"Generating detections for: {num_dects} bboxes")

    frame_dects: list[tuple[tuple[int, int, int, int], float, int]] = []
    for idx in range(num_dects):
        x1, y1, x2, y2 = bboxes[idx]
        score = float(scores[idx])
        np_classid = classes[idx]

        if score >= conf_thres:
            entry = ((int(x1), int(y1), int(x2), int(y2)), score, int(np_classid))
            frame_dects.append(entry)

    if extra_nms:
        frame_dects = nms(
            frame_dects,
            iou_threshold=nms_iou_thres,
            agnostic=agnostic_nms,
        )

    return frame_dects
