# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import numpy as np
from cv2ext.bboxes import nms

from trtutils._jit import register_jit
from trtutils._log import LOG

# EfficientNMS as 4 outputs
_EFF_NUM_OUTPUTS = 4


def _postprocess_yolov10(
    outputs: list[np.ndarray],
    ratios: tuple[float, float],
    padding: tuple[float, float],
    conf_thres: float | None = None,
    *,
    no_copy: bool | None = None,
    verbose: bool | None = None,
) -> list[np.ndarray]:
    if verbose:
        LOG.debug(f"V10 postprocess, output shape: {outputs[0].shape}")

    return _postprocess_yolov10_core(
        outputs,
        ratios,
        padding,
        conf_thres=conf_thres,
        no_copy=no_copy,
    )


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


def postprocess_detections(
    outputs: list[np.ndarray],
    ratios: tuple[float, float] = (1.0, 1.0),
    padding: tuple[float, float] = (0.0, 0.0),
    conf_thres: float | None = None,
    *,
    no_copy: bool | None = None,
    verbose: bool | None = None,
) -> list[np.ndarray]:
    """
    Postprocess outputs from a YOLO network.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from a YOLO network.
    ratios : tuple[float, float]
        The ratio of original image to preprocessed shape
    padding : tuple[float, float]
        The amount of padding added during preprocessing
    conf_thres : float, optional
        The confidence score for which detections below will be thrown out.
    no_copy : bool, optional
        If True, the outputs will not be copied out
        from the cuda allocated host memory. Instead,
        the host memory will be returned directly.
        This memory WILL BE OVERWRITTEN INPLACE
        by future inference calls.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[np.ndarray]
        The postprocessed outputs.

    """
    if len(outputs) == _EFF_NUM_OUTPUTS:
        return postprocess_efficient_nms(
            outputs,
            ratios,
            padding,
            conf_thres=conf_thres,
            no_copy=no_copy,
            verbose=verbose,
        )
    return _postprocess_yolov10(
        outputs,
        ratios,
        padding,
        conf_thres=conf_thres,
        no_copy=no_copy,
        verbose=verbose,
    )


def _get_detections_yolov10(
    outputs: list[np.ndarray],
    conf_thres: float | None = None,
    nms_iou_thres: float = 0.5,
    *,
    extra_nms: bool | None = None,
    agnostic_nms: bool | None = None,
    verbose: bool | None = None,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    if verbose:
        LOG.debug(f"Decoding: {outputs[0].shape[0]} bboxes")

    results = _get_detections_yolov10_core(outputs, conf_thres)

    if extra_nms:
        results = nms(results, iou_threshold=nms_iou_thres, agnostic=agnostic_nms)

    return results


@register_jit(nogil=True)
def _get_detections_yolov10_core(
    outputs: list[np.ndarray],
    conf_thres: float | None = None,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    # set conf_thres to zero if not provided (include all bboxes)
    if conf_thres is None:
        conf_thres = 0.0

    # unpack
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
    if verbose:
        LOG.debug(f"EfficientNMS postprocess, bboxes shape: {outputs[1].shape}")

    return _postprocess_efficient_nms_core(
        outputs,
        ratios,
        padding,
        conf_thres,
        no_copy=no_copy,
    )


@register_jit(nogil=True)
def _postprocess_efficient_nms_core(
    outputs: list[np.ndarray],
    ratios: tuple[float, float],
    padding: tuple[float, float],
    conf_thres: float | None = None,
    *,
    no_copy: bool | None = None,
) -> list[np.ndarray]:
    # efficient NMS postprocessor essentially
    # inputs are list[num_dets, bboxes, scores, classes]
    num_dets, bboxes, scores, class_ids = outputs
    ratio_width, ratio_height = ratios
    pad_x, pad_y = padding

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
    if verbose:
        LOG.debug(f"Generating detections for: {int(outputs[0][0])} bboxes")

    frame_dects = _decode_efficient_nms_core(outputs, conf_thres)

    if extra_nms:
        frame_dects = nms(
            frame_dects,
            iou_threshold=nms_iou_thres,
            agnostic=agnostic_nms,
        )

    return frame_dects


@register_jit(nogil=True)
def _decode_efficient_nms_core(
    outputs: list[np.ndarray],
    conf_thres: float | None = None,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    num_dects: int = int(outputs[0][0])
    bboxes: np.ndarray = outputs[1][0]
    scores: np.ndarray = outputs[2][0]
    classes: np.ndarray = outputs[3][0]

    conf_thres = conf_thres or 0.0

    frame_dects: list[tuple[tuple[int, int, int, int], float, int]] = []
    for idx in range(num_dects):
        x1, y1, x2, y2 = bboxes[idx]
        score = float(scores[idx])
        np_classid = classes[idx]

        if score >= conf_thres:
            entry = ((int(x1), int(y1), int(x2), int(y2)), score, int(np_classid))
            frame_dects.append(entry)

    return frame_dects


def get_detections(
    outputs: list[np.ndarray],
    conf_thres: float | None = None,
    nms_iou_thres: float = 0.5,
    *,
    extra_nms: bool | None = None,
    agnostic_nms: bool | None = None,
    verbose: bool | None = None,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    """
    Get the detections from the output of a YOLO network.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from a YOLO networks.
    conf_thres : float, optional
        The confidence threshold to use for getting detections.
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
        The detections from the YOLO netowrk.
        Each detection is a bounding box in form x1, y1, x2, y2, a confidence score and a class id.

    """
    if len(outputs) == _EFF_NUM_OUTPUTS:
        if verbose:
            LOG.debug("Using EfficientNMS decoding")
        return decode_efficient_nms(
            outputs,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            extra_nms=extra_nms,
            agnostic_nms=agnostic_nms,
        )
    if verbose:
        LOG.debug("Using V10 decoding")
    return _get_detections_yolov10(
        outputs,
        conf_thres=conf_thres,
        nms_iou_thres=nms_iou_thres,
        extra_nms=extra_nms,
        agnostic_nms=agnostic_nms,
    )
