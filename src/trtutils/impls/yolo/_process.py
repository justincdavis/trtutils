# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging

import cv2
import numpy as np
from cv2ext.bboxes import nms
from cv2ext.image import letterbox, rescale, resize_linear

from trtutils.impls.common import decode_efficient_nms, postprocess_efficient_nms

# EfficientNMS as 4 outputs
_EFF_NUM_OUTPUTS = 4

_log = logging.getLogger(__name__)


def preprocess(
    image: np.ndarray,
    input_shape: tuple[int, int],
    dtype: np.dtype,
    input_range: tuple[float, float] = (0.0, 1.0),
    method: str = "letterbox",
    *,
    verbose: bool | None = None,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """
    Preprocess inputs for a YOLO network.

    Parameters
    ----------
    image : np.ndarray
        The inputs to be preprocessed.
    input_shape : tuple[int, int]
        The shape to resize the inputs.
    dtype : np.dtype
        The datatype of the inputs to the network.
    input_range : tuple[float, float]
        The range of the model expects for inputs.
        By default, [0.0, 1.0] (divide input by 255.0)
    method : str
        The method by which to resize the image.
        By default letterbox will be used.
        Options are [letterbox, linear]
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    tuple[np.ndarray, tuple[float, float], tuple[float, float]]
        The preprocessed data.

    Raises
    ------
    ValueError
        If the method for resizing is not 'letterbox' or 'linear'

    """
    if verbose:
        _log.debug(f"Preprocess input shape: {image.shape}, output: {input_shape}")

    if method == "letterbox":
        tensor, ratios, padding = letterbox(image, new_shape=input_shape)
    elif method == "linear":
        tensor, ratios = resize_linear(image, new_shape=input_shape)
        padding = (0.0, 0.0)
    else:
        err_msg = (
            "Unknown method for image resizing. Options are ['letterbox', 'linear']"
        )
        raise ValueError(err_msg)

    tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)

    # tensor = tensor / 255.0  # type: ignore[assignment]
    tensor = rescale(tensor, input_range)

    tensor = tensor[np.newaxis, :]
    tensor = np.transpose(tensor, (0, 3, 1, 2))
    # large performance hit to assemble contiguous array
    if not tensor.flags["C_CONTIGUOUS"]:
        tensor = np.ascontiguousarray(tensor)
    tensor = tensor.astype(dtype)

    if verbose:
        _log.debug(f"Ratios: {ratios}")
        _log.debug(f"Padding: {padding}")
    return tensor, ratios, padding


def _postprocess_v_10(
    outputs: list[np.ndarray],
    ratios: tuple[float, float],
    padding: tuple[float, float],
    conf_thres: float | None = None,
    *,
    no_copy: bool | None = None,
    verbose: bool | None = None,
) -> list[np.ndarray]:
    # V10 outputs (1, 300, 6)
    # each final entry is (bbox (4 parts), score, classid)
    ratio_width, ratio_height = ratios
    pad_x, pad_y = padding

    output = outputs[0]

    if verbose:
        _log.debug(f"V10 postprocess, output shape: {output.shape}")

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


def postprocess(
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
    return _postprocess_v_10(
        outputs,
        ratios,
        padding,
        conf_thres=conf_thres,
        no_copy=no_copy,
        verbose=verbose,
    )


def _get_detections_v_10(
    outputs: list[np.ndarray],
    conf_thres: float | None = None,
    nms_iou_thres: float = 0.5,
    *,
    extra_nms: bool | None = None,
    agnostic_nms: bool | None = None,
    verbose: bool | None = None,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    # set conf_thres to zero if not provided (include all bboxes)
    if conf_thres is None:
        conf_thres = 0.0

    # unpack
    bboxes = outputs[0]
    scores = outputs[1]
    class_ids = outputs[2]

    if verbose:
        _log.debug(f"Decoding: {bboxes.shape[0]} bboxes")

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

    if extra_nms:
        results = nms(results, iou_threshold=nms_iou_thres, agnostic=agnostic_nms)

    return results


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
            _log.debug("Using EfficientNMS decoding")
        return decode_efficient_nms(
            outputs,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            extra_nms=extra_nms,
            agnostic_nms=agnostic_nms,
        )
    if verbose:
        _log.debug("Using V10 decoding")
    return _get_detections_v_10(
        outputs,
        conf_thres=conf_thres,
        nms_iou_thres=nms_iou_thres,
        extra_nms=extra_nms,
        agnostic_nms=agnostic_nms,
    )
