# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging

import cv2
import numpy as np
from cv2ext.image import letterbox

from trtutils.impls.common import decode_efficient_nms

from ._version import VALID_VERSIONS

_log = logging.getLogger(__name__)
_VERSION_CUTOFF = 10


def preprocess(
    image: np.ndarray,
    input_shape: tuple[int, int],
    dtype: np.dtype,
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

    Returns
    -------
    tuple[np.ndarray, tuple[float, float], tuple[float, float]]
        The preprocessed data.

    """
    _log.debug(f"Preprocess input shape: {image.shape}, output: {input_shape}")

    tensor, ratios, padding = letterbox(image, new_shape=input_shape)
    tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
    # tensor = tensor.transpose((2, 0, 1))
    # tensor = np.expand_dims(tensor, 0)
    # tensor = tensor / 255.0

    tensor = tensor / 255.0  # type: ignore[assignment]
    tensor = tensor[np.newaxis, :]
    tensor = np.transpose(tensor, (0, 3, 1, 2))
    # large performance hit to assemble contiguous array
    tensor = np.ascontiguousarray(tensor).astype(dtype)

    _log.debug(f"Ratios: {ratios}")
    _log.debug(f"Padding: {padding}")
    return tensor, ratios, padding


def _postprocess_v_7_8_9(
    outputs: list[np.ndarray],
    ratios: tuple[float, float],
    padding: tuple[float, float],
) -> list[np.ndarray]:
    # efficient NMS postprocessor essentially
    # inputs are list[num_dets, bboxes, scores, classes]
    num_dets, bboxes, scores, class_ids = outputs
    ratio_width, ratio_height = ratios
    pad_x, pad_y = padding

    n_boxes = len(bboxes) // 4
    adjusted_bboxes = bboxes.copy().reshape(n_boxes, 4)
    adjusted_bboxes[:, 0] = (adjusted_bboxes[:, 0] - pad_x) / ratio_width  # x1
    adjusted_bboxes[:, 1] = (adjusted_bboxes[:, 1] - pad_y) / ratio_height  # y1
    adjusted_bboxes[:, 2] = (adjusted_bboxes[:, 2] - pad_x) / ratio_width  # x2
    adjusted_bboxes[:, 3] = (adjusted_bboxes[:, 3] - pad_y) / ratio_height  # y2

    adjusted_bboxes = np.clip(adjusted_bboxes, 0, None)

    return [num_dets, adjusted_bboxes, scores, class_ids]


def _postprocess_v_10(
    outputs: list[np.ndarray],
    ratios: tuple[float, float],
    padding: tuple[float, float],
) -> list[np.ndarray]:
    # V10 outputs (1, 300, 6)
    # each final entry is (bbox (4 parts), score, classid)
    ratio_width, ratio_height = ratios
    pad_x, pad_y = padding

    output = outputs[0]
    n_boxes = len(output) // 6
    output = output.reshape((n_boxes, 6))

    bboxes: np.ndarray = output[:, :4]
    scores: np.ndarray = output[:, 4]
    class_ids: np.ndarray = output[:, 5].astype(int)

    # each bounding box is cx, cy, dx, dy
    adjusted_bboxes = bboxes.copy()
    adjusted_bboxes[:, 0] = (adjusted_bboxes[:, 0] - pad_x) / ratio_width  # x1
    adjusted_bboxes[:, 1] = (adjusted_bboxes[:, 1] - pad_y) / ratio_height  # y1
    adjusted_bboxes[:, 2] = (adjusted_bboxes[:, 2] - pad_x) / ratio_width  # x2
    adjusted_bboxes[:, 3] = (adjusted_bboxes[:, 3] - pad_y) / ratio_height  # y2

    # Clip the bounding boxes to ensure they're within valid ranges
    adjusted_bboxes = np.clip(adjusted_bboxes, 0, None)

    return [adjusted_bboxes, scores, class_ids]


def postprocess(
    outputs: list[np.ndarray],
    version: int,
    ratios: tuple[float, float] = (1.0, 1.0),
    padding: tuple[float, float] = (0.0, 0.0),
) -> list[np.ndarray]:
    """
    Postprocess outputs from a YOLO network.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from a YOLO network.
    version : int
        The version of the YOLO networks.
    ratios : tuple[float, float]
        The ratio of original image to preprocessed shape
    padding : tuple[float, float]
        The amount of padding added during preprocessing

    Returns
    -------
    list[np.ndarray]
        The postprocessed outputs.

    """
    if version < _VERSION_CUTOFF:
        return _postprocess_v_7_8_9(outputs, ratios, padding)
    return _postprocess_v_10(outputs, ratios, padding)


def _get_detections_v_7_8_9(
    outputs: list[np.ndarray],
    conf_thres: float | None = None,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    return decode_efficient_nms(outputs, conf_thres=conf_thres)


def _get_detections_v_10(
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
                scores[idx],
                class_ids[idx],
            )
            results.append(entry)
    return results


def get_detections(
    outputs: list[np.ndarray],
    version: int,
    conf_thres: float | None = None,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    """
    Get the detections from the output of a YOLO network.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from a YOLO networks.
    version : int
        Which version of YOLO used to generate the outputs.
    conf_thres : float, optional
        The confidence threshold to use for getting detections.

    Returns
    -------
    list[tuple[tuple[int, int, int, int], float, int]]
        The detections from the YOLO netowrk.
        Each detection is a bounding box in form x1, y1, x2, y2, a confidence score and a class id.

    Raises
    ------
    ValueError
        If the version provided is invalid
        If version is V10 and image width/height are not provided

    """
    if version not in VALID_VERSIONS:
        err_msg = (
            f"Invalid version provided. Found: {version}, not in: {VALID_VERSIONS}"
        )
        raise ValueError(err_msg)
    # Handle YoloV 7/8/9
    if version < _VERSION_CUTOFF:
        return _get_detections_v_7_8_9(outputs, conf_thres=conf_thres)
    # Handle YoloV10
    if conf_thres is None:
        return _get_detections_v_10(outputs, conf_thres=conf_thres)
    return _get_detections_v_10(outputs, conf_thres=conf_thres)
