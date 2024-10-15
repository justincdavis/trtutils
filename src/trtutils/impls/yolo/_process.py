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

    tensor = tensor / 255.0
    tensor = tensor[np.newaxis, :]
    tensor = np.transpose(tensor, (0, 3, 1, 2))
    # large performance hit to assemble contiguous array
    tensor = np.ascontiguousarray(tensor).astype(dtype)

    _log.debug(f"Ratios: {ratios}")
    _log.debug(f"Padding: {padding}")
    return tensor, ratios, padding


def _postprocess_v_7_8_9(outputs: list[np.ndarray], ratios: tuple[float, float], padding: tuple[float, float]) -> list[np.ndarray]:
    num_dets, bboxes, scores, class_ids = outputs
    ratio_width, ratio_height = ratios
    pad_x, pad_y = padding

    adjusted_bboxes = bboxes.copy()
    adjusted_bboxes[..., 0] = (adjusted_bboxes[..., 0] - pad_x) / ratio_width  # x1
    adjusted_bboxes[..., 1] = (adjusted_bboxes[..., 1] - pad_y) / ratio_height  # y1
    adjusted_bboxes[..., 2] = (adjusted_bboxes[..., 2] - pad_x) / ratio_width  # x2
    adjusted_bboxes[..., 3] = (adjusted_bboxes[..., 3] - pad_y) / ratio_height  # y2

    adjusted_bboxes = np.clip(adjusted_bboxes, 0, None)

    return [num_dets, adjusted_bboxes, scores, class_ids]


def _postprocess_v_10(outputs: list[np.ndarray], ratios: tuple[float, float], padding: tuple[float, float]) -> list[np.ndarray]:
    # TODO: implement scaling for ratio/padding
    return outputs


def postprocess(outputs: list[np.ndarray], version: int, ratios: tuple[float, float] = (1.0, 1.0), padding: tuple[float, float] = (0.0, 0.0)) -> list[np.ndarray]:
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
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    return decode_efficient_nms(outputs)


def _get_detections_v_10(
    outputs: list[np.ndarray],
    img_width: int,
    img_height: int,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    output = outputs[0]  # yolov10 has a single output
    detections = output.reshape(-1, 6)  # remove batch dimension

    boxes = detections[:, :4]
    confs = detections[:, 4]
    class_ids = detections[:, 5].astype(int)

    # convert to pixel
    boxes[:, 0] *= img_width   # x_center
    boxes[:, 1] *= img_height  # y_center
    boxes[:, 2] *= img_width   # width
    boxes[:, 3] *= img_height  # height

    # convert to xyxy
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    x1 = x1.astype(int)
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    y1 = y1.astype(int)
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    x1 = x2.astype(int)
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    y2 = y2.astype(int)

    # convert to output format
    results: list[tuple[tuple[int, int, int, int], float, int]] = []
    for idx in range(len(x1)):
        entry = (
            (x1[idx], y1[idx], x2[idx], y2[idx]),
            confs[idx],
            class_ids[idx],
        )
        results.append(entry)
    return results

def get_detections(
    outputs: list[np.ndarray],
    version: int,
    img_width: int | None = None,
    img_height: int | None = None,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    """
    Get the detections from the output of a YOLO network.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from a YOLO networks.
    version : int
        Which version of YOLO used to generate the outputs.
    img_width : int, optional
        The image width, needed for V10 decode.
    img_height : int, optional
        The image height, needed for V10 decode.
    
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
        err_msg = f"Invalid version provided. Found: {version}, not in: {VALID_VERSIONS}"
        raise ValueError(err_msg)
    if version < _VERSION_CUTOFF:
        return _get_detections_v_7_8_9(outputs)
    if img_width is None or img_height is None:
        err_msg = "Image width and height must be provided for V10 decode."
        raise ValueError(err_msg)
    return _get_detections_v_10(outputs, img_width, img_height)
