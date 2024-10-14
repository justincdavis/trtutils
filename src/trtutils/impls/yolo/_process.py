# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import cv2
import numpy as np
from cv2ext.image import letterbox


def preprocess(
    inputs: list[np.ndarray], input_shape: tuple[int, int], dtype: np.dtype
) -> list[np.ndarray]:
    """
    Preprocess inputs for a YOLO network.

    Parameters
    ----------
    inputs : list[np.ndarray]
        The inputs to be preprocessed.
    input_shape : tuple[int, int]
        The shape to resize the inputs.
    dtype : np.dtype
        The datatype of the inputs to the network.

    Returns
    -------
    list[np.ndarray]
        The preprocessed data.

    """
    # store preprocessed inputs
    preprocessed: list[np.ndarray] = []
    # process each input
    for img in inputs:
        img = letterbox(img, new_shape=input_shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = img[np.newaxis, :]
        img = np.transpose(img, (0, 3, 1, 2))
        # large performance hit to assemble contiguous array
        img = np.ascontiguousarray(img).astype(dtype)
        # save in array
        preprocessed.append(img)

    return preprocessed


def _postprocess_v_7_8_9(outputs: list[np.ndarray]) -> list[np.ndarray]:
    return outputs


def _postprocess_v_10(outputs: list[np.ndarray]) -> list[np.ndarray]:
    return outputs


def postprocess(outputs: list[np.ndarray], version: int) -> list[np.ndarray]:
    """
    Postprocess outputs from a YOLO network.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from a YOLO network.
    version : int
        The version of the YOLO networks.

    Returns
    -------
    list[np.ndarray]
        The postprocessed outputs.

    """
    if version < 10:
        return _postprocess_v_7_8_9(outputs)
    return _postprocess_v_10(outputs)


def _get_detections_v_7_8_9(
    outputs: list[np.ndarray],
) -> list[list[tuple[tuple[int, int, int, int], float, int]]]:
    results: list[list[tuple[tuple[int, int, int, int], float, int]]] = []
    for output in outputs:
        num_dects = int(output[0][0])
        bboxes = output[1][0][0]
        scores = output[2][0]
        classes = output[3][0]

        frame_dects: list[tuple[tuple[int, int, int, int], float, int]] = []
        for idx in range(num_dects):
            y1, x1, y2, x2 = bboxes[idx]
            score = scores[idx]
            classid = classes[idx]

            entry = ((x1, y1, x2, y2), score, classid)
            frame_dects.append(entry)

        results.append(frame_dects)

    return results


def _get_detections_v_10(
    outputs: list[np.ndarray],
) -> list[list[tuple[tuple[int, int, int, int], float, int]]]:
    # TODO impl
    return []


def get_detections(
    outputs: list[np.ndarray], version: int
) -> list[list[tuple[tuple[int, int, int, int], float, int]]]:
    """
    Get the detections from the output of a YOLO network.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from a YOLO networks.
    version : int
        Which version of YOLO used to generate the outputs.

    Returns
    -------
    list[list[tuple[tuple[int, int, int, int], float, int]]]
        The detections from the YOLO netowrk.
        Each detection is a bounding box in form x1, y1, x2, y2, a confidence score and a class id.

    """
    if version < 10:
        return _get_detections_v_7_8_9(outputs)
    return _get_detections_v_10(outputs)
