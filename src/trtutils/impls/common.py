# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Common implementations for TensorRT engines.

Functions
---------
decode_efficient_nms
    Processes the output of a model with EfficientNMS plugin outputs

"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def decode_efficient_nms(
    outputs: list[np.ndarray],
    conf_thres: float | None = None,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    """
    Decode EfficientNMS plugin output.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from a model with EfficientNMS output
    conf_thres : float
        A confidence value to threshold detctions by.
        By default None.

    Returns
    -------
    list[tuple[tuple[int, int, int, int], float, int]]
        The decoded outputs.
        Bounding box (x1, y1, x2, y2), confidence score, classid

    """
    num_dects: int = outputs[0][0]
    bboxes: np.ndarray = outputs[1]
    scores: np.ndarray = outputs[2]
    classes: np.ndarray = outputs[3]

    frame_dects: list[tuple[tuple[int, int, int, int], float, int]] = []
    for idx in range(num_dects):
        x1, y1, x2, y2 = bboxes[idx]
        # y1, x1, y2, x2 = bboxes[idx]
        score = scores[idx]
        classid = classes[idx]

        entry = ((int(x1), int(y1), int(x2), int(y2)), score, classid)
        frame_dects.append(entry)

    if conf_thres:
        filtered_dects: list[tuple[tuple[int, int, int, int], float, int]] = []
        for bbox, score, class_id in frame_dects:
            if score >= conf_thres:
                filtered_dects.append((bbox, score, class_id))
        frame_dects = filtered_dects

    return frame_dects
