# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Common implementations for TensorRT engines.

Functions
---------
decode_efficient_nms
    Processes the output of a model with EfficientNMS plugin outputs
postprocess_efficient_nms
    Postprocesses the output of a model to reshape and scale based on preprocessing

"""

from __future__ import annotations

import numpy as np


def postprocess_efficient_nms(
    outputs: list[np.ndarray],
    ratios: tuple[float, float] = (1.0, 1.0),
    padding: tuple[float, float] = (0.0, 0.0),
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

    n_boxes = len(bboxes) // 4
    adjusted_bboxes = bboxes.copy().reshape(n_boxes, 4)
    adjusted_bboxes[:, 0] = (adjusted_bboxes[:, 0] - pad_x) / ratio_width  # x1
    adjusted_bboxes[:, 1] = (adjusted_bboxes[:, 1] - pad_y) / ratio_height  # y1
    adjusted_bboxes[:, 2] = (adjusted_bboxes[:, 2] - pad_x) / ratio_width  # x2
    adjusted_bboxes[:, 3] = (adjusted_bboxes[:, 3] - pad_y) / ratio_height  # y2

    adjusted_bboxes = np.clip(adjusted_bboxes, 0, None)

    return [num_dets, adjusted_bboxes, scores, class_ids]


def decode_efficient_nms(
    outputs: list[np.ndarray],
    conf_thres: float | None = None,
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

    Returns
    -------
    list[tuple[tuple[int, int, int, int], float, int]]
        The decoded outputs.
        Bounding box (x1, y1, x2, y2), confidence score, classid

    """
    num_dects: int = int(outputs[0][0])
    bboxes: np.ndarray = outputs[1]
    scores: np.ndarray = outputs[2]
    classes: np.ndarray = outputs[3]

    frame_dects: list[tuple[tuple[int, int, int, int], float, int]] = []
    for idx in range(num_dects):
        x1, y1, x2, y2 = bboxes[idx]
        # y1, x1, y2, x2 = bboxes[idx]
        np_score = scores[idx]
        np_classid = classes[idx]

        entry = ((int(x1), int(y1), int(x2), int(y2)), float(np_score), int(np_classid))
        frame_dects.append(entry)

    if conf_thres:
        filtered_dects: list[tuple[tuple[int, int, int, int], float, int]] = []
        for bbox, score, class_id in frame_dects:
            if score >= conf_thres:
                filtered_dects.append((bbox, score, class_id))
        frame_dects = filtered_dects

    return frame_dects
