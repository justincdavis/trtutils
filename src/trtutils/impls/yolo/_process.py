# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import cv2
import numpy as np
from cv2ext.image import letterbox


def preprocess(inputs: list[np.ndarray], input_shape: list[tuple[int, int]] | tuple[int, int]) -> list[np.ndarray]:
    # store preprocessed inputs
    preprocessed: list[np.ndarray] = []
    # ensure input shapes exist for letter boxing
    if not isinstance(input_shape, list):
        input_shapes = [input_shape] * len(inputs)
    else:
        input_shapes = input_shape
    # process each input
    for img, imgsz in zip(inputs, input_shapes):
        img = letterbox(img, new_shape=imgsz)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = img[np.newaxis, :]
        img = np.transpose(img, (0, 3, 1, 2))
        # large performance hit to assemble contiguous array
        img = np.ascontiguousarray(img)
        # save in array
        preprocessed.append(img)

    return preprocessed


def postprocess(outputs: list[np.ndarray]) -> list[np.ndarray]:
    """
    Postprocess outputs from a YOLO V7/8/9 network.
    
    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from a YOLO network.
        
    Returns
    -------
    list[np.ndarray]
        The postprocessed outputs.

    """
    return outputs


def postprocess_v10(outputs: list[np.ndarray]) -> list[np.ndarray]:
    """
    Postprocess outputs from a YOLO V10 network.
    
    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from a YOLO network.
        
    Returns
    -------
    list[np.ndarray]
        The postprocessed outputs.

    """
    return outputs


def get_detections(outputs: list[np.ndarray]) -> list[tuple[tuple[int, int, int, int], float, int]]:
    """
    Get the detections from the output of a YOLO V7/8/9 network.
    
    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from a YOLO networks.
    
    Returns
    -------
    list[tuple[tuple[int, int, int, int], float, int]]
        The detections from the YOLO netowrk.
        Each detection is a bounding box in form x1, y1, x2, y2, a confidence score and a class id.
    
    """
    # TODO impl
    return []


def get_detections_v10(outputs: list[np.ndarray]) -> list[tuple[tuple[int, int, int, int], float, int]]:
    """
    Get the detections from the output of a YOLO V10 network.
    
    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from a YOLO networks.
    
    Returns
    -------
    list[tuple[tuple[int, int, int, int], float, int]]
        The detections from the YOLO netowrk.
        Each detection is a bounding box in form x1, y1, x2, y2, a confidence score and a class id.
    
    """
    # TODO impl
    return []
