# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import cv2
import numpy as np
from cv2ext.image import letterbox, rescale, resize_linear

from trtutils._log import LOG


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
        LOG.debug(f"Preprocess input shape: {image.shape}, output: {input_shape}")

    tensor: np.ndarray
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
        LOG.debug(f"Ratios: {ratios}")
        LOG.debug(f"Padding: {padding}")
    return tensor, ratios, padding
