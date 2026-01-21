# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import cv2
import numpy as np
from cv2ext.image import letterbox, rescale, resize_linear

from trtutils._log import LOG


def _preprocess_single(
    image: np.ndarray,
    input_shape: tuple[int, int],
    dtype: np.dtype,
    input_range: tuple[float, float],
    method: str,
    mean: tuple[float, float, float] | None,
    std: tuple[float, float, float] | None,
    *,
    verbose: bool | None = None,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """
    Preprocess a single image.

    Parameters
    ----------
    image : np.ndarray
        The image to be preprocessed.
    input_shape : tuple[int, int]
        The shape to resize the inputs.
    dtype : np.dtype
        The datatype of the inputs to the network.
    input_range : tuple[float, float]
        The range of the model expects for inputs.
    method : str
        The method by which to resize the image.
    mean : tuple[float, float, float], optional
        The mean to subtract from the image.
    std : tuple[float, float, float], optional
        The standard deviation to divide the image by.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    tuple[np.ndarray, tuple[float, float], tuple[float, float]]
        The preprocessed tensor, ratios, and padding.

    Raises
    ------
    ValueError
        If the resize method is unsupported.

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
        err_msg = "Unknown method for image resizing. Options are ['letterbox', 'linear']"
        raise ValueError(err_msg)

    tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)

    if mean is not None and std is not None:
        tensor = tensor / 255.0  # type: ignore[assignment]
        tensor = (tensor - mean) / std
    else:
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


def preprocess(
    images: list[np.ndarray],
    input_shape: tuple[int, int],
    dtype: np.dtype,
    input_range: tuple[float, float] = (0.0, 1.0),
    method: str = "letterbox",
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
    *,
    verbose: bool | None = None,
) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Preprocess inputs for a YOLO network.

    Parameters
    ----------
    images : list[np.ndarray]
        The images to be preprocessed.
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
    mean : tuple[float, float, float], optional
        The mean to subtract from the image.
        By default, None, which will not subtract any mean.
    std : tuple[float, float, float], optional
        The standard deviation to divide the image by.
        By default, None, which will not divide by any standard deviation.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]
        The preprocessed batch tensor, list of ratios, and list of padding per image.

    Raises
    ------
    ValueError
        If the method for resizing is not 'letterbox' or 'linear'
    ValueError
        If only one of mean or std is provided

    """
    # asses if mean and std are provided and valid
    if mean is not None and std is None:
        err_msg = "Mean provided but std is not"
        raise ValueError(err_msg)
    if std is not None and mean is None:
        err_msg = "Std provided but mean is not"
        raise ValueError(err_msg)

    # Process each image
    tensors: list[np.ndarray] = []
    ratios_list: list[tuple[float, float]] = []
    padding_list: list[tuple[float, float]] = []

    for image in images:
        tensor, ratios, padding = _preprocess_single(
            image,
            input_shape,
            dtype,
            input_range,
            method,
            mean,
            std,
            verbose=verbose,
        )
        tensors.append(tensor)
        ratios_list.append(ratios)
        padding_list.append(padding)

    # Stack into batch
    batch_tensor = np.concatenate(tensors, axis=0)
    return batch_tensor, ratios_list, padding_list
