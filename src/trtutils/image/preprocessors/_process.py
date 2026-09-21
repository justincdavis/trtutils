# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import cv2
import numpy as np
from cv2ext.image import letterbox, resize_linear

from trtutils._log import LOG


def _validate_args(
    images: list[np.ndarray],
    method: str,
    mean: tuple[float, float, float] | None,
    std: tuple[float, float, float] | None,
) -> None:
    """
    Reject argument combinations preprocess cannot serve.

    Parameters
    ----------
    images : list[np.ndarray]
        The images to be preprocessed.
    method : str
        The method by which to resize the image.
    mean : tuple[float, float, float], optional
        The mean to subtract from the image.
    std : tuple[float, float, float], optional
        The standard deviation to divide the image by.

    Raises
    ------
    ValueError
        If only one of mean or std is provided, the method is unknown, or
        no images are given.

    """
    if (mean is None) != (std is None):
        err_msg = (
            "Mean provided but std is not" if mean is not None else "Std provided but mean is not"
        )
        raise ValueError(err_msg)
    if method not in ("letterbox", "linear"):
        err_msg = "Unknown method for image resizing. Options are ['letterbox', 'linear']"
        raise ValueError(err_msg)
    if len(images) == 0:
        err_msg = "No images provided for preprocessing"
        raise ValueError(err_msg)


def _affine(
    input_range: tuple[float, float],
    mean: tuple[float, float, float] | None,
    std: tuple[float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Fold the normalization into one per-channel float32 multiply-add.

    Parameters
    ----------
    input_range : tuple[float, float]
        The range the model expects for inputs.
    mean : tuple[float, float, float], optional
        The mean to subtract from the image.
    std : tuple[float, float, float], optional
        The standard deviation to divide the image by.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        ``(alpha, beta)`` of shape (1, 1, 3) such that
        ``normalized = pixel * alpha + beta``, or None when the range is the
        identity (0, 255) and pixels pass through untouched.

    """
    if mean is not None and std is not None:
        # (t / 255 - mean) / std == t * (1 / (255 * std)) + (-mean / std)
        std_arr = np.asarray(std, dtype=np.float32)
        mean_arr = np.asarray(mean, dtype=np.float32)
        alpha = (1.0 / (255.0 * std_arr)).reshape(1, 1, 3).astype(np.float32)
        beta = (-mean_arr / std_arr).reshape(1, 1, 3).astype(np.float32)
        return alpha, beta
    # rescale [0, 255] -> input_range == t * ((hi - lo) / 255) + lo
    low, high = input_range
    scale = np.float32((high - low) / 255.0)
    offset = np.float32(low)
    if scale == np.float32(1.0) and offset == np.float32(0.0):
        return None
    return np.full((1, 1, 3), scale, dtype=np.float32), np.full((1, 1, 3), offset, dtype=np.float32)


def _resize(
    image: np.ndarray,
    input_shape: tuple[int, int],
    method: str,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """
    Resize one image to the model input shape and convert BGR to RGB.

    Parameters
    ----------
    image : np.ndarray
        The image to resize.
    input_shape : tuple[int, int]
        The shape to resize the inputs.
    method : str
        The method by which to resize the image.

    Returns
    -------
    tuple[np.ndarray, tuple[float, float], tuple[float, float]]
        The resized RGB image, ratios, and padding.

    """
    if method == "letterbox":
        resized, ratios, padding = letterbox(image, new_shape=input_shape)
    else:
        resized, ratios = resize_linear(image, new_shape=input_shape)
        padding = (0.0, 0.0)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), ratios, padding


def preprocess(
    images: list[np.ndarray],
    input_shape: tuple[int, int],
    dtype: np.dtype,
    input_range: tuple[float, float] = (0.0, 1.0),
    method: str = "letterbox",
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
    out: np.ndarray | None = None,
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
    out : np.ndarray, optional
        A preallocated batch tensor to write into. Used when its shape and
        dtype match the batch, avoiding a fresh allocation per call.
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
    ValueError
        If no images are provided

    """
    _validate_args(images, method, mean, std)
    # the normalization is algebraically the same as the previous per-image
    # implementation, but runs as one fused multiply-add in float32 instead
    # of promoting to float64 through Python-float arithmetic
    affine = _affine(input_range, mean, std)

    # Process each image directly into a preallocated batch tensor. The
    # slice assignment performs the HWC->CHW reorder, the cast to the target
    # dtype, and the contiguous copy in a single pass, replacing the
    # per-image transpose/ascontiguousarray/astype plus final concatenate.
    batch_tensor: np.ndarray | None = None
    ratios_list: list[tuple[float, float]] = []
    padding_list: list[tuple[float, float]] = []

    for i, image in enumerate(images):
        resized, ratios, padding = _resize(image, input_shape, method)
        if verbose:
            LOG.debug(
                f"Preprocess {image.shape} -> {input_shape}: ratios {ratios}, padding {padding}"
            )

        if batch_tensor is None:
            shape = (len(images), 3, *resized.shape[:2])
            # Reuse the caller's buffer when it fits. Above glibc's 32 MB
            # mmap ceiling every fresh allocation is mmap'd and munmap'd, so
            # each write first-touches a new zero page and the whole tensor
            # faults in on every call: ~5x the actual work at batch 7 and up
            # for a 640x640 float32 batch.
            reuse = out is not None and out.shape == shape and out.dtype == dtype
            batch_tensor = out if reuse else np.empty(shape, dtype=dtype)

        if affine is None:
            batch_tensor[i] = resized.transpose(2, 0, 1)
        else:
            chw = np.multiply(resized, affine[0], dtype=np.float32)
            chw += affine[1]
            batch_tensor[i] = chw.transpose(2, 0, 1)

        ratios_list.append(ratios)
        padding_list.append(padding)

    # _validate_args rejected an empty list, so the tensor was allocated
    return batch_tensor, ratios_list, padding_list  # ty: ignore[invalid-return-type]
