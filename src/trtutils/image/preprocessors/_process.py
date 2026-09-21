# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import cv2
import numpy as np
from cv2ext.image import letterbox, resize_linear

from trtutils._log import LOG


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
    # asses if mean and std are provided and valid
    if mean is not None and std is None:
        err_msg = "Mean provided but std is not"
        raise ValueError(err_msg)
    if std is not None and mean is None:
        err_msg = "Std provided but mean is not"
        raise ValueError(err_msg)
    if method not in ("letterbox", "linear"):
        err_msg = "Unknown method for image resizing. Options are ['letterbox', 'linear']"
        raise ValueError(err_msg)
    if len(images) == 0:
        err_msg = "No images provided for preprocessing"
        raise ValueError(err_msg)

    # precompute the affine constants once per batch, in float32
    # the normalization pipelines below are algebraically identical to the
    # previous per-image implementation, but run as a single fused
    # multiply-add in float32 instead of promoting to float64 through
    # Python-float arithmetic
    use_mean_std = mean is not None and std is not None
    alpha_arr: np.ndarray | None = None
    beta_arr: np.ndarray | None = None
    scale = np.float32(1.0)
    offset = np.float32(0.0)
    if use_mean_std:
        # (t / 255 - mean) / std == t * (1 / (255 * std)) + (-mean / std)
        std_arr = np.asarray(std, dtype=np.float32)
        mean_arr = np.asarray(mean, dtype=np.float32)
        alpha_arr = (1.0 / (255.0 * std_arr)).reshape(1, 1, 3).astype(np.float32)
        beta_arr = (-mean_arr / std_arr).reshape(1, 1, 3).astype(np.float32)
    else:
        # rescale [0, 255] -> input_range == t * ((hi - lo) / 255) + lo
        low, high = input_range
        scale = np.float32((high - low) / 255.0)
        offset = np.float32(low)
    identity_range = not use_mean_std and scale == np.float32(1.0) and offset == np.float32(0.0)

    # Process each image directly into a preallocated batch tensor.
    # The slice assignment performs the HWC->CHW reorder, the cast to the
    # target dtype, and the contiguous copy in a single pass, replacing the
    # per-image transpose/ascontiguousarray/astype plus final concatenate.
    batch_tensor: np.ndarray | None = None
    ratios_list: list[tuple[float, float]] = []
    padding_list: list[tuple[float, float]] = []

    for i, image in enumerate(images):
        if verbose:
            LOG.debug(f"Preprocess input shape: {image.shape}, output: {input_shape}")

        resized: np.ndarray
        if method == "letterbox":
            resized, ratios, padding = letterbox(image, new_shape=input_shape)
        else:
            resized, ratios = resize_linear(image, new_shape=input_shape)
            padding = (0.0, 0.0)

        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if batch_tensor is None:
            height, width = resized.shape[:2]
            shape = (len(images), 3, height, width)
            if out is not None and out.shape == shape and out.dtype == dtype:
                # Reuse the caller's buffer. Above glibc's 32 MB mmap ceiling
                # every fresh allocation is mmap'd and munmap'd, so each write
                # first-touches a new zero page and the whole tensor faults in
                # on every call. That costs ~5x the actual work at batch 7 and
                # up for a 640x640 float32 batch.
                batch_tensor = out
            else:
                batch_tensor = np.empty(shape, dtype=dtype)

        if use_mean_std:
            chw = np.multiply(resized, alpha_arr, dtype=np.float32)
            chw += beta_arr
            batch_tensor[i] = chw.transpose(2, 0, 1)
        elif identity_range:
            batch_tensor[i] = resized.transpose(2, 0, 1)
        else:
            chw = np.multiply(resized, scale, dtype=np.float32)
            if offset != np.float32(0.0):
                chw += offset
            batch_tensor[i] = chw.transpose(2, 0, 1)

        if verbose:
            LOG.debug(f"Ratios: {ratios}")
            LOG.debug(f"Padding: {padding}")

        ratios_list.append(ratios)
        padding_list.append(padding)

    if batch_tensor is None:
        # no images were given, so no batch tensor was ever allocated. before
        # the preallocated-batch rewrite this raised out of np.concatenate,
        # keep it an explicit error rather than returning None
        err_msg = "Cannot preprocess an empty list of images."
        raise ValueError(err_msg)

    return batch_tensor, ratios_list, padding_list
