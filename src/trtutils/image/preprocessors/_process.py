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
        A batch tensor to write into, used when its shape and dtype match.
        By default None, which allocates a new tensor.
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

    if method not in ("letterbox", "linear"):
        err_msg = "Unknown method for image resizing. Options are ['letterbox', 'linear']"
        raise ValueError(err_msg)

    # fold the normalization into one float32 multiply-add per pixel
    if mean is not None and std is not None:
        std_arr = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        scale = 1.0 / (255.0 * std_arr)
        offset = -np.array(mean, dtype=np.float32).reshape(3, 1, 1) / std_arr
    else:
        low, high = input_range
        scale = np.float32((high - low) / 255.0)
        offset = np.float32(low)

    width, height = input_shape
    shape = (len(images), 3, height, width)
    reuse = out is not None and out.shape == shape and out.dtype == dtype and out.flags.c_contiguous
    batch_tensor = out if reuse else np.empty(shape, dtype=dtype)
    # other dtypes go through float32 so the affine never rounds twice
    scratch = None if dtype == np.float32 else np.empty(shape[1:], dtype=np.float32)

    ratios_list: list[tuple[float, float]] = []
    padding_list: list[tuple[float, float]] = []
    for i, image in enumerate(images):
        if method == "letterbox":
            tensor, ratios, padding = letterbox(image, new_shape=input_shape)
        else:
            tensor, ratios = resize_linear(image, new_shape=input_shape)
            padding = (0.0, 0.0)
        tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)

        # writing the transposed view straight into the slot does HWC->CHW,
        # the cast and the affine in one pass
        dst = batch_tensor[i] if scratch is None else scratch
        np.multiply(tensor.transpose(2, 0, 1), scale, out=dst, dtype=np.float32)
        dst += offset
        if dtype == np.float16:
            # numpy's half cast is ~30x slower than opencv's, which writes through an int16 view
            dst16 = batch_tensor[i].view(np.int16).reshape(-1, width)
            cv2.convertFp16(dst.reshape(-1, width), dst=dst16)
        elif scratch is not None:
            batch_tensor[i] = scratch

        if verbose:
            LOG.debug(
                f"Preprocess {image.shape} -> {input_shape}, ratios: {ratios}, padding: {padding}"
            )
        ratios_list.append(ratios)
        padding_list.append(padding)

    return batch_tensor, ratios_list, padding_list
