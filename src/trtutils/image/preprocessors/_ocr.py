# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from trtutils._log import LOG
from trtutils.image._ops import resize_linear
from trtutils.image._warp import perspective_crop

if TYPE_CHECKING:
    from collections.abc import Sequence

_ROTATE_ASPECT = 1.5
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_REC_MEAN = (0.5, 0.5, 0.5)
_REC_STD = (0.5, 0.5, 0.5)


def _round32(value: float) -> int:
    # paddleocr floors each side at 32 so tiny or thin images never resize to nothing
    return max(32, int(round(value / 32) * 32))


def resize_text_detection(
    image: np.ndarray,
    limit_side_len: int = 960,
    limit_type: str = "max",
    max_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Resize an image for a DB text detection model.

    The image is scaled so that its shorter (``limit_type="min"``) or longer
    (``limit_type="max"``) side matches ``limit_side_len``, then both sides are
    rounded to a multiple of 32 (at least 32) as required by the network.

    Parameters
    ----------
    image : np.ndarray
        The image in HWC format.
    limit_side_len : int
        The target side length. Default is 960.
    limit_type : str
        Either ``"min"`` (upscale until the shorter side is at least
        ``limit_side_len``) or ``"max"`` (downscale until the longer side is at
        most ``limit_side_len``). Default is ``"max"``.
    max_shape : tuple[int, int], optional
        The largest (height, width) the engine accepts. If the resized image would
        exceed it, the image is scaled down further (keeping its aspect ratio).

    Returns
    -------
    np.ndarray
        The resized image, with height and width multiples of 32.

    Raises
    ------
    ValueError
        If ``limit_type`` is not valid.

    """
    h, w = image.shape[:2]
    if limit_type == "max":
        ratio = limit_side_len / max(h, w) if max(h, w) > limit_side_len else 1.0
    elif limit_type == "min":
        ratio = limit_side_len / min(h, w) if min(h, w) < limit_side_len else 1.0
    else:
        err_msg = f"Invalid limit_type {limit_type!r}, options are: ['min', 'max']"
        raise ValueError(err_msg)

    resize_h = _round32(int(h * ratio))
    resize_w = _round32(int(w * ratio))
    if max_shape is not None and (resize_h > max_shape[0] or resize_w > max_shape[1]):
        # ponytail: fit inside the engine's profile by shrinking; build a larger
        # profile if this costs accuracy on very large or very wide images
        fit = min(max_shape[0] / resize_h, max_shape[1] / resize_w)
        resize_h = min(_round32(resize_h * fit), max_shape[0] // 32 * 32)
        resize_w = min(_round32(resize_w * fit), max_shape[1] // 32 * 32)
    return resize_linear(image, (resize_w, resize_h))[0]


def preprocess_text_detection(
    images: Sequence[np.ndarray],
    limit_side_len: int = 960,
    limit_type: str = "max",
    input_range: tuple[float, float] = (0.0, 1.0),
    mean: tuple[float, float, float] | None = _IMAGENET_MEAN,
    std: tuple[float, float, float] | None = _IMAGENET_STD,
    max_shape: tuple[int, int] | None = None,
    *,
    verbose: bool | None = None,
) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Preprocess images for a DB text detection model.

    Follows PaddleOCR's ``tools/infer/predict_det.py``: each image is resized
    with :func:`resize_text_detection` and normalized in BGR order. Images of a
    batch which resize to different shapes are zero-padded on the right and
    bottom to the largest one.

    Parameters
    ----------
    images : Sequence[np.ndarray]
        The images in HWC, BGR, uint8 format.
    limit_side_len : int
        The target side length, see :func:`resize_text_detection`. Default is 960.
    limit_type : str
        ``"min"`` or ``"max"``, see :func:`resize_text_detection`. Default is ``"max"``.
    input_range : tuple[float, float]
        The range of input values the model expects, used when mean and std
        are None. By default [0.0, 1.0].
    mean : tuple[float, float, float] | None, optional
        The per-channel mean on the [0, 1] scale. Default is the ImageNet mean
        (0.485, 0.456, 0.406), as in PaddleOCR.
    std : tuple[float, float, float] | None, optional
        The per-channel standard deviation. Default is the ImageNet std
        (0.229, 0.224, 0.225), as in PaddleOCR.
    max_shape : tuple[int, int], optional
        The largest (height, width) the engine accepts.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]
        The float32 batch tensor of shape (N, 3, H, W), the (width, height)
        resize ratio per image, and the (right, bottom) padding per image.

    """
    resized = [resize_text_detection(img, limit_side_len, limit_type, max_shape) for img in images]
    height = max(r.shape[0] for r in resized)
    width = max(r.shape[1] for r in resized)
    # uint8 -> normalized float32 through a 256-entry table per channel: same values as
    # normalizing in float, without the per-pixel float64 math (large on big inputs)
    levels = np.arange(256, dtype=np.float32)[:, None] * (1 / 255.0)
    if mean is not None and std is not None:
        lut = ((levels - np.array(mean)) / np.array(std)).astype(np.float32)
    else:
        low, high = input_range
        lut = np.repeat(levels * (high - low) + low, 3, axis=1).astype(np.float32)
    batch = np.zeros((len(resized), 3, height, width), dtype=np.float32)
    ratios: list[tuple[float, float]] = []
    padding: list[tuple[float, float]] = []
    for i, (image, img) in enumerate(zip(images, resized)):
        h, w = img.shape[:2]
        batch[i, :, :h, :w] = lut[img, np.arange(3)].transpose(2, 0, 1)
        ratios.append((w / image.shape[1], h / image.shape[0]))
        padding.append((float(width - w), float(height - h)))
        if verbose:
            LOG.debug(f"Text detection preprocess {image.shape} -> {(h, w)}")
    return batch, ratios, padding


def crop_text_region(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Crop and rectify a (possibly rotated) text region from an image.

    Parameters
    ----------
    image : np.ndarray
        The image in HWC format.
    box : np.ndarray
        The four corner points (top-left, top-right, bottom-right, bottom-left)
        of the text region, shape (4, 2).

    Returns
    -------
    np.ndarray
        The perspective-corrected crop. Crops that are much taller than wide
        (vertical text) are rotated by 90 degrees.

    """
    points = box.astype(np.float32)
    width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    crop = perspective_crop(image, points, width, height)
    if crop.shape[0] / crop.shape[1] >= _ROTATE_ASPECT:
        crop = np.rot90(crop)
    return crop


def preprocess_text_recognition(
    images: Sequence[np.ndarray],
    image_height: int = 48,
    min_width: int = 320,
    max_width: int | None = None,
    input_range: tuple[float, float] = (0.0, 1.0),
    mean: tuple[float, float, float] | None = _REC_MEAN,
    std: tuple[float, float, float] | None = _REC_STD,
    *,
    verbose: bool | None = None,
) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Preprocess text line images for a CTC text recognition model.

    Follows PaddleOCR's ``tools/infer/predict_rec.py``: every image is resized
    to ``image_height`` keeping its aspect ratio, normalized (to [-1, 1] by
    default) and right-padded with zeros to a common width: the widest aspect
    ratio in the batch (at least ``min_width``).

    Parameters
    ----------
    images : Sequence[np.ndarray]
        The text line images in HWC, BGR, uint8 format.
    image_height : int
        The model input height. Default is 48.
    min_width : int
        The minimum padded batch width. Default is 320.
    max_width : int, optional
        The largest width the engine accepts. Wider lines are squeezed to fit.
    input_range : tuple[float, float]
        The range of input values the model expects, used when mean and std
        are None. By default [0.0, 1.0].
    mean : tuple[float, float, float] | None, optional
        The per-channel mean on the [0, 1] scale. Default is (0.5, 0.5, 0.5).
    std : tuple[float, float, float] | None, optional
        The per-channel standard deviation. Default is (0.5, 0.5, 0.5).
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]
        The float32 batch tensor of shape (N, 3, image_height, W), the
        (width, height) resize ratio per image, and the (right, bottom)
        padding per image.

    """
    max_wh_ratio = max(
        [min_width / image_height] + [img.shape[1] / img.shape[0] for img in images],
    )
    batch_width = int(image_height * max_wh_ratio)
    if max_width is not None:
        batch_width = min(batch_width, max_width)
    if mean is not None and std is not None:
        shift = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        scale = np.array(std, dtype=np.float32).reshape(3, 1, 1)
    else:
        low, high = input_range
        scale = np.float32(1 / (high - low))
        shift = np.float32(-low * scale)
    batch = np.zeros((len(images), 3, image_height, batch_width), dtype=np.float32)
    ratios: list[tuple[float, float]] = []
    padding: list[tuple[float, float]] = []
    for i, img in enumerate(images):
        resized_w = min(batch_width, math.ceil(image_height * img.shape[1] / img.shape[0]))
        resized = resize_linear(img, (resized_w, image_height))[0]
        dst = batch[i, :, :, :resized_w]
        np.divide(resized.transpose(2, 0, 1), np.float32(255), out=dst)
        dst -= shift
        dst /= scale
        ratios.append((resized_w / img.shape[1], image_height / img.shape[0]))
        padding.append((float(batch_width - resized_w), 0.0))
        if verbose:
            LOG.debug(f"Text recognition preprocess {img.shape} -> {(image_height, resized_w)}")
    return batch, ratios, padding
