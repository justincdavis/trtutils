# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Preprocessing for PaddleOCR (PP-OCR) style text detection and recognition models.

The resize and normalization follow PaddleOCR's inference pipeline
(``tools/infer/predict_det.py`` / ``predict_rec.py``) and its defaults.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

_ROTATE_ASPECT = 1.5
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


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
    return cv2.resize(image, (resize_w, resize_h))


def preprocess_text_detection(
    image: np.ndarray,
    limit_side_len: int = 960,
    limit_type: str = "max",
    mean: tuple[float, float, float] = _IMAGENET_MEAN,
    std: tuple[float, float, float] = _IMAGENET_STD,
    max_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Preprocess an image for a DB text detection model.

    Parameters
    ----------
    image : np.ndarray
        The image in HWC, BGR, uint8 format.
    limit_side_len : int
        The target side length, see :func:`resize_text_detection`. Default is 960.
    limit_type : str
        ``"min"`` or ``"max"``, see :func:`resize_text_detection`. Default is ``"max"``.
    mean : tuple[float, float, float]
        The per-channel mean (on the [0, 1] scale, BGR order as PaddleOCR applies it).
        Default is the ImageNet mean (0.485, 0.456, 0.406), as in PaddleOCR.
    std : tuple[float, float, float]
        The per-channel standard deviation. Default is the ImageNet std
        (0.229, 0.224, 0.225), as in PaddleOCR.
    max_shape : tuple[int, int], optional
        The largest (height, width) the engine accepts.

    Returns
    -------
    np.ndarray
        The float32 input tensor of shape (1, 3, H, W).

    """
    resized = resize_text_detection(image, limit_side_len, limit_type, max_shape)
    # uint8 -> normalized float32 through a 256-entry table per channel: same values as
    # normalizing in float, without the per-pixel float64 math (large on big inputs)
    levels = np.arange(256, dtype=np.float32)[:, None] * (1 / 255.0)
    lut = ((levels - np.array(mean)) / np.array(std)).astype(np.float32)
    out = np.empty((1, 3, *resized.shape[:2]), dtype=np.float32)
    for c, plane in enumerate(cv2.split(resized)):
        cv2.LUT(plane, lut[:, c], dst=out[0, c])
    return out


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
    target = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points, target)
    crop = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    if crop.shape[0] / crop.shape[1] >= _ROTATE_ASPECT:
        crop = np.rot90(crop)
    return crop


def preprocess_text_recognition(
    images: Sequence[np.ndarray],
    image_height: int = 48,
    min_width: int = 320,
    max_width: int | None = None,
) -> np.ndarray:
    """
    Preprocess a batch of text line images for a CTC text recognition model.

    Every image is resized to ``image_height`` keeping its aspect ratio,
    normalized to [-1, 1] and right-padded with zeros to a common width: the
    widest aspect ratio in the batch (at least ``min_width``).

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

    Returns
    -------
    np.ndarray
        The float32 input tensor of shape (N, 3, image_height, W).

    """
    max_wh_ratio = max(
        [min_width / image_height] + [img.shape[1] / img.shape[0] for img in images],
    )
    batch_width = int(image_height * max_wh_ratio)
    if max_width is not None:
        batch_width = min(batch_width, max_width)
    batch = np.zeros((len(images), 3, image_height, batch_width), dtype=np.float32)
    for i, img in enumerate(images):
        resized_w = min(batch_width, math.ceil(image_height * img.shape[1] / img.shape[0]))
        resized = cv2.resize(img, (resized_w, image_height)).astype(np.float32)
        batch[i, :, :, :resized_w] = (resized.transpose(2, 0, 1) / 255 - 0.5) / 0.5
    return batch
