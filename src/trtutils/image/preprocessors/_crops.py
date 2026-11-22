# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Crop detection boxes out of an image."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from trtutils.image.types import Bbox


def crop_boxes(image: np.ndarray, boxes: np.ndarray | Sequence[Bbox]) -> list[np.ndarray]:
    """
    Crop boxes out of an image, without copying.

    Each box is clipped to the image. Float coordinates are widened to whole
    pixels: ``x1`` / ``y1`` are floored and ``x2`` / ``y2`` are ceiled, so a crop
    covers every pixel its box overlaps.

    Parameters
    ----------
    image : np.ndarray
        The image (HWC format) to crop.
    boxes : np.ndarray | Sequence[Bbox]
        The boxes as (x1, y1, x2, y2) in pixels of ``image``, an (N, 4) array or a
        sequence of Bbox.

    Returns
    -------
    list[np.ndarray]
        One crop per box in box order, each a view of ``image``: writing to a crop
        writes to the image. No boxes gives an empty list.

    Raises
    ------
    ValueError
        If ``boxes`` is not (N, 4), or a box has no area inside the image.

    """
    coords = np.asarray(boxes, dtype=np.float64)
    if coords.size == 0:
        return []
    if coords.ndim != 2 or coords.shape[1] != 4:  # noqa: PLR2004
        err_msg = f"boxes must be (N, 4) as (x1, y1, x2, y2), got shape {coords.shape}"
        raise ValueError(err_msg)
    height, width = image.shape[:2]
    limits = (width, height)
    starts = np.clip(np.floor(coords[:, :2]), 0, limits).astype(np.int64).tolist()
    ends = np.clip(np.ceil(coords[:, 2:]), 0, limits).astype(np.int64).tolist()
    crops: list[np.ndarray] = []
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(starts, ends)):
        if x2 <= x1 or y2 <= y1:
            err_msg = f"box {i} {coords[i].tolist()} has no area inside the {width}x{height} image"
            raise ValueError(err_msg)
        crops.append(image[y1:y2, x1:x2])
    return crops
