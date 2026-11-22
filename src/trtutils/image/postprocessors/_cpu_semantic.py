# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from trtutils._log import LOG

from ._common import original_size

if TYPE_CHECKING:
    from trtutils.image.types import SemanticMask

# class ids are returned as uint8
_MAX_CLASSES = 256


def postprocess_class_map(
    outputs: list[np.ndarray],
    ratios: list[tuple[float, float]],
    padding: list[tuple[float, float]],
    input_size: tuple[int, int],
    *,
    image_shapes: list[tuple[int, int]] | None = None,
    verbose: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Postprocess the class map of a semantic segmentation network.

    Each class map is mapped back to its original image with a nearest-neighbour lookup
    that undoes the resize (and crops the letterbox padding), then cast to uint8.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The engine outputs, a single integer class map of shape (B, 1, h, w) or (B, h, w).
    ratios : list[tuple[float, float]]
        The rescale ratios used during preprocessing for each image.
    padding : list[tuple[float, float]]
        The padding values used during preprocessing for each image.
    input_size : tuple[int, int]
        The (width, height) of the model input.
    image_shapes : list[tuple[int, int]], optional
        The (height, width) of each original image. By default None, which recovers
        the sizes from the ratios and padding; letterbox rounding can make those
        a pixel or two off when the image was downscaled.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[np.ndarray]]
        Per image, a list holding its (H, W) uint8 class map at the original resolution.

    """
    class_maps = outputs[0]
    if verbose:
        LOG.debug(f"Semantic postprocess, class map shape: {class_maps.shape}")
    if class_maps.ndim == 4:  # noqa: PLR2004
        class_maps = class_maps[:, 0]
    results = []
    for idx, class_map in enumerate(class_maps):
        if image_shapes is not None:
            height, width = image_shapes[idx]
        else:
            width, height = original_size(ratios[idx], padding[idx], input_size)
        mapped = _unresize(
            class_map.astype(np.uint8), padding[idx], input_size, (max(height, 1), max(width, 1))
        )
        results.append([mapped])
    return results


def postprocess_semantic_logits(
    outputs: list[np.ndarray],
    ratios: list[tuple[float, float]],
    padding: list[tuple[float, float]],
    input_size: tuple[int, int],
    *,
    image_shapes: list[tuple[int, int]] | None = None,
    verbose: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Postprocess the logits of a semantic segmentation network.

    The argmax over the classes runs at the logits resolution, then the class map is
    mapped back as in :func:`postprocess_class_map`. Engines built with the argmax in the
    engine upsample the logits bilinearly first, so for logits smaller than the input
    the two paths can differ along class borders.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The engine outputs, a single logits tensor of shape (B, C, h, w).
    ratios : list[tuple[float, float]]
        The rescale ratios used during preprocessing for each image.
    padding : list[tuple[float, float]]
        The padding values used during preprocessing for each image.
    input_size : tuple[int, int]
        The (width, height) of the model input.
    image_shapes : list[tuple[int, int]], optional
        The (height, width) of each original image, see :func:`postprocess_class_map`.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[np.ndarray]]
        Per image, a list holding its (H, W) uint8 class map at the original resolution.

    Raises
    ------
    ValueError
        If the logits have more than 256 classes, which do not fit in uint8.

    """
    logits = outputs[0]
    if verbose:
        LOG.debug(f"Semantic postprocess, logits shape: {logits.shape}")
    if logits.shape[1] > _MAX_CLASSES:
        err_msg = f"Semantic class maps are uint8, got {logits.shape[1]} classes (max 256)"
        raise ValueError(err_msg)
    return postprocess_class_map(
        [logits.argmax(axis=1)],
        ratios,
        padding,
        input_size,
        image_shapes=image_shapes,
        verbose=verbose,
    )


def _unresize(
    class_map: np.ndarray,
    padding: tuple[float, float],
    input_size: tuple[int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    # original pixel center -> input pixel -> class map cell, nearest neighbour. the image
    # spans input_size - 2 * padding pixels from the offset the preprocessors pad with
    # (negative padding is the center crop offset)
    map_h, map_w = class_map.shape
    height, width = shape
    input_w, input_h = input_size
    pad_x, pad_y = padding
    cols = round(pad_x - 0.1) + (np.arange(width) + 0.5) * ((input_w - 2 * pad_x) / width)
    rows = round(pad_y - 0.1) + (np.arange(height) + 0.5) * ((input_h - 2 * pad_y) / height)
    cols = np.clip((cols * (map_w / input_w)).astype(np.int64), 0, map_w - 1)
    rows = np.clip((rows * (map_h / input_h)).astype(np.int64), 0, map_h - 1)
    return class_map.take(rows, axis=0).take(cols, axis=1)


def get_class_maps(
    outputs: list[list[np.ndarray]],
    *,
    verbose: bool | None = None,
) -> list[SemanticMask]:
    """
    Get the class maps from postprocessed semantic segmentation outputs.

    Parameters
    ----------
    outputs : list[list[np.ndarray]]
        The postprocessed outputs per image.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[SemanticMask]
        The (H, W) uint8 class map per image.

    """
    if verbose:
        LOG.debug("Getting class maps")

    return [image_outputs[0] for image_outputs in outputs]
