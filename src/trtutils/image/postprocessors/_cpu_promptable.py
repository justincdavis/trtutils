# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from trtutils._log import LOG

if TYPE_CHECKING:
    from trtutils.image.types import PromptMask


def postprocess_image_embeddings(
    outputs: list[np.ndarray],
    *,
    no_copy: bool | None = None,
    verbose: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Split the outputs of an image encoder into per-image outputs.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The image embeddings (B, C, H, W), then any other (B, ...) features of the
        encoder's schema (e.g. SAM 2's high-res features).
    no_copy : bool, optional
        If True, return views of the outputs instead of copies. Views are
        overwritten by the next inference when the outputs are engine memory.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[np.ndarray]]
        Per image, its (1, ...) outputs, the (1, C, H, W) embeddings first.

    """
    if verbose:
        LOG.debug(f"Image embedding postprocess, output shapes: {[o.shape for o in outputs]}")
    return [
        [out[i : i + 1] if no_copy else out[i : i + 1].copy() for out in outputs]
        for i in range(outputs[0].shape[0])
    ]


def get_image_embeddings(
    outputs: list[list[np.ndarray]],
    *,
    verbose: bool | None = None,
) -> list[np.ndarray]:
    """
    Get the image embeddings from postprocessed image encoder outputs.

    Parameters
    ----------
    outputs : list[list[np.ndarray]]
        The postprocessed outputs per image.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[np.ndarray]
        The (1, C, H, W) embeddings per image.

    """
    if verbose:
        LOG.debug("Getting image embeddings")
    return [image_outputs[0] for image_outputs in outputs]


def postprocess_prompt_masks(
    outputs: list[np.ndarray],
    input_size: tuple[int, int],
    image_shape: tuple[int, int],
    ratios: tuple[float, float],
    padding: tuple[float, float],
    *,
    multimask: bool = True,
    mask_thres: float = 0.0,
    verbose: bool | None = None,
) -> list[PromptMask]:
    """
    Postprocess the outputs of a SAM-style mask decoder into one mask per prompt set.

    Each set's mask is picked from its candidates (the single-mask output 0, or the best
    predicted IoU among the multimask outputs 1..), upsampled to the model input, cropped
    to the image region, resized to the image and thresholded.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The predicted IoU (B, M) and low-res mask logits (B, M, h, w), in that order.
    input_size : tuple[int, int]
        The (width, height) of the image encoder input.
    image_shape : tuple[int, int]
        The (height, width) of the original image.
    ratios : tuple[float, float]
        The (x, y) scale from the image to the model input.
    padding : tuple[float, float]
        The (x, y) offset of the image in the model input.
    multimask : bool
        Whether to pick the best of the multimask outputs instead of the single mask.
        By default True.
    mask_thres : float
        The logit above which a pixel is in the mask. By default 0.0.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[PromptMask]
        Per prompt set, the (H, W) uint8 mask (0 or 1) and its predicted IoU.

    """
    iou_predictions, low_res_masks = outputs
    height, width = image_shape
    left, top = round(padding[0]), round(padding[1])
    region_w, region_h = round(width * ratios[0]), round(height * ratios[1])
    if verbose:
        LOG.debug(
            f"Prompt mask postprocess: {low_res_masks.shape} -> {image_shape}, "
            f"region {region_w}x{region_h} at ({left}, {top})"
        )
    results: list[PromptMask] = []
    for ious, masks in zip(iou_predictions, low_res_masks):
        index = int(np.argmax(ious[1:])) + 1 if multimask else 0
        logits = cv2.resize(masks[index], input_size, interpolation=cv2.INTER_LINEAR)
        logits = logits[top : top + region_h, left : left + region_w]
        logits = cv2.resize(logits, (width, height), interpolation=cv2.INTER_LINEAR)
        results.append(((logits > mask_thres).astype(np.uint8), float(ious[index])))
    return results
