# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import cv2
import numpy as np

from trtutils._log import LOG


def postprocess_restoration(
    outputs: list[np.ndarray],
    sizes: list[tuple[int, int]] | None = None,
    scale: int = 1,
    *,
    no_copy: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Postprocess outputs from an image restoration network.

    Each image is clamped to [0, 1], scaled to 0-255 and rounded (as Real-ESRGAN does),
    turned from RGB CHW into BGR HWC uint8, and cropped to ``scale`` times its input size.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs of the network, the first is the restored batch (B, 3, H, W) in RGB.
    sizes : list[tuple[int, int]], optional
        The (height, width) of each input image before it was padded to the engine
        input. By default None, which keeps the whole output.
    scale : int
        The upscale factor of the network, the output size over the input size.
        By default 1. Only used with ``sizes``.
    no_copy : bool, optional
        Kept for interface symmetry with the other postprocessors. The images are
        always new arrays, so this has no effect.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[np.ndarray]]
        The postprocessed outputs per image, each a single (height, width, 3) uint8 image.

    Raises
    ------
    ValueError
        If ``sizes`` does not have one entry per image.

    """
    batch = outputs[0]
    if verbose:
        LOG.debug(f"Restoration postprocess, output shape: {batch.shape}, scale: {scale}")
    if sizes is not None and len(sizes) != batch.shape[0]:
        err_msg = f"Expected {batch.shape[0]} image sizes, got {len(sizes)}"
        raise ValueError(err_msg)

    results = []
    # one float32 plane is reused for the three channels
    plane = np.empty(batch.shape[2:], dtype=np.float32)
    for i in range(batch.shape[0]):
        height, width = batch.shape[2:]
        if sizes is not None:
            height, width = sizes[i][0] * scale, sizes[i][1] * scale
        work = plane[:height, :width]
        # BGR planes, merged into HWC by OpenCV
        channels = []
        for c in (2, 1, 0):
            np.clip(batch[i, c, :height, :width], 0.0, 1.0, out=work)
            work *= 255.0
            np.rint(work, out=work)
            channels.append(work.astype(np.uint8))
        results.append([cv2.merge(channels)])
    return results


def get_restored_images(
    outputs: list[list[np.ndarray]],
    *,
    verbose: bool | None = None,
) -> list[np.ndarray]:
    """
    Get the restored images from postprocessed image restoration outputs.

    Parameters
    ----------
    outputs : list[list[np.ndarray]]
        The postprocessed outputs per image from an image restoration network.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[np.ndarray]
        The restored image per image, each (height, width, 3) uint8 BGR.

    """
    if verbose:
        LOG.debug("Getting restored images")

    return [image_outputs[0] for image_outputs in outputs]
