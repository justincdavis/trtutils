# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import numpy as np

from trtutils._log import LOG


def postprocess_depth(
    outputs: list[np.ndarray],
    *,
    no_copy: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Postprocess outputs from a depth estimation network.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from a depth estimation network with batch dimension.
        Expected shape is (B, 1, H, W) or (B, H, W).
    no_copy : bool, optional
        Kept for interface symmetry with the other postprocessors. The depth maps are
        always new arrays, so this has no effect.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[np.ndarray]]
        The postprocessed depth maps per image.
        Each image has a single depth map of shape (1, H, W)
        with values normalized to [0, 1].

    """
    if verbose:
        LOG.debug(f"Depth postprocess, output shape: {outputs[0].shape}")

    # drop the channel axis so every depth model yields (B, H, W)
    outputs = [out[:, 0] if out.ndim == 4 and out.shape[1] == 1 else out for out in outputs]  # noqa: PLR2004
    batch_size = outputs[0].shape[0]
    results = []
    for i in range(batch_size):
        batch_outputs = [out[i : i + 1] for out in outputs]
        result = _postprocess_depth_core(batch_outputs)
        results.append(result)
    return results


def _postprocess_depth_core(outputs: list[np.ndarray]) -> list[np.ndarray]:
    # min-max normalize to [0, 1] into one new array each, divided in place
    results = []
    for output in outputs:
        d_min = output.min()
        # float32: 1e-8 underflows to 0 in float16, turning a constant map into NaN
        normed = np.subtract(output, d_min, dtype=np.float32)
        np.divide(normed, np.float32(output.max()) - np.float32(d_min) + 1e-8, out=normed)
        results.append(normed)
    return results


def get_depth_maps(
    outputs: list[list[np.ndarray]],
    *,
    verbose: bool | None = None,
) -> list[np.ndarray]:
    """
    Get the depth maps from postprocessed depth estimation outputs.

    Parameters
    ----------
    outputs : list[list[np.ndarray]]
        The postprocessed outputs per image from a depth estimation network.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[np.ndarray]
        The depth map per image, each of shape (1, H, W) with values in [0, 1].

    """
    if verbose:
        LOG.debug("Getting depth maps")

    return [image_outputs[0] for image_outputs in outputs]
