# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import numpy as np

from trtutils._jit import register_jit
from trtutils._log import LOG


def postprocess_depth(
    outputs: list[np.ndarray],
    *,
    no_copy: bool | None = None,
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
        If True, the outputs will not be copied out
        from the cuda allocated host memory. Instead,
        the host memory will be returned directly.
        This memory WILL BE OVERWRITTEN INPLACE
        by future inference calls.
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

    batch_size = outputs[0].shape[0]
    results = []
    for i in range(batch_size):
        batch_outputs = [out[i : i + 1] for out in outputs]
        result = _postprocess_depth_core(batch_outputs, no_copy=no_copy)
        results.append(result)
    return results


@register_jit(nogil=True)
def _postprocess_depth_core(
    outputs: list[np.ndarray],
    *,
    no_copy: bool | None = None,
) -> list[np.ndarray]:
    for output in outputs:
        # ensure (1, H, W) shape
        if output.ndim == 2 or (output.ndim == 3 and output.shape[0] != 1):  # noqa: PLR2004
            output = np.expand_dims(output, axis=0)  # noqa: PLW2901

        # min-max normalize to [0, 1]
        d_min = output.min()
        d_max = output.max()
        output[:] = (output - d_min) / (d_max - d_min + 1e-8)

    if no_copy:
        return outputs
    return [output.copy() for output in outputs]


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
