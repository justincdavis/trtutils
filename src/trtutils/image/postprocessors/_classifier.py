# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import numpy as np

from trtutils._jit import register_jit
from trtutils._log import LOG


def postprocess_classifications(
    outputs: list[np.ndarray],
    *,
    no_copy: bool | None = None,
    verbose: bool | None = None,
) -> list[np.ndarray]:
    """
    Postprocess outputs from a classification network.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from a classification network.
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
    list[np.ndarray]
        The postprocessed outputs.

    """
    if verbose:
        LOG.debug(f"Classification postprocess, output shape: {outputs[0].shape}")

    return _postprocess_classifications_core(outputs, no_copy=no_copy)


@register_jit(nogil=True)
def _postprocess_classifications_core(
    outputs: list[np.ndarray],
    *,
    no_copy: bool | None = None,
) -> list[np.ndarray]:
    # convert logits to probabilities
    for output in outputs:
        exp_values = np.exp(output - np.max(output, axis=-1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        output[:] = probabilities
    
    if no_copy:
        return outputs
    return [output.copy() for output in outputs]

def get_classifications(
    outputs: list[np.ndarray],
    top_k: int = 5,
    *,
    verbose: bool | None = None,
) -> list[tuple[int, float]]:
    """
    Get the classifications from the output of a classification network.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from a classification network.
    top_k : int, optional
        The number of top predictions to return. Default is 5.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[tuple[int, float]]
        The classifications where each entry is (class_id, confidence).

    """
    if verbose:
        LOG.debug(f"Getting top-{top_k} classifications")

    return _get_classifications_core(outputs, top_k)


@register_jit(nogil=True)
def _get_classifications_core(
    outputs: list[np.ndarray],
    top_k: int = 5,
) -> list[tuple[int, float]]:
    probabilities = outputs[0][0]  # (1000,) - flatten from (1, 1000)
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    return [(int(idx), float(probabilities[idx])) for idx in top_indices]
