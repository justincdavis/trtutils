# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from trtutils._log import LOG

if TYPE_CHECKING:
    from trtutils.image.types import Embedding, FeatureMap


def postprocess_features(
    outputs: list[np.ndarray],
    *,
    normalize: bool = False,
    no_copy: bool | None = None,
    verbose: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Postprocess outputs from a feature extraction network.

    Parameters
    ----------
    outputs : list[np.ndarray]
        The outputs from a feature extraction network with batch dimension.
        The first output is the embedding, reshaped to (B, C); any further outputs,
        such as a (B, C, h, w) feature map, are kept as they are.
    normalize : bool, optional
        Whether to L2 normalize the embeddings. The feature maps are never
        normalized. Default is False.
    no_copy : bool, optional
        If True, the outputs after the embedding (the feature maps) are returned as
        views of the engine memory when they need no conversion; those views WILL BE
        OVERWRITTEN INPLACE by future inference calls. The embeddings are always
        copied, they are a few KB and end2end returns them.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[np.ndarray]]
        The postprocessed outputs per image: the (C,) float32 embedding first,
        then the remaining outputs per image as float32, e.g. a (C, h, w) feature map.

    """
    if verbose:
        LOG.debug(f"Feature postprocess, output shapes: {[out.shape for out in outputs]}")

    # always a new array: end2end hands the embeddings to the caller without a copy
    embeddings = outputs[0].reshape(outputs[0].shape[0], -1).astype(np.float32)
    if normalize:
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        # a zero vector stays zero instead of turning into NaN
        embeddings = embeddings / np.maximum(norms, np.float32(1e-12))
    rest = [out.astype(np.float32, copy=not no_copy) for out in outputs[1:]]
    return [[embeddings[i]] + [out[i] for out in rest] for i in range(embeddings.shape[0])]


def get_embeddings(
    outputs: list[list[np.ndarray]],
    *,
    verbose: bool | None = None,
) -> list[Embedding]:
    """
    Get the embeddings from postprocessed feature extraction outputs.

    Parameters
    ----------
    outputs : list[list[np.ndarray]]
        The postprocessed outputs per image from a feature extraction network.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[Embedding]
        The (C,) float32 embedding per image.

    """
    if verbose:
        LOG.debug("Getting embeddings")

    return [image_outputs[0] for image_outputs in outputs]


def get_feature_maps(
    outputs: list[list[np.ndarray]],
    *,
    verbose: bool | None = None,
) -> list[FeatureMap]:
    """
    Get the dense feature maps from postprocessed feature extraction outputs.

    Parameters
    ----------
    outputs : list[list[np.ndarray]]
        The postprocessed outputs per image from a feature extraction network
        that has a feature map output.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[FeatureMap]
        The (C, h, w) float32 feature map per image.

    Raises
    ------
    ValueError
        If the outputs hold no feature map.

    """
    if verbose:
        LOG.debug("Getting feature maps")

    if any(len(image_outputs) < 2 for image_outputs in outputs):  # noqa: PLR2004
        err_msg = "The outputs hold no feature map, only an embedding"
        raise ValueError(err_msg)
    return [image_outputs[1] for image_outputs in outputs]
