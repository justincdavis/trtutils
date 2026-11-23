# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import numpy as np

from trtutils._log import LOG

# SAM point labels of the box corners (top-left, bottom-right) and of the padding point;
# user points are 0 (background) or 1 (foreground)
_BOX_LABELS = (2.0, 3.0)
_PAD_LABEL = -1.0


def encode_prompts(
    points: np.ndarray | None,
    labels: np.ndarray | None,
    boxes: np.ndarray | None,
    ratios: tuple[float, float],
    padding: tuple[float, float],
    *,
    pad_boxes: bool = False,
    verbose: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode image-pixel prompts as the point inputs of a SAM-style mask decoder.

    Each prompt set holds the points followed by the two corners of its box (labels 2 and
    3). A set without a box gets a ``(0, 0)`` padding point labelled -1, as SAM's own
    predictor adds; with ``pad_boxes`` every set gets it, as SAM 2's predictor does.
    Coordinates map to the model input as ``p * ratio + padding`` per axis.

    Parameters
    ----------
    points : np.ndarray, optional
        The (N, 2) points as (x, y) in image pixels, shared by every prompt set.
    labels : np.ndarray, optional
        The (N,) point labels, 1 for foreground and 0 for background.
        By default None, which labels every point foreground.
    boxes : np.ndarray, optional
        One (4,) box or (K, 4) boxes as (x1, y1, x2, y2) in image pixels.
        Each box makes one prompt set.
    ratios : tuple[float, float]
        The (x, y) scale from the image to the model input.
    padding : tuple[float, float]
        The (x, y) offset of the image in the model input.
    pad_boxes : bool
        Whether prompt sets with a box get the padding point too, for SAM 2 decoders.
        By default False.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The (K, N', 2) float32 coordinates and (K, N') float32 labels, with K the number
        of boxes (1 without boxes) and N' the points per set.

    Raises
    ------
    ValueError
        If neither points nor boxes are given, or labels do not match the points.

    """
    if points is None and boxes is None:
        err_msg = "A prompt needs points, boxes, or both"
        raise ValueError(err_msg)
    scale = np.array(ratios, np.float32)
    offset = np.array(padding, np.float32)
    pts = np.zeros((0, 2), np.float32)
    lab = np.zeros(0, np.float32)
    if points is not None:
        pts = np.asarray(points, np.float32).reshape(-1, 2) * scale + offset
        lab = np.ones(len(pts), np.float32)
        if labels is not None:
            lab = np.asarray(labels, np.float32).reshape(-1)
        if len(lab) != len(pts):
            err_msg = f"Got {len(pts)} points but {len(lab)} labels"
            raise ValueError(err_msg)
    if boxes is None:
        coords, point_labels = pts[None], lab[None]
    else:
        corners = np.asarray(boxes, np.float32).reshape(-1, 2, 2) * scale + offset
        num_sets = len(corners)
        coords = np.concatenate([np.broadcast_to(pts, (num_sets, *pts.shape)), corners], axis=1)
        point_labels = np.concatenate(
            [
                np.broadcast_to(lab, (num_sets, len(lab))),
                np.broadcast_to(np.array(_BOX_LABELS, np.float32), (num_sets, 2)),
            ],
            axis=1,
        )
    if boxes is None or pad_boxes:
        num_sets = len(coords)
        coords = np.concatenate([coords, np.zeros((num_sets, 1, 2), np.float32)], axis=1)
        point_labels = np.concatenate(
            [point_labels, np.full((num_sets, 1), _PAD_LABEL, np.float32)], axis=1
        )
    if verbose:
        LOG.debug(f"Encoded prompts: coords {coords.shape}, labels {point_labels.shape}")
    return coords.astype(np.float32), point_labels.astype(np.float32)
