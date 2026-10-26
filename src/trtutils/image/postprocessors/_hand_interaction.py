# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from trtutils._log import LOG

from ._common import nms_indices, unletterbox_boxes

if TYPE_CHECKING:
    from trtutils.image.types import Bbox, HandInteraction


def postprocess_hand_interactions(
    outputs: list[np.ndarray],
    ratios: list[tuple[float, float]],
    padding: list[tuple[float, float]],
    conf_thres: float = 0.3,
    nms_iou_thres: float = 0.5,
    *,
    no_copy: bool | None = None,  # noqa: ARG001
    input_size: tuple[int, int] | None = None,
    verbose: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Postprocess outputs from a hand-object interaction network.

    Expects the unified engine output contract: [boxes (B,K,4), scores (B,K), labels
    (B,K), pair_probs (B,K,K,C), side (B,K)]; side is optional. Boxes are remapped out
    of letterboxed network-input coordinates, filtered by confidence, then deduplicated
    with class-aware NMS.

    Parameters
    ----------
    outputs : list[np.ndarray]
        Raw engine outputs [boxes, scores, labels, pair_probs, (side)].
    ratios : list[tuple[float, float]]
        Preprocessing resize ratios per image.
    padding : list[tuple[float, float]]
        Preprocessing padding per image.
    conf_thres : float
        Confidence threshold used both to filter candidates and as the NMS score
        threshold.
    nms_iou_thres : float
        IoU threshold for class-aware NMS.
    no_copy : bool, optional
        Kept for interface symmetry with the other postprocessors. NMS-based
        indexing always allocates new arrays, so this has no effect.
    input_size : tuple[int, int] | None
        Model input (width, height), used with ratios/padding to clip boxes to the image.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[np.ndarray]]
        One list per image: [bboxes (N,4), scores (N,), labels (N,), pair_probs
        (N,N,C), side (N,1) if present else (N,0)] in original image coordinates.

    """
    if verbose:
        LOG.debug(f"Hand interaction postprocess, output shape: {outputs[0].shape}")

    has_side = len(outputs) == 5  # noqa: PLR2004
    batch_size = outputs[0].shape[0]
    results = []
    for i in range(batch_size):
        boxes = outputs[0][i]
        scores = outputs[1][i]
        labels = outputs[2][i].astype(int)
        pair_probs = outputs[3][i]

        remapped = unletterbox_boxes(boxes, ratios[i], padding[i], input_size)
        keep = scores >= conf_thres

        boxes_f = remapped[keep]
        scores_f = scores[keep]
        labels_f = labels[keep]
        pair_probs_f = pair_probs[keep][:, keep]

        nms_idx = nms_indices(boxes_f, scores_f, labels_f, conf_thres, nms_iou_thres)

        out_boxes = boxes_f[nms_idx]
        out_scores = scores_f[nms_idx]
        out_labels = labels_f[nms_idx].astype(int)
        out_pairs = pair_probs_f[nms_idx][:, nms_idx]
        if has_side:
            side = outputs[4][i].astype(int)
            side_f = side[keep]
            out_side = side_f[nms_idx].astype(int).reshape(-1, 1)
        else:
            out_side = np.zeros((len(nms_idx), 0), dtype=int)

        results.append([out_boxes, out_scores, out_labels, out_pairs, out_side])
    return results


def get_interactions(
    outputs: list[list[np.ndarray]],
    pair_thres: float = 0.5,
    second_pair_thres: float | None = None,
    *,
    verbose: bool | None = None,
) -> list[list[HandInteraction]]:
    """
    Pair hands with objects from postprocessed hand-object interaction outputs.

    For each hand (label 0), links to the first object (label 1) with the highest
    link probability above pair_thres, then from that object to the second object
    (label 2) with the highest link probability above second_pair_thres. Objects may
    be shared across hands.

    Parameters
    ----------
    outputs : list[list[np.ndarray]]
        Postprocessed outputs per image, as returned by postprocess_hand_interactions.
    pair_thres : float
        Minimum link probability to link a hand to a first object.
    second_pair_thres : float, optional
        Minimum link probability to link a first object to a second object.
        Defaults to pair_thres.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    list[list[HandInteraction]]
        Interactions per image, one HandInteraction (hand, first, second, side, contact) per hand.

    """
    if second_pair_thres is None:
        second_pair_thres = pair_thres

    results = []
    for image_outputs in outputs:
        if verbose:
            LOG.debug(f"Pairing interactions, num candidates: {len(image_outputs[0])}")
        results.append(_get_interactions_core(image_outputs, pair_thres, second_pair_thres))
    return results


def _get_interactions_core(
    image_outputs: list[np.ndarray],
    pair_thres: float,
    second_pair_thres: float,
) -> list[HandInteraction]:
    boxes, scores, labels, pair_probs, side = image_outputs
    num_classes = pair_probs.shape[-1]
    has_side = side.shape[1] > 0
    link = 1.0 - pair_probs[..., 0]

    hand_idx = np.where(labels == 0)[0]
    obj_idx = np.where(labels == 1)[0]
    second_idx = np.where(labels == 2)[0]  # noqa: PLR2004

    results: list[HandInteraction] = []
    for i in hand_idx:
        hand_entry = (_to_bbox(boxes[i]), float(scores[i]))

        j = _best_link(link, i, obj_idx, pair_thres)
        obj_entry = None
        second_entry = None
        contact = None

        if j is not None:
            obj_entry = (_to_bbox(boxes[j]), float(scores[j]))

            k = _best_link(link, j, second_idx, second_pair_thres)
            if k is not None:
                second_entry = (_to_bbox(boxes[k]), float(scores[k]))

            if num_classes > 2:  # noqa: PLR2004
                contact = int(np.argmax(pair_probs[i, j, 1:])) + 1
        elif num_classes > 2:  # noqa: PLR2004
            contact = 0

        side_i = int(side[i, 0]) if has_side else None

        results.append((hand_entry, obj_entry, second_entry, side_i, contact))

    return results


def _best_link(
    link: np.ndarray,
    row: int,
    candidates: np.ndarray,
    thres: float,
) -> int | None:
    if candidates.shape[0] == 0:
        return None
    links = link[row, candidates]
    best = int(np.argmax(links))
    if links[best] < thres:
        return None
    return int(candidates[best])


def _to_bbox(box: np.ndarray) -> Bbox:
    return (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
