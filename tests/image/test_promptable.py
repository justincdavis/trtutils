# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for the CPU stages of promptable segmentation -- prompt encoding and mask postprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.image.postprocessors import (
    get_image_embeddings,
    postprocess_image_embeddings,
    postprocess_prompt_masks,
)
from trtutils.image.preprocessors import encode_prompts

pytestmark = pytest.mark.cpu

INPUT = 1024
LOW_RES = 256


@pytest.mark.parametrize(
    ("points", "labels", "boxes", "coords", "expected_labels"),
    [
        pytest.param(
            [[10, 20]],
            None,
            None,
            [[[20, 40], [0, 0]]],
            [[1, -1]],
            id="point-gets-padding-point",
        ),
        pytest.param(
            [[10, 20], [30, 5]],
            [1, 0],
            None,
            [[[20, 40], [60, 10], [0, 0]]],
            [[1, 0, -1]],
            id="points-with-labels",
        ),
        pytest.param(
            None,
            None,
            [1, 2, 3, 4],
            [[[2, 4], [6, 8]]],
            [[2, 3]],
            id="box-no-padding-point",
        ),
        pytest.param(
            [[10, 20]],
            [0],
            [1, 2, 3, 4],
            [[[20, 40], [2, 4], [6, 8]]],
            [[0, 2, 3]],
            id="point-and-box",
        ),
        pytest.param(
            [[10, 20]],
            None,
            [[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 1, 1]],
            [[[20, 40], [2, 4], [6, 8]], [[20, 40], [10, 12], [14, 16]], [[20, 40], [0, 0], [2, 2]]],
            [[1, 2, 3]] * 3,
            id="k-boxes-share-points",
        ),
    ],
)
def test_encode_prompts(points, labels, boxes, coords, expected_labels) -> None:
    """Points come first, boxes add labelled corners, and only box-less sets get a padding point."""
    got_coords, got_labels = encode_prompts(points, labels, boxes, (2.0, 2.0), (0.0, 0.0))
    np.testing.assert_array_equal(got_coords, np.array(coords, np.float32))
    np.testing.assert_array_equal(got_labels, np.array(expected_labels, np.float32))
    assert got_coords.dtype == np.float32
    assert got_labels.dtype == np.float32


def test_encode_prompts_ratio_and_padding() -> None:
    """Image pixels map as p * ratio + padding per axis; the padding point stays at the origin."""
    coords, labels = encode_prompts([[100, 50]], [1], None, (0.5, 0.25), (3.0, 7.0))
    np.testing.assert_allclose(coords, [[[53.0, 19.5], [0.0, 0.0]]])
    np.testing.assert_array_equal(labels, [[1.0, -1.0]])
    coords, _ = encode_prompts(None, None, [[100, 40, 200, 80]], (0.5, 0.25), (3.0, 7.0))
    np.testing.assert_allclose(coords, [[[53.0, 17.0], [103.0, 27.0]]])


def test_encode_prompts_no_boxes_is_empty_batch() -> None:
    """Zero boxes (e.g. a detector found nothing) give zero prompt sets, not padded ones."""
    coords, labels = encode_prompts(None, None, np.zeros((0, 4)), (1.0, 1.0), (0.0, 0.0))
    assert coords.shape == (0, 2, 2)
    assert labels.shape == (0, 2)


@pytest.mark.parametrize(
    ("points", "labels", "boxes", "match"),
    [
        pytest.param(None, None, None, "points, boxes", id="no-prompt"),
        pytest.param([[1, 2], [3, 4]], [1], None, "2 points but 1 labels", id="label-count"),
    ],
)
def test_encode_prompts_rejects(points, labels, boxes, match) -> None:
    """A prompt without points or boxes, or with mismatched labels, raises."""
    with pytest.raises(ValueError, match=match):
        encode_prompts(points, labels, boxes, (1.0, 1.0), (0.0, 0.0))


def _pad_geometry(height: int, width: int) -> tuple[tuple[float, float], tuple[int, int]]:
    """The ratios of a 'pad' resize and the (rows, cols) of the image region in the low-res mask."""
    r = min(INPUT / height, INPUT / width)
    rows = round(height * r) * LOW_RES // INPUT
    cols = round(width * r) * LOW_RES // INPUT
    return (r, r), (rows, cols)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((300, 600), id="landscape"),
        pytest.param((600, 300), id="portrait"),
        pytest.param((480, 480), id="square"),
    ],
)
def test_postprocess_prompt_masks_geometry(shape) -> None:
    """The padded region is cropped away and the image region is resized to the image shape."""
    height, width = shape
    ratios, (rows, cols) = _pad_geometry(height, width)
    # positive everywhere but the left half of the image region, and over all the padding
    logits = np.full((1, 4, LOW_RES, LOW_RES), 10.0, np.float32)
    logits[:, :, :rows, : cols // 2] = 10.0
    logits[:, :, :rows, cols // 2 : cols] = -10.0
    ious = np.array([[0.5, 0.9, 0.1, 0.1]], np.float32)

    ((mask, score),) = postprocess_prompt_masks(
        [ious, logits], (INPUT, INPUT), shape, ratios, (0.0, 0.0)
    )

    assert mask.shape == shape
    assert mask.dtype == np.uint8
    assert score == pytest.approx(0.9)
    # the left half is in the mask and the right half (padding next to it included) is not
    margin = max(width // 50, 2)
    assert mask[:, : width // 2 - margin].all()
    assert not mask[:, width // 2 + margin :].any()


@pytest.mark.parametrize(
    ("multimask", "index"),
    [
        pytest.param(True, 2, id="multimask-best-iou"),
        pytest.param(False, 0, id="single-mask"),
    ],
)
def test_postprocess_prompt_masks_selection(multimask, index) -> None:
    """Multimask picks the best predicted IoU among masks 1..3 (never 0), else mask 0."""
    # mask k is positive only in column block k, and mask 0 has the highest IoU
    logits = np.full((2, 4, LOW_RES, LOW_RES), -5.0, np.float32)
    for k in range(4):
        logits[:, k, :, k * 64 : (k + 1) * 64] = 5.0
    ious = np.array([[0.99, 0.2, 0.8, 0.3], [0.99, 0.2, 0.8, 0.3]], np.float32)

    masks = postprocess_prompt_masks(
        [ious, logits], (INPUT, INPUT), (INPUT, INPUT), (1.0, 1.0), (0.0, 0.0), multimask=multimask
    )

    assert len(masks) == 2
    for mask, score in masks:
        assert score == pytest.approx(ious[0, index])
        columns = np.flatnonzero(mask.any(axis=0))
        assert index * 256 - 4 <= columns.min() <= index * 256 + 4
        assert (index + 1) * 256 - 4 <= columns.max() <= (index + 1) * 256 + 4


@pytest.mark.parametrize(
    ("mask_thres", "expected"),
    [
        pytest.param(0.0, 1, id="above"),
        pytest.param(1.0, 0, id="below"),
    ],
)
def test_postprocess_prompt_masks_threshold(mask_thres, expected) -> None:
    """Pixels are in the mask when their logit is above mask_thres."""
    logits = np.full((1, 4, LOW_RES, LOW_RES), 0.5, np.float32)
    ious = np.zeros((1, 4), np.float32)
    ((mask, _),) = postprocess_prompt_masks(
        [ious, logits], (INPUT, INPUT), (64, 32), (16.0, 16.0), (0.0, 0.0), mask_thres=mask_thres
    )
    np.testing.assert_array_equal(mask, np.full((64, 32), expected, np.uint8))


@pytest.mark.parametrize("no_copy", [pytest.param(True, id="view"), pytest.param(False, id="copy")])
def test_image_embeddings_split(no_copy) -> None:
    """A batch of embeddings splits into one (1, C, H, W) array per image, copied unless no_copy."""
    embeddings = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)
    per_image = postprocess_image_embeddings([embeddings], no_copy=no_copy)
    result = get_image_embeddings(per_image)
    assert len(result) == 2
    for i, emb in enumerate(result):
        np.testing.assert_array_equal(emb, embeddings[i : i + 1])
        assert np.shares_memory(emb, embeddings) is no_copy
