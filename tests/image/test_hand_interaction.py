# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/postprocessors/_hand_interaction.py -- box remap, NMS and pairing."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.image.postprocessors import get_interactions, postprocess_hand_interactions

pytestmark = pytest.mark.cpu


def _f32(value: float) -> float:
    """Round-trip a float through float32 to match engine-output precision."""
    return float(np.float32(value))


def _pair_probs(k: int, num_classes: int, default_p0: float = 0.9) -> np.ndarray:
    """Build a (k, k, num_classes) pair probability tensor with a uniform default."""
    probs = np.zeros((k, k, num_classes), dtype=np.float32)
    probs[:, :, 0] = default_p0
    if num_classes > 1:
        probs[:, :, 1:] = (1.0 - default_p0) / (num_classes - 1)
    return probs


def _remap_case(*, with_side: bool) -> list[np.ndarray]:
    """Six candidates with two below conf_thres, optionally with a side output."""
    k = 6
    boxes = np.zeros((1, k, 4), dtype=np.float32)
    boxes[0, 0] = [20, 40, 40, 60]  # hand, kept
    boxes[0, 1] = [100, 140, 120, 160]  # first object, kept
    boxes[0, 2] = [200, 240, 220, 260]  # first object, below threshold
    boxes[0, 3] = [300, 340, 320, 360]  # second object, kept
    boxes[0, 4] = [400, 440, 420, 460]  # hand, below threshold
    boxes[0, 5] = [500, 540, 520, 560]  # first object, kept
    scores = np.array([[0.9, 0.9, 0.1, 0.5, 0.05, 0.4]], dtype=np.float32)
    labels = np.array([[0, 1, 1, 2, 0, 1]], dtype=np.int64)
    outputs = [boxes, scores, labels, _pair_probs(k, num_classes=2)[None]]
    if with_side:
        outputs.append(np.array([[0, 1, 0, 1, 1, 0]], dtype=np.int64))
    return outputs


def _overlap_case(labels: list[int]) -> list[np.ndarray]:
    """Two heavily overlapping boxes plus one far away box."""
    boxes = np.array(
        [[[10, 10, 50, 50], [12, 12, 52, 52], [200, 200, 240, 240]]],
        dtype=np.float32,
    )
    scores = np.array([[0.9, 0.6, 0.8]], dtype=np.float32)
    return [
        boxes,
        scores,
        np.array([labels], dtype=np.int64),
        _pair_probs(3, num_classes=2)[None],
    ]


_REMAPPED = [[20, 40, 60, 80], [180, 240, 220, 280], [580, 640, 620, 680], [980, 1040, 1020, 1080]]
_REMAPPED_SCORES = [0.9, 0.9, 0.5, 0.4]
_REMAPPED_LABELS = [0, 1, 2, 1]


@pytest.mark.parametrize(
    ("outputs", "ratios", "padding", "boxes", "scores", "labels", "side"),
    [
        pytest.param(
            _remap_case(with_side=True),
            (0.5, 0.5),
            (10.0, 20.0),
            _REMAPPED,
            _REMAPPED_SCORES,
            _REMAPPED_LABELS,
            np.array([[0], [1], [1], [0]]),
            id="remap-with-side",
        ),
        pytest.param(
            _remap_case(with_side=False),
            (0.5, 0.5),
            (10.0, 20.0),
            _REMAPPED,
            _REMAPPED_SCORES,
            _REMAPPED_LABELS,
            np.zeros((4, 0), dtype=int),
            id="remap-no-side",
        ),
        pytest.param(
            _overlap_case([0, 0, 1]),
            (1.0, 1.0),
            (0.0, 0.0),
            [[10, 10, 50, 50], [200, 200, 240, 240]],
            [0.9, 0.8],
            [0, 1],
            np.zeros((2, 0), dtype=int),
            id="nms-dedupes-same-class",
        ),
        pytest.param(
            _overlap_case([0, 1, 1]),
            (1.0, 1.0),
            (0.0, 0.0),
            [[10, 10, 50, 50], [200, 200, 240, 240], [12, 12, 52, 52]],
            [0.9, 0.8, 0.6],
            [0, 1, 1],
            np.zeros((3, 0), dtype=int),
            id="nms-keeps-other-class",
        ),
    ],
)
def test_postprocess_remaps_and_filters(
    outputs, ratios, padding, boxes, scores, labels, side
) -> None:
    """Boxes are remapped, low-confidence rows dropped and same-class overlaps deduped."""
    result = postprocess_hand_interactions(outputs, ratios=[ratios], padding=[padding])[0]
    out_boxes, out_scores, out_labels, out_pairs, out_side = result

    np.testing.assert_allclose(out_boxes, boxes)
    np.testing.assert_allclose(out_scores, scores)
    np.testing.assert_array_equal(out_labels, labels)
    assert out_pairs.shape == (len(boxes), len(boxes), 2)
    np.testing.assert_array_equal(out_side, side)


def _postprocessed(
    boxes: list[list[float]],
    scores: list[float],
    labels: list[int],
    pair_probs: np.ndarray,
    side: list[int] | None = None,
) -> list[np.ndarray]:
    """Build a single image's postprocessed output list for get_interactions."""
    n = len(boxes)
    side_arr = (
        np.array(side, dtype=int).reshape(n, 1) if side is not None else np.zeros((n, 0), dtype=int)
    )
    return [
        np.array(boxes, dtype=np.float32),
        np.array(scores, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        pair_probs,
        side_arr,
    ]


def _hand_obj_second_pairs(num_classes: int) -> np.ndarray:
    probs = _pair_probs(3, num_classes)
    probs[0, 1, 0] = 0.1  # hand -> first object linked
    probs[1, 2, 0] = 0.1  # first object -> second object linked
    return probs


def _pairing_cases() -> list[pytest.param]:
    cases = []

    # hand links to a first object which links to a second object
    cases.append(
        pytest.param(
            _postprocessed(
                [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
                [0.875, 0.75, 0.625],
                [0, 1, 2],
                _hand_obj_second_pairs(num_classes=2),
            ),
            0.5,
            None,
            [
                (
                    ((0, 0, 10, 10), _f32(0.875)),
                    ((20, 20, 30, 30), _f32(0.75)),
                    ((40, 40, 50, 50), _f32(0.625)),
                    None,
                    None,
                ),
            ],
            id="hand-obj-second",
        ),
    )

    # a single hand with no other detections at all
    cases.append(
        pytest.param(
            _postprocessed(
                [[5, 5, 15, 15]],
                [0.9375],
                [0],
                _pair_probs(1, num_classes=2),
            ),
            0.5,
            None,
            [(((5, 5, 15, 15), _f32(0.9375)), None, None, None, None)],
            id="hand-only",
        ),
    )

    # hand and object present, but link probability is below pair_thres
    below_thres = _pair_probs(2, num_classes=5)
    below_thres[0, 1, 0] = 0.8  # link = 0.2 < 0.5
    cases.append(
        pytest.param(
            _postprocessed(
                [[0, 0, 10, 10], [20, 20, 30, 30]],
                [0.875, 0.75],
                [0, 1],
                below_thres,
            ),
            0.5,
            None,
            [(((0, 0, 10, 10), _f32(0.875)), None, None, None, 0)],
            id="pair-below-thres",
        ),
    )

    # hand->object linked but object->second link is below second_pair_thres
    second_below = _hand_obj_second_pairs(num_classes=5)
    second_below[1, 2, 0] = 0.9  # link = 0.1 < 0.5
    second_below[0, 1] = [0.1, 0.05, 0.05, 0.7, 0.1]
    cases.append(
        pytest.param(
            _postprocessed(
                [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
                [0.875, 0.75, 0.625],
                [0, 1, 2],
                second_below,
            ),
            0.5,
            None,
            [
                (
                    ((0, 0, 10, 10), _f32(0.875)),
                    ((20, 20, 30, 30), _f32(0.75)),
                    None,
                    None,
                    3,
                ),
            ],
            id="second-below-thres",
        ),
    )

    # two hands linking to the same shared object
    shared = _pair_probs(3, num_classes=2)
    shared[0, 2, 0] = 0.1
    shared[1, 2, 0] = 0.1
    cases.append(
        pytest.param(
            _postprocessed(
                [[0, 0, 10, 10], [50, 50, 60, 60], [20, 20, 30, 30]],
                [0.875, 0.8125, 0.75],
                [0, 0, 1],
                shared,
            ),
            0.5,
            None,
            [
                (
                    ((0, 0, 10, 10), _f32(0.875)),
                    ((20, 20, 30, 30), _f32(0.75)),
                    None,
                    None,
                    None,
                ),
                (
                    ((50, 50, 60, 60), _f32(0.8125)),
                    ((20, 20, 30, 30), _f32(0.75)),
                    None,
                    None,
                    None,
                ),
            ],
            id="shared-object",
        ),
    )

    # linked hand -> object with a contact class read off the pair distribution
    contact_pairs = _pair_probs(2, num_classes=5)
    contact_pairs[0, 1] = [0.05, 0.1, 0.6, 0.1, 0.15]
    cases.append(
        pytest.param(
            _postprocessed(
                [[0, 0, 10, 10], [20, 20, 30, 30]],
                [0.875, 0.75],
                [0, 1],
                contact_pairs,
            ),
            0.5,
            None,
            [
                (
                    ((0, 0, 10, 10), _f32(0.875)),
                    ((20, 20, 30, 30), _f32(0.75)),
                    None,
                    None,
                    2,
                ),
            ],
            id="contact-from-pair",
        ),
    )

    # linked hand -> object but only two pair classes means no contact vocabulary
    no_contact = _pair_probs(2, num_classes=2)
    no_contact[0, 1, 0] = 0.1
    cases.append(
        pytest.param(
            _postprocessed(
                [[0, 0, 10, 10], [20, 20, 30, 30]],
                [0.875, 0.75],
                [0, 1],
                no_contact,
            ),
            0.5,
            None,
            [
                (
                    ((0, 0, 10, 10), _f32(0.875)),
                    ((20, 20, 30, 30), _f32(0.75)),
                    None,
                    None,
                    None,
                ),
            ],
            id="no-contact-c2",
        ),
    )

    # obj->second link of 0.6 clears pair_thres but not an explicit stricter second_pair_thres
    strict_second = _hand_obj_second_pairs(num_classes=2)
    strict_second[1, 2, 0] = 0.4
    # obj->second link of 0.4 misses pair_thres but clears an explicit looser second_pair_thres
    loose_second = _hand_obj_second_pairs(num_classes=2)
    loose_second[1, 2, 0] = 0.6
    hand_obj = (((0, 0, 10, 10), _f32(0.875)), ((20, 20, 30, 30), _f32(0.75)))
    second = ((40, 40, 50, 50), _f32(0.625))
    for probs, second_thres, expected_second, case_id in (
        (strict_second, 0.7, None, "second-thres-stricter"),
        (loose_second, 0.3, second, "second-thres-looser"),
    ):
        cases.append(
            pytest.param(
                _postprocessed(
                    [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
                    [0.875, 0.75, 0.625],
                    [0, 1, 2],
                    probs,
                ),
                0.5,
                second_thres,
                [(*hand_obj, expected_second, None, None)],
                id=case_id,
            ),
        )

    return cases


@pytest.mark.parametrize(
    ("image_outputs", "pair_thres", "second_pair_thres", "expected"), _pairing_cases()
)
def test_get_interactions_pairing(image_outputs, pair_thres, second_pair_thres, expected) -> None:
    """Hand-object-second pairing matches the expected tuples for each scenario."""
    result = get_interactions(
        [image_outputs], pair_thres=pair_thres, second_pair_thres=second_pair_thres
    )
    assert result == [expected]


def test_batch_round_trip_through_both_functions() -> None:
    """A batch of 2 images flows through postprocess then get_interactions correctly."""
    k = 4
    boxes0 = np.zeros((k, 4), dtype=np.float32)
    boxes0[0] = [20, 40, 40, 60]
    boxes0[1] = [100, 140, 120, 160]
    boxes0[2] = [300, 340, 320, 360]
    boxes0[3] = [1, 1, 2, 2]
    scores0 = np.array([0.9, 0.8, 0.7, 0.05], dtype=np.float32)
    labels0 = np.array([0, 1, 2, 1], dtype=np.int64)
    pair0 = _pair_probs(k, num_classes=2)
    pair0[0, 1, 0] = 0.1
    pair0[1, 2, 0] = 0.1
    side0 = np.array([0, 0, 0, 0], dtype=np.int64)

    k1 = 2
    boxes1 = np.zeros((k, 4), dtype=np.float32)
    boxes1[0] = [0, 0, 20, 20]
    boxes1[1] = [100, 100, 120, 120]
    scores1 = np.zeros(k, dtype=np.float32)
    scores1[:k1] = [0.95, 0.85]
    labels1 = np.zeros(k, dtype=np.int64)
    labels1[:k1] = [0, 1]
    pair1 = np.zeros((k, k, 2), dtype=np.float32)
    pair1[:k1, :k1] = _pair_probs(k1, num_classes=2)
    pair1[0, 1, 0] = 0.1
    side1 = np.zeros(k, dtype=np.int64)
    side1[:k1] = [1, 0]

    boxes = np.stack([boxes0, boxes1])
    scores = np.stack([scores0, scores1])
    labels = np.stack([labels0, labels1])
    pair_probs = np.stack([pair0, pair1])
    side = np.stack([side0, side1])

    ratios = [(0.5, 0.5), (1.0, 1.0)]
    padding = [(10.0, 20.0), (0.0, 0.0)]

    postprocessed = postprocess_hand_interactions(
        [boxes, scores, labels, pair_probs, side],
        ratios,
        padding,
    )
    assert len(postprocessed) == 2

    interactions = get_interactions(postprocessed)

    expected = [
        [
            (
                ((20, 40, 60, 80), _f32(0.9)),
                ((180, 240, 220, 280), _f32(0.8)),
                ((580, 640, 620, 680), _f32(0.7)),
                0,
                None,
            ),
        ],
        [
            (
                ((0, 0, 20, 20), _f32(0.95)),
                ((100, 100, 120, 120), _f32(0.85)),
                None,
                1,
                None,
            ),
        ],
    ]
    assert interactions == expected


@pytest.mark.parametrize(
    ("ratios", "padding", "expected"),
    [
        pytest.param((1.0, 1.0), (0.0, 0.0), [90, 90, 100, 100], id="identity"),
        # original image is 200 x 160
        pytest.param((0.5, 0.5), (0.0, 10.0), [180, 160, 200, 160], id="letterboxed"),
    ],
)
def test_input_size_clips_boxes(ratios, padding, expected) -> None:
    """A box overshooting the original image is clipped to it when input_size is given."""
    boxes = np.array([[[90, 90, 105, 105]]], dtype=np.float32)
    outputs = [
        boxes,
        np.array([[0.9]], dtype=np.float32),
        np.array([[0]], dtype=np.int64),
        _pair_probs(1, num_classes=2)[None],
    ]
    kwargs = {"ratios": [ratios], "padding": [padding], "input_size": (100, 100)}
    clipped = postprocess_hand_interactions(outputs, **kwargs)[0][0]
    np.testing.assert_allclose(clipped[0], expected)
    unclipped = postprocess_hand_interactions(outputs, ratios=[ratios], padding=[padding])[0][0]
    assert unclipped[0, 2] > clipped[0, 2]
