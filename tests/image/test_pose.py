# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for the pose estimation postprocessor."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.image.postprocessors._pose import get_poses, postprocess_yolo_pose

pytestmark = pytest.mark.cpu


def _make_pred(boxes_cxcywh: list[list[float]], scores: list[float], kpts: np.ndarray) -> np.ndarray:
    """
    Build a raw (1, 5 + K*3, N) YOLO-style pose head output tensor.

    Parameters
    ----------
    boxes_cxcywh : list[list[float]]
        N rows of (cx, cy, w, h) in network coordinates.
    scores : list[float]
        N confidence scores.
    kpts : np.ndarray
        (N, K, 3) keypoints (x, y, visibility) in network coordinates.

    Returns
    -------
    np.ndarray
        (1, 5 + K*3, N) tensor matching the YOLO-style pose head layout.

    """
    n = len(boxes_cxcywh)
    k = kpts.shape[1]
    pred = np.empty((n, 5 + k * 3), dtype=np.float32)
    pred[:, :4] = np.array(boxes_cxcywh, dtype=np.float32)
    pred[:, 4] = np.array(scores, dtype=np.float32)
    pred[:, 5:] = kpts.reshape(n, k * 3)
    return pred.T[None]  # (1, 5+K*3, N)


def test_round_trip_postprocess_to_get_poses() -> None:
    """A single candidate flows through postprocess_yolo_pose then get_poses."""
    kpts = np.array([[[50.0, 50.0, 1.0], [60.0, 60.0, 0.5]]], dtype=np.float32)
    output = _make_pred([[50, 50, 20, 20]], [0.9], kpts)

    postprocessed = postprocess_yolo_pose(
        [output],
        ratios=[(1.0, 1.0)],
        padding=[(0.0, 0.0)],
        conf_thres=0.5,
    )
    assert len(postprocessed) == 1
    bboxes, scores, class_ids, _out_kpts = postprocessed[0]
    assert bboxes.shape == (1, 4)
    assert class_ids.tolist() == [0]

    poses = get_poses(postprocessed)
    assert len(poses) == 1
    assert len(poses[0]) == 1
    bbox, score, pose_kpts = poses[0][0]
    assert bbox == (40, 40, 60, 60)
    assert score == pytest.approx(0.9)
    np.testing.assert_allclose(pose_kpts, kpts[0])
    np.testing.assert_allclose(scores, [0.9])


def test_unletterbox_boxes_and_keypoints() -> None:
    """Boxes and keypoint x/y are unletterboxed identically with known ratio/padding."""
    # network coords: box center (100, 100), size (40, 40) -> xyxy (80,80,120,120)
    # keypoint at network (90, 90)
    kpts = np.array([[[90.0, 90.0, 1.0]]], dtype=np.float32)
    output = _make_pred([[100, 100, 40, 40]], [0.9], kpts)

    ratio = (2.0, 2.0)
    padding = (10.0, 20.0)
    postprocessed = postprocess_yolo_pose(
        [output],
        ratios=[ratio],
        padding=[padding],
        conf_thres=0.5,
    )
    bboxes, _scores, _class_ids, out_kpts = postprocessed[0]

    # orig = (coord - pad) / ratio
    expected_box = [(80 - 10) / 2.0, (80 - 20) / 2.0, (120 - 10) / 2.0, (120 - 20) / 2.0]
    np.testing.assert_allclose(bboxes[0], expected_box)

    expected_kpt_xy = [(90 - 10) / 2.0, (90 - 20) / 2.0]
    np.testing.assert_allclose(out_kpts[0, 0, :2], expected_kpt_xy)


def test_visibility_channel_passthrough() -> None:
    """The keypoint visibility channel is left untouched by unletterboxing."""
    kpts = np.array([[[50.0, 50.0, 0.37], [10.0, 10.0, 0.81]]], dtype=np.float32)
    output = _make_pred([[50, 50, 20, 20]], [0.9], kpts)

    postprocessed = postprocess_yolo_pose(
        [output],
        ratios=[(1.5, 1.5)],
        padding=[(5.0, 5.0)],
        conf_thres=0.5,
    )
    _bboxes, _scores, _class_ids, out_kpts = postprocessed[0]
    np.testing.assert_allclose(out_kpts[0, :, 2], [0.37, 0.81])


def test_keypoints_are_not_clipped() -> None:
    """An occluded keypoint predicted outside the frame is preserved, unlike boxes."""
    # keypoint at network (-40, -40) unletterboxes to a negative coordinate
    kpts = np.array([[[-40.0, -40.0, 0.05]]], dtype=np.float32)
    output = _make_pred([[50, 50, 20, 20]], [0.9], kpts)

    postprocessed = postprocess_yolo_pose(
        [output],
        ratios=[(1.0, 1.0)],
        padding=[(0.0, 0.0)],
        conf_thres=0.5,
    )
    bboxes, _scores, _class_ids, out_kpts = postprocessed[0]

    assert (bboxes >= 0).all()
    np.testing.assert_allclose(out_kpts[0, 0, :2], [-40.0, -40.0])


def test_nms_dedupes_overlapping_boxes() -> None:
    """Two heavily overlapping boxes are deduped by NMS, keeping the higher score."""
    kpts = np.zeros((2, 1, 3), dtype=np.float32)
    output = _make_pred(
        [[50, 50, 20, 20], [51, 51, 20, 20]],
        [0.9, 0.6],
        kpts,
    )

    postprocessed = postprocess_yolo_pose(
        [output],
        ratios=[(1.0, 1.0)],
        padding=[(0.0, 0.0)],
        conf_thres=0.5,
        nms_iou_thres=0.5,
    )
    bboxes, scores, _class_ids, _kpts = postprocessed[0]
    assert bboxes.shape[0] == 1
    assert scores[0] == pytest.approx(0.9)


def test_batch_of_two_images() -> None:
    """A batch of 2 images returns 2 postprocessed entries and 2 pose lists."""
    # both images carry the same fixed anchor count N, matching real engine output
    kpts0 = np.array([[[50.0, 50.0, 1.0]], [[0.0, 0.0, 0.0]]], dtype=np.float32)
    kpts1 = np.array([[[30.0, 30.0, 1.0]], [[70.0, 70.0, 1.0]]], dtype=np.float32)

    pred0 = _make_pred([[50, 50, 20, 20], [200, 200, 10, 10]], [0.9, 0.1], kpts0)[0]
    pred1 = _make_pred([[30, 30, 10, 10], [70, 70, 10, 10]], [0.8, 0.7], kpts1)[0]
    batched = np.stack([pred0, pred1])

    postprocessed = postprocess_yolo_pose(
        [batched],
        ratios=[(1.0, 1.0), (1.0, 1.0)],
        padding=[(0.0, 0.0), (0.0, 0.0)],
        conf_thres=0.5,
    )
    assert len(postprocessed) == 2
    assert postprocessed[0][0].shape[0] == 1
    assert postprocessed[1][0].shape[0] == 2

    poses = get_poses(postprocessed)
    assert len(poses) == 2
    assert len(poses[0]) == 1
    assert len(poses[1]) == 2


def test_non_standard_keypoint_count_is_derived() -> None:
    """K is derived from the output shape, not hardcoded to 17."""
    k = 5
    kpts = np.random.default_rng(0).random((1, k, 3)).astype(np.float32)
    output = _make_pred([[50, 50, 20, 20]], [0.9], kpts)
    assert output.shape[1] == 5 + k * 3

    postprocessed = postprocess_yolo_pose(
        [output],
        ratios=[(1.0, 1.0)],
        padding=[(0.0, 0.0)],
        conf_thres=0.5,
    )
    _bboxes, _scores, _class_ids, out_kpts = postprocessed[0]
    assert out_kpts.shape == (1, k, 3)
