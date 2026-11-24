# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/postprocessors/_cpu_pose.py -- YOLO and RTMO keypoint decoding."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.image.postprocessors._cpu_pose import (
    get_poses,
    postprocess_rtmo_pose,
    postprocess_yolo_pose,
)

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


@pytest.mark.parametrize(
    ("kpt", "ratio", "padding", "bbox", "kpt_xy"),
    [
        pytest.param(
            [50.0, 50.0, 0.37], 1.0, (0.0, 0.0), (40, 40, 60, 60), [50.0, 50.0], id="identity"
        ),
        pytest.param(
            [90.0, 90.0, 0.37], 2.0, (10.0, 20.0), (15, 10, 25, 20), [40.0, 35.0], id="letterboxed"
        ),
        # occluded keypoints outside the frame are kept, unlike boxes
        pytest.param(
            [-40.0, -40.0, 0.05], 1.0, (0.0, 0.0), (40, 40, 60, 60), [-40.0, -40.0], id="offscreen"
        ),
    ],
)
def test_round_trip_postprocess_to_get_poses(kpt, ratio, padding, bbox, kpt_xy) -> None:
    """Boxes and keypoint x/y are unletterboxed, visibility passes through, into get_poses."""
    output = _make_pred([[50, 50, 20, 20]], [0.9], np.array([[kpt]], dtype=np.float32))

    postprocessed = postprocess_yolo_pose(
        [output], ratios=[(ratio, ratio)], padding=[padding], conf_thres=0.5
    )
    assert len(postprocessed) == 1
    bboxes, scores, class_ids, out_kpts = postprocessed[0]
    assert bboxes.shape == (1, 4)
    assert class_ids.tolist() == [0]
    np.testing.assert_allclose(scores, [0.9])
    np.testing.assert_allclose(out_kpts[0, 0], [*kpt_xy, kpt[2]])

    poses = get_poses(postprocessed)
    assert len(poses) == 1
    assert len(poses[0]) == 1
    pose_bbox, score, pose_kpts = poses[0][0]
    assert pose_bbox == bbox
    assert score == pytest.approx(0.9)
    np.testing.assert_allclose(pose_kpts, out_kpts[0])


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


@pytest.mark.parametrize(
    ("ratio", "padding", "bbox", "kpt_xy"),
    [
        pytest.param(1.0, (0.0, 0.0), [85, 85, 100, 100], [100.0, 0.0], id="identity"),
        # original image is 200 x 160
        pytest.param(0.5, (0.0, 10.0), [170, 150, 200, 160], [200.0, 0.0], id="letterboxed"),
    ],
)
def test_input_size_clips_boxes_and_keypoints(ratio, padding, bbox, kpt_xy) -> None:
    """Overshooting boxes and keypoints are clipped to the original image like ultralytics."""
    kpts = np.array([[[1000.0, -50.0, 0.9]]], dtype=np.float32)
    output = _make_pred([[95, 95, 20, 20]], [0.9], kpts)
    bboxes, _, _, out_kpts = postprocess_yolo_pose(
        [output],
        ratios=[(ratio, ratio)],
        padding=[padding],
        conf_thres=0.5,
        input_size=(100, 100),
    )[0]
    np.testing.assert_allclose(bboxes[0], bbox)
    np.testing.assert_allclose(out_kpts[0, 0], [*kpt_xy, 0.9], rtol=1e-6)


def _rtmo_outputs(rows: list[tuple[list[float], float, list[float]]]) -> list[np.ndarray]:
    """Build RTMO [dets (1, 50, 5), keypoints (1, 50, 1, 3)], unused rows zero like the engine."""
    dets = np.zeros((1, 50, 5), dtype=np.float32)
    kpts = np.zeros((1, 50, 1, 3), dtype=np.float32)
    for i, (box, score, kpt) in enumerate(rows):
        dets[0, i] = [*box, score]
        kpts[0, i, 0] = kpt
    return [dets, kpts]


@pytest.mark.parametrize(
    ("nms_iou_thres", "expected_scores"),
    [
        # the engine already ran NMS at 0.65, a stricter threshold drops the overlap
        pytest.param(0.5, [0.9], id="strict-nms"),
        pytest.param(0.65, [0.9, 0.8], id="engine-nms"),
    ],
)
def test_rtmo_round_trip_postprocess_to_get_poses(nms_iou_thres, expected_scores) -> None:
    """RTMO rows are filtered by score, NMS'd, unletterboxed and turned into poses."""
    outputs = _rtmo_outputs(
        [
            ([90.0, 120.0, 190.0, 220.0], 0.9, [110.0, 140.0, 0.7]),
            # IoU 0.6 with the first row
            ([115.0, 120.0, 215.0, 220.0], 0.8, [130.0, 150.0, 0.6]),
            ([300.0, 300.0, 400.0, 400.0], 0.05, [350.0, 350.0, 0.9]),
        ]
    )
    postprocessed = postprocess_rtmo_pose(
        outputs,
        ratios=[(0.5, 0.5)],
        padding=[(10.0, 20.0)],
        conf_thres=0.1,
        nms_iou_thres=nms_iou_thres,
        input_size=(640, 640),
    )
    bboxes, scores, class_ids, kpts = postprocessed[0]
    np.testing.assert_allclose(scores, expected_scores)
    assert class_ids.tolist() == [0] * len(expected_scores)
    np.testing.assert_allclose(bboxes[0], [160.0, 200.0, 360.0, 400.0])
    np.testing.assert_allclose(kpts[0, 0], [200.0, 240.0, 0.7], rtol=1e-6)

    poses = get_poses(postprocessed, conf_thres=0.1)[0]
    assert len(poses) == len(expected_scores)
    assert poses[0][0] == (160, 200, 360, 400)
    np.testing.assert_allclose(poses[0][2], kpts[0])


def test_rtmo_batch_of_two_images() -> None:
    """Each image of an RTMO batch keeps its own rows."""
    first = _rtmo_outputs([([10.0, 10.0, 50.0, 50.0], 0.9, [20.0, 20.0, 0.9])])
    second = _rtmo_outputs(
        [
            ([10.0, 10.0, 50.0, 50.0], 0.8, [20.0, 20.0, 0.9]),
            ([100.0, 100.0, 150.0, 150.0], 0.7, [120.0, 120.0, 0.9]),
        ]
    )
    outputs = [np.concatenate([a, b]) for a, b in zip(first, second)]
    postprocessed = postprocess_rtmo_pose(
        outputs, ratios=[(1.0, 1.0)] * 2, padding=[(0.0, 0.0)] * 2, conf_thres=0.5
    )
    assert [len(result[1]) for result in postprocessed] == [1, 2]
    np.testing.assert_allclose(postprocessed[1][3][1, 0], [120.0, 120.0, 0.9])
