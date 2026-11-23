# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/postprocessors/_cpu_pose.py -- YOLO, RTMO and YuNet decoding."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.image.postprocessors._cpu_pose import (
    get_poses,
    postprocess_rtmo_pose,
    postprocess_yolo_pose,
    postprocess_yunet_pose,
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


# a 64 x 32 input: stride 8, 16 and 32 grids of 8 x 4, 4 x 2 and 2 x 1 cells
_YUNET_SIZE = (64, 32)


def _yunet_outputs(faces: list[tuple[int, int, float, float, list[float]]]) -> list[np.ndarray]:
    """Build YuNet's twelve heads for one image, each face (stride, cell, cls, obj, bbox)."""
    counts = {s: (_YUNET_SIZE[0] // s) * (_YUNET_SIZE[1] // s) for s in (8, 16, 32)}
    heads = {
        (name, s): np.zeros((1, n, channels), dtype=np.float32)
        for name, channels in (("cls", 1), ("obj", 1), ("bbox", 4), ("kps", 10))
        for s, n in counts.items()
    }
    for stride, cell, cls, obj, bbox in faces:
        heads["cls", stride][0, cell] = cls
        heads["obj", stride][0, cell] = obj
        heads["bbox", stride][0, cell] = bbox
        # landmark n sits 0.1 * n cells right of the cell and half a cell up
        heads["kps", stride][0, cell] = [v for n in range(5) for v in (0.1 * n, -0.5)]
    return [heads[name, s] for name in ("cls", "obj", "bbox", "kps") for s in (8, 16, 32)]


# box (20, 10, 36, 26) at stride 8 cell (3, 2), score sqrt(0.81 * 1) = 0.9
_FACE = (8, 19, 0.81, 1.0, [0.5, 0.25, np.log(2), np.log(2)])
# box (32, 0, 64, 32) at stride 32 cell (1, 0), cls clipped to 1: score sqrt(0.64) = 0.8
_FAR_FACE = (32, 1, 1.5, 0.64, [0.5, 0.5, 0.0, 0.0])
# box (24, 10, 40, 26) at stride 16 cell (2, 1), IoU 0.6 with _FACE, score 0.7
_DUPLICATE = (16, 6, 0.49, 1.0, [0.0, 0.125, 0.0, 0.0])


@pytest.mark.parametrize(
    ("conf_thres", "nms_iou_thres", "expected_scores"),
    [
        pytest.param(0.5, 0.3, [0.9, 0.8], id="nms-drops-duplicate"),
        pytest.param(0.5, 0.7, [0.9, 0.8, 0.7], id="loose-nms"),
        pytest.param(0.85, 0.3, [0.9], id="conf-filter"),
        pytest.param(None, 0.3, [0.9, 0.8], id="no-conf"),
    ],
)
def test_yunet_round_trip_postprocess_to_get_poses(
    conf_thres, nms_iou_thres, expected_scores
) -> None:
    """YuNet heads decode to faces like FaceDetectorYN, then NMS, unletterbox and get_poses."""
    postprocessed = postprocess_yunet_pose(
        _yunet_outputs([_FACE, _FAR_FACE, _DUPLICATE]),
        ratios=[(1.0, 1.0)],
        padding=[(0.0, 0.0)],
        conf_thres=conf_thres,
        nms_iou_thres=nms_iou_thres,
        input_size=_YUNET_SIZE,
    )
    bboxes, scores, class_ids, kpts = postprocessed[0]
    np.testing.assert_allclose(scores, expected_scores, rtol=1e-6)
    assert class_ids.tolist() == [0] * len(expected_scores)
    np.testing.assert_allclose(bboxes[0], [20.0, 10.0, 36.0, 26.0], rtol=1e-6)
    assert kpts.shape == (len(expected_scores), 5, 3)
    expected_kpts = [[(3 + 0.1 * n) * 8, 12.0, 0.9] for n in range(5)]
    np.testing.assert_allclose(kpts[0], expected_kpts, rtol=1e-6)
    if len(expected_scores) > 1:
        np.testing.assert_allclose(bboxes[1], [32.0, 0.0, 64.0, 32.0], rtol=1e-6)

    poses = get_poses(postprocessed, conf_thres=conf_thres)[0]
    assert len(poses) == len(expected_scores)
    assert poses[0][0] == (20, 10, 36, 26)
    np.testing.assert_allclose(poses[0][2], kpts[0])


def test_yunet_batch_maps_each_image_back() -> None:
    """Each image of a YuNet batch is unletterboxed with its own ratios and padding."""
    first = _yunet_outputs([_FACE])
    second = _yunet_outputs([])
    outputs = [np.concatenate([a, b]) for a, b in zip(first, second)]
    postprocessed = postprocess_yunet_pose(
        outputs,
        ratios=[(2.0, 2.0), (1.0, 1.0)],
        padding=[(2.0, 1.0), (0.0, 0.0)],
        conf_thres=0.5,
        nms_iou_thres=0.3,
        input_size=_YUNET_SIZE,
    )
    bboxes, _, _, kpts = postprocessed[0]
    np.testing.assert_allclose(bboxes, [[9.0, 4.5, 17.0, 12.5]], rtol=1e-6)
    np.testing.assert_allclose(kpts[0, 0], [11.0, 5.5, 0.9], rtol=1e-6)
    assert [len(result[1]) for result in postprocessed] == [1, 0]


def test_yunet_needs_input_size() -> None:
    """The YuNet grids come from the input size, so decoding without it raises."""
    with pytest.raises(ValueError, match="needs input_size"):
        postprocess_yunet_pose(_yunet_outputs([_FACE]), ratios=[(1.0, 1.0)], padding=[(0.0, 0.0)])
