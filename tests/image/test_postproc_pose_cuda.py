# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CUDA pose postprocessing matches postprocess_yolo_pose."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.image.postprocessors._cpu_pose import postprocess_yolo_pose
from trtutils.image.postprocessors._cuda_pose import CUDAPosePostprocessor

_INPUT = (64, 64)


def _pose_head(rng: np.random.Generator, batch: int, num: int, nk: int) -> np.ndarray:
    """Clustered boxes with quantized scores, keypoints spilling past the input for clipping."""
    centers = rng.uniform(8, 56, size=(batch, 2, 6))
    pick = rng.integers(0, 6, size=(batch, num))
    xy = np.take_along_axis(centers, pick[:, None], axis=2) + rng.normal(0, 2, (batch, 2, num))
    wh = rng.uniform(4, 30, size=(batch, 2, num))
    scores = np.round(rng.uniform(0, 1, size=(batch, 1, num)), 2)
    kpts = rng.uniform(-10, 74, size=(batch, nk * 3, num))
    kpts[:, 2::3] = rng.uniform(0, 1, size=(batch, nk, num))
    return np.concatenate([xy, wh, scores, kpts], axis=1).astype(np.float32)


@pytest.mark.parametrize(
    ("batch", "num", "conf_thres", "nms_iou_thres", "max_det"),
    [
        pytest.param(1, 500, 0.25, 0.5, 300, id="default"),
        pytest.param(1, 500, None, 0.5, 300, id="no-conf"),
        pytest.param(1, 500, 0.99, 0.5, 300, id="nothing-passes"),
        pytest.param(1, 500, 0.1, 0.0, 300, id="iou-zero"),
        pytest.param(1, 500, 0.1, 0.9, 7, id="max-det"),
        pytest.param(2, 500, 0.25, 0.5, 300, id="batch2"),
        pytest.param(1, 5000, None, 0.5, 300, id="past-nms-rows"),
    ],
)
def test_cuda_pose_matches_cpu(
    cuda_stream, to_device, batch, num, conf_thres, nms_iou_thres, max_det
) -> None:
    """Boxes, scores, classes and every keypoint match postprocess_yolo_pose."""
    rng = np.random.default_rng(0)
    head = _pose_head(rng, batch, num=num, nk=17)
    ratios = [(0.6, 0.6), (0.35, 0.35)][:batch]
    padding = [(0.0, 6.0), (10.0, 0.0)][:batch]
    kwargs = {"max_det": max_det, "input_size": _INPUT}
    ref = postprocess_yolo_pose([head], ratios, padding, conf_thres, nms_iou_thres, **kwargs)

    outputs = [to_device(head)]
    post = CUDAPosePostprocessor(cuda_stream)
    # twice, so the second run reuses the result buffers
    for _ in range(2):
        ours = post(outputs, ratios, padding, conf_thres, nms_iou_thres, **kwargs)
        assert len(ours) == len(ref)
        for got, want in zip(ours, ref):
            assert len(got[1]) == len(want[1])
            for field, expected in zip(got, want):
                np.testing.assert_array_equal(field, expected)
