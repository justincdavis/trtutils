# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CUDA oriented-box postprocessing matches postprocess_yolo_obb."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.image.postprocessors._cpu_obb import postprocess_yolo_obb
from trtutils.image.postprocessors._cuda_obb import CUDAObbPostprocessor


def _obb_head(rng: np.random.Generator, batch: int, num: int, nc: int) -> np.ndarray:
    """Clustered rotated boxes with quantized scores, so probiou overlaps and score ties happen."""
    centers = rng.uniform(100, 900, size=(batch, 2, 8))
    pick = rng.integers(0, 8, size=(batch, num))
    xy = np.take_along_axis(centers, pick[:, None], axis=2) + rng.normal(0, 25, (batch, 2, num))
    wh = rng.uniform(30, 300, size=(batch, 2, num))
    scores = np.round(rng.uniform(0, 1, size=(batch, nc, num)), 2)
    angle = rng.uniform(-np.pi / 4, 3 * np.pi / 4, size=(batch, 1, num))
    return np.concatenate([xy, wh, scores, angle], axis=1).astype(np.float32)


@pytest.mark.parametrize(
    ("batch", "conf_thres", "nms_iou_thres", "max_det"),
    [
        pytest.param(1, 0.25, 0.5, 300, id="default"),
        pytest.param(1, None, 0.5, 300, id="no-conf"),
        pytest.param(1, 1.01, 0.5, 300, id="nothing-passes"),
        pytest.param(1, 0.1, 0.0, 300, id="iou-zero"),
        pytest.param(1, 0.1, 0.9, 7, id="max-det"),
        pytest.param(1, 0.1, 0.3, 2000, id="many-candidates"),
        pytest.param(2, 0.25, 0.5, 300, id="batch2"),
    ],
)
def test_cuda_obb_matches_cpu(
    cuda_stream, to_device, batch, conf_thres, nms_iou_thres, max_det
) -> None:
    """Enclosing boxes, scores, classes, and rotated boxes match postprocess_yolo_obb."""
    rng = np.random.default_rng(0)
    head = _obb_head(rng, batch, num=2000, nc=3)
    ratios = [(0.6, 0.6), (0.35, 0.35)][:batch]
    padding = [(0.0, 6.0), (10.0, 0.0)][:batch]
    ref = postprocess_yolo_obb([head], ratios, padding, conf_thres, nms_iou_thres, max_det=max_det)

    outputs = [to_device(head)]
    post = CUDAObbPostprocessor(cuda_stream)
    # twice, so the second run reuses the buffers
    for _ in range(2):
        ours = post(outputs, ratios, padding, conf_thres, nms_iou_thres, max_det=max_det)
        for got, want in zip(ours, ref):
            assert len(got[1]) == len(want[1])
            for k, (field, expected) in enumerate(zip(got, want)):
                assert field.dtype == expected.dtype
                # enclosing xyxy uses cos/sin; numpy's float32 cos is not correctly rounded
                if k == 0:
                    np.testing.assert_allclose(field, expected, rtol=1e-5, atol=1e-3)
                else:
                    np.testing.assert_array_equal(field, expected)
            assert got[2].dtype.kind == "i"


def test_cuda_obb_rejects_nonuniform_scale(cuda_stream, to_device) -> None:
    """Like the CPU path, a non-uniform resize ratio raises."""
    head = _obb_head(np.random.default_rng(0), 1, num=10, nc=3)
    with pytest.raises(ValueError, match="uniform"):
        CUDAObbPostprocessor(cuda_stream)(
            [to_device(head)], [(0.5, 0.25)], [(0.0, 0.0)], 0.1, 0.5, max_det=10
        )
