# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CUDA hand-interaction postprocessing matches postprocess_hand_interactions."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.image.postprocessors._gpu_hoi import CudaHoiPostprocessor
from trtutils.image.postprocessors._hand_interaction import postprocess_hand_interactions

_RATIOS = [(1.0, 1.0), (0.37, 0.21)]
_PADDING = [(0.0, 0.0), (3.3, 1.7)]
_INPUT = (64, 48)


def _outputs(conf: float, has_side: bool, ids: type) -> list[np.ndarray]:
    # image 0 is an identity letterbox so NMS sees these corners
    boxes = np.zeros((2, 8, 4), np.float32)
    scores = np.zeros((2, 8), np.float32)
    labels = np.zeros((2, 8), ids)
    boxes[0, 0] = (10, 10, 40, 40)
    scores[0, 0] = 0.95
    boxes[0, 1] = (12, 12, 42, 42)  # same class, high overlap
    scores[0, 1] = 0.80
    boxes[0, 2] = (12, 12, 42, 42)  # other class, kept
    scores[0, 2] = 0.70
    labels[0, 2] = 1
    boxes[0, 3] = (0, 0, 10, 10)
    scores[0, 3] = 0.90
    labels[0, 3] = 2
    boxes[0, 4] = (3, 0, 13, 10)  # IoU ~0.54 with box 3
    scores[0, 4] = 0.55
    labels[0, 4] = 2
    boxes[0, 5] = (50, 30, 60, 40)
    scores[0, 5] = np.float32(conf)  # equal to conf, OpenCV drops it
    boxes[0, 6] = (20, 0, 28, 6)
    scores[0, 6] = 0.50
    labels[0, 6] = 1
    boxes[0, 7] = (30, 0, 38, 6)  # tie with box 6, stable order
    scores[0, 7] = 0.50
    labels[0, 7] = 1
    rng = np.random.default_rng(0)
    boxes[1, :, :2] = rng.uniform(0, 40, (8, 2))
    boxes[1, :, 2:] = boxes[1, :, :2] + rng.uniform(1, 25, (8, 2))
    boxes[1, 0] = (0, 0, 2, 2)  # falls in the padding and clips to 0
    boxes[1, 1] = (70, 50, 90, 70)  # clips to the image
    scores[1] = rng.random(8).astype(np.float32)
    scores[1, 2] = 0.05  # below conf
    labels[1] = rng.integers(0, 3, 8)
    pairs = rng.random((2, 8, 8, 3)).astype(np.float32)
    outputs = [boxes, scores, labels, pairs]
    if has_side:
        outputs.append((np.arange(16).reshape(2, 8) % 2).astype(ids))
    return outputs


@pytest.mark.parametrize("input_size", [_INPUT, None], ids=["clip", "no-input-size"])
@pytest.mark.parametrize("ids", [np.int32, np.int64, np.float32])
@pytest.mark.parametrize(
    "has_side",
    [
        pytest.param(True, id="with-side"),
        pytest.param(False, id="no-side"),
    ],
)
@pytest.mark.parametrize(
    ("conf", "iou"),
    [
        pytest.param(0.3, 0.5, id="conf-0.3-iou-0.5"),
        pytest.param(0.3, 0.99, id="conf-0.3-iou-0.99"),
        pytest.param(0.01, 0.5, id="conf-0.01-iou-0.5"),
    ],
)
def test_cuda_hoi_matches_cpu(
    cuda_stream, to_device, input_size, ids: type, has_side: bool, conf: float, iou: float
) -> None:
    """Unletterbox, the confidence filter, and class-aware NMS match the CPU path, dtypes too."""
    outputs = _outputs(conf, has_side, ids)
    cpu = postprocess_hand_interactions(
        [part.copy() for part in outputs], _RATIOS, _PADDING, conf, iou, input_size=input_size
    )
    gpu = CudaHoiPostprocessor(cuda_stream)(
        [to_device(part) for part in outputs], _RATIOS, _PADDING, conf, iou, input_size=input_size
    )
    assert len(gpu) == len(cpu)
    for got, exp in zip(gpu, cpu):
        assert len(got) == len(exp)
        # clip_boxes upcasts the CPU boxes to float64, the values still match
        np.testing.assert_array_equal(got[0], exp[0])
        for g, e in zip(got[1:], exp[1:]):
            assert g.dtype == e.dtype
            np.testing.assert_array_equal(g, e)
