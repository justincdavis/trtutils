# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine-level tests for the depth estimator wrappers in src/trtutils/models/depth_estimators/."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import DATA_DIR
from trtutils.core import Buffer, MemoryLocation
from trtutils.models import DepthAnythingV1, DepthAnythingV2, DepthAnythingV3

DEPTH_ESTIMATORS = [
    pytest.param(
        DepthAnythingV1,
        DATA_DIR / "depth_anything_v1" / "depth_anything_v1_small_518.onnx",
        id="depth-anything-v1-small",
    ),
    pytest.param(
        DepthAnythingV2,
        DATA_DIR / "depth_anything_v2" / "depth_anything_v2_small_518.onnx",
        id="depth-anything-v2-small",
    ),
    pytest.param(
        DepthAnythingV3,
        DATA_DIR / "depth_anything_v3" / "depth_anything_v3_mono_large_518.onnx",
        id="depth-anything-v3-mono-large",
    ),
]


@pytest.mark.parametrize(("model_cls", "onnx_path"), DEPTH_ESTIMATORS)
@pytest.mark.parametrize("preprocessor", ["cpu", "cuda", "trt"])
def test_depth_estimator_end2end(
    build_model_engine, images, model_cls, onnx_path, preprocessor
) -> None:
    """run(), get_depth_maps(), and end2end() agree and yield a finite depth map normalized to [0, 1]."""
    engine = build_model_engine(model_cls, onnx_path)
    model = model_cls(engine, preprocessor=preprocessor, warmup=False)
    image = images["horse"]

    postprocessed = model.run(image.array)
    via_run = model.get_depth_maps(postprocessed)
    via_e2e = model.end2end(image.array)
    np.testing.assert_array_equal(via_run, via_e2e)

    assert via_run.ndim == 3
    assert via_run.shape[0] == 1
    assert np.all(np.isfinite(via_run))
    assert via_run.min() == 0.0
    assert via_run.max() == pytest.approx(1.0)


@pytest.mark.parametrize(("model_cls", "onnx_path"), DEPTH_ESTIMATORS)
@pytest.mark.parametrize("preprocessor", ["cpu", "cuda", "trt"])
def test_depth_estimator_device_buffer_matches_ndarray(
    build_model_engine, images, model_cls, onnx_path, preprocessor
) -> None:
    """A device Buffer passes through preprocess() and run(postprocess=False) like an ndarray."""
    engine = build_model_engine(model_cls, onnx_path)
    model = model_cls(engine, preprocessor=preprocessor, warmup=False)
    image = images["horse"].array

    expected_tensor, expected_ratios, expected_padding = model.preprocess(image)
    expected_raw = model.run(image, postprocess=False)

    buf = Buffer.from_array(image, MemoryLocation.DEVICE)
    try:
        tensor, ratios, padding = model.preprocess(buf)
        raw = model.run(buf, postprocess=False)
    finally:
        buf.free()

    assert tensor.shape == expected_tensor.shape
    assert ratios == expected_ratios
    assert padding == expected_padding
    assert [o.shape for o in raw] == [o.shape for o in expected_raw]
