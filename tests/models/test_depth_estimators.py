# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine-level tests for the depth estimator wrappers in src/trtutils/models/depth_estimators/."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import DATA_DIR, PREPROCESSORS
from trtutils._flags import FLAGS
from trtutils.image._schema import DepthEstimatorOutputSchema
from trtutils.models import (
    DepthAnythingV1,
    DepthAnythingV2,
    DepthAnythingV2Metric,
    DepthAnythingV3,
    DepthPro2,
)

_trt86_v1 = pytest.mark.skipif(
    FLAGS.TRT_VERSION < (10, 0), reason="TensorRT 8.6 cannot build the /If conditional shapes"
)
_trt86_v3 = pytest.mark.skipif(
    FLAGS.TRT_VERSION < (10, 0), reason="TensorRT 8.6 has no Gelu plugin for the v3 ONNX"
)

_V2 = DATA_DIR / "depth_anything_v2"
_V2_METRIC = DATA_DIR / "depth_anything_v2_metric"
_V3 = DATA_DIR / "depth_anything_v3"

# (wrapper, onnx, max depth in metres for metric models, None for relative ones)
DEPTH_ESTIMATORS = [
    pytest.param(
        DepthAnythingV1,
        DATA_DIR / "depth_anything_v1" / "depth_anything_v1_small_518.onnx",
        None,
        id="depth-anything-v1-small",
        marks=_trt86_v1,
    ),
    pytest.param(
        DepthAnythingV2,
        _V2 / "depth_anything_v2_small_518.onnx",
        None,
        id="depth-anything-v2-small",
    ),
    pytest.param(
        DepthAnythingV2Metric,
        _V2_METRIC / "depth_anything_v2_metric_indoor_small_518.onnx",
        20.0,
        id="depth-anything-v2-metric-indoor-small",
    ),
    pytest.param(
        DepthAnythingV2Metric,
        _V2_METRIC / "depth_anything_v2_metric_outdoor_small_518.onnx",
        80.0,
        id="depth-anything-v2-metric-outdoor-small",
    ),
    pytest.param(
        DepthAnythingV2Metric,
        _V2_METRIC / "depth_anything_v2_metric_indoor_base_518.onnx",
        20.0,
        id="depth-anything-v2-metric-indoor-base",
    ),
    pytest.param(
        DepthAnythingV2Metric,
        _V2_METRIC / "depth_anything_v2_metric_outdoor_large_518.onnx",
        80.0,
        id="depth-anything-v2-metric-outdoor-large",
    ),
    pytest.param(
        DepthAnythingV3,
        _V3 / "depth_anything_v3_small_518.onnx",
        None,
        id="depth-anything-v3-small",
        marks=_trt86_v3,
    ),
    pytest.param(
        DepthAnythingV3,
        _V3 / "depth_anything_v3_base_518.onnx",
        None,
        id="depth-anything-v3-base",
        marks=_trt86_v3,
    ),
    pytest.param(
        DepthAnythingV3,
        _V3 / "depth_anything_v3_large_518.onnx",
        None,
        id="depth-anything-v3-large",
        marks=_trt86_v3,
    ),
    pytest.param(
        DepthAnythingV3,
        _V3 / "depth_anything_v3_mono_large_518.onnx",
        None,
        id="depth-anything-v3-mono-large",
        marks=_trt86_v3,
    ),
]


@pytest.mark.parametrize(("model_cls", "onnx_path", "max_depth"), DEPTH_ESTIMATORS)
@pytest.mark.parametrize(("preprocessor", "cpu_resize"), PREPROCESSORS)
def test_depth_estimator_end2end(
    build_model_engine, images, model_cls, onnx_path, max_depth, preprocessor, cpu_resize
) -> None:
    """run(), get_depth_maps() and end2end() agree: relative maps in [0, 1], metric ones in metres."""
    engine = build_model_engine(model_cls, onnx_path)
    model = model_cls(engine, preprocessor=preprocessor, cpu_resize=cpu_resize, warmup=False)
    assert model.output_schema is DepthEstimatorOutputSchema.DEPTH_ANYTHING
    image = images["horse"]

    postprocessed = model.run(image.array)
    via_run = model.get_depth_maps(postprocessed)
    via_e2e = model.end2end(image.array)
    np.testing.assert_array_equal(via_run, via_e2e)

    assert via_run.ndim == 3
    assert via_run.shape[0] == 1
    assert np.all(np.isfinite(via_run))
    if max_depth is None:
        assert via_run.min() == 0.0
        assert via_run.max() == pytest.approx(1.0)
    else:
        # metric heads end in sigmoid * max_depth: the wrapper returns the engine's metres
        raw = model.run(image.array, postprocess=False)[0]
        np.testing.assert_array_equal(via_run, raw.reshape(via_run.shape))
        assert via_run.min() > 0.0
        assert via_run.max() <= max_depth


_DP2 = DATA_DIR / "depth_pro_2"

# (wrapper, onnx): metric depth plus the estimated focal length
DEPTH_PRO_ESTIMATORS = [
    pytest.param(DepthPro2, _DP2 / "depth_pro_2_small_576x1024.onnx", id="depth-pro-2-small"),
    pytest.param(DepthPro2, _DP2 / "depth_pro_2_large_576x1024.onnx", id="depth-pro-2-large"),
    pytest.param(DepthPro2, _DP2 / "depth_pro_2_huge_576x1024.onnx", id="depth-pro-2-huge"),
]


@pytest.mark.parametrize(("model_cls", "onnx_path"), DEPTH_PRO_ESTIMATORS)
@pytest.mark.parametrize(("preprocessor", "cpu_resize"), PREPROCESSORS)
def test_depth_pro_end2end(
    build_model_engine, images, model_cls, onnx_path, preprocessor, cpu_resize
) -> None:
    """Depth Pro returns metres; the focal length is each image's second output, in pixels."""
    engine = build_model_engine(model_cls, onnx_path)
    model = model_cls(engine, preprocessor=preprocessor, cpu_resize=cpu_resize, warmup=False)
    assert model.output_schema is DepthEstimatorOutputSchema.DEPTH_PRO
    image = images["horse"]

    postprocessed = model.run(image.array)
    via_run = model.get_depth_maps(postprocessed)
    via_e2e = model.end2end(image.array)
    np.testing.assert_array_equal(via_run, via_e2e)

    height, width = (int(s) for s in onnx_path.stem.rsplit("_", 1)[1].split("x"))
    assert via_run.shape == (1, height, width)
    # the horse photo has no sky: every pixel has a finite distance
    assert np.all(np.isfinite(via_run))
    assert via_run.min() > 0.0
    focal_length = postprocessed[1]
    assert focal_length.shape == (1, 1)
    # a field of view between 10 and 160 degrees
    assert 0.09 * height < focal_length.item() < 5.7 * height


@pytest.mark.parametrize(("model_cls", "onnx_path"), DEPTH_PRO_ESTIMATORS[:1])
def test_depth_pro_normalize_keeps_focal_length(
    build_model_engine, images, model_cls, onnx_path
) -> None:
    """normalize=True min-max scales the depth map only, not the focal length."""
    engine = build_model_engine(model_cls, onnx_path)
    image = images["horse"].array
    metric = model_cls(engine, warmup=False).run(image)
    normalized = model_cls(engine, warmup=False, normalize=True).run(image)
    assert normalized[0].min() == 0.0
    assert normalized[0].max() == pytest.approx(1.0)
    np.testing.assert_array_equal(normalized[1], metric[1])
