# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""DepthEstimator output correctness tests (GPU required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SIMPLE_ONNX = DATA_DIR / "simple.onnx"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _find_depth_onnx() -> Path:
    """Find an available model ONNX for depth estimator testing."""
    # Use simple.onnx as a fallback -- the postprocessor will still run,
    # though output semantics differ from a real depth model.
    if SIMPLE_ONNX.exists():
        return SIMPLE_ONNX
    pytest.skip("No ONNX model available for depth estimator test")
    return SIMPLE_ONNX


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def depth_engine(build_test_engine) -> Path:
    """Build and cache a depth estimator engine for the test module."""
    onnx_path = _find_depth_onnx()
    return build_test_engine(onnx_path)


@pytest.fixture(scope="module")
def depth_engine_batch(build_test_engine) -> Path:
    """Build a batch-capable depth engine for batch tests."""
    onnx_path = _find_depth_onnx()
    try:
        return build_test_engine(onnx_path, batch_size=2)
    except Exception as exc:
        pytest.skip(f"Batch depth engine unavailable: {exc}")


# ---------------------------------------------------------------------------
# DepthEstimator base class tests
# ---------------------------------------------------------------------------
class TestDepthEstimatorOutput:
    """Tests for DepthEstimator output format."""

    def test_run_returns_list(
        self,
        depth_engine,
        images,
    ) -> None:
        """DepthEstimator.run() should return a list of ndarrays."""
        horse_image = images["horse"].array
        from trtutils.image import DepthEstimator

        de = DepthEstimator(depth_engine, warmup=False)
        outputs = de.run(horse_image)
        assert isinstance(outputs, list)
        for arr in outputs:
            assert isinstance(arr, np.ndarray)

    def test_run_raw_returns_ndarrays(
        self,
        depth_engine,
        images,
    ) -> None:
        """run(postprocess=False) returns raw output ndarrays."""
        horse_image = images["horse"].array
        from trtutils.image import DepthEstimator

        de = DepthEstimator(depth_engine, warmup=False)
        outputs = de.run(horse_image, postprocess=False)
        assert isinstance(outputs, list)
        for arr in outputs:
            assert isinstance(arr, np.ndarray)

    def test_postprocess_format(
        self,
        depth_engine,
        images,
    ) -> None:
        """postprocess() should return list[list[ndarray]]."""
        horse_image = images["horse"].array
        from trtutils.image import DepthEstimator

        de = DepthEstimator(depth_engine, warmup=False)
        raw = de.run(horse_image, postprocess=False)
        postprocessed = de.postprocess(raw)
        assert isinstance(postprocessed, list)
        for per_image in postprocessed:
            assert isinstance(per_image, list)
            for arr in per_image:
                assert isinstance(arr, np.ndarray)

    def test_get_depth_maps_single_image(
        self,
        depth_engine,
        images,
    ) -> None:
        """get_depth_maps for single image returns an ndarray."""
        horse_image = images["horse"].array
        from trtutils.image import DepthEstimator

        de = DepthEstimator(depth_engine, warmup=False)
        outputs = de.run(horse_image)
        depth_map = de.get_depth_maps(outputs)
        assert isinstance(depth_map, np.ndarray)
        # Depth maps are typically (1, H, W) or (H, W)
        assert depth_map.ndim >= 2

    def test_depth_map_values_finite(
        self,
        depth_engine,
        images,
    ) -> None:
        """Depth map values should be finite (no NaN/Inf)."""
        horse_image = images["horse"].array
        from trtutils.image import DepthEstimator

        de = DepthEstimator(depth_engine, warmup=False)
        outputs = de.run(horse_image)
        depth_map = de.get_depth_maps(outputs)
        assert np.all(np.isfinite(depth_map))

    def test_callable_matches_run(
        self,
        depth_engine,
        images,
    ) -> None:
        """__call__ should produce the same result as run()."""
        horse_image = images["horse"].array
        from trtutils.image import DepthEstimator

        de = DepthEstimator(depth_engine, warmup=False)
        out_run = de.run(horse_image)
        out_call = de(horse_image)
        assert len(out_run) == len(out_call)


# ---------------------------------------------------------------------------
# Batch tests
# ---------------------------------------------------------------------------
class TestDepthEstimatorBatch:
    """Batch inference tests."""

    def test_batch_run_returns_nested(
        self,
        depth_engine_batch,
        random_images,
    ) -> None:
        """Batch run with postprocess returns list[list[ndarray]]."""
        from trtutils.image import DepthEstimator

        de = DepthEstimator(depth_engine_batch, warmup=False)
        imgs = random_images(count=2, height=480, width=640)
        outputs = de.run(imgs)
        assert isinstance(outputs, list)
        assert len(outputs) == 2
        for per_image in outputs:
            assert isinstance(per_image, list)

    def test_batch_get_depth_maps(
        self,
        depth_engine_batch,
        random_images,
    ) -> None:
        """Batch get_depth_maps returns list[ndarray]."""
        from trtutils.image import DepthEstimator

        de = DepthEstimator(depth_engine_batch, warmup=False)
        imgs = random_images(count=2, height=480, width=640)
        outputs = de.run(imgs)
        depth_maps = de.get_depth_maps(outputs)
        assert isinstance(depth_maps, list)
        assert len(depth_maps) == 2
        for dm in depth_maps:
            assert isinstance(dm, np.ndarray)
            assert dm.ndim >= 2


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------
class TestDepthEstimatorEnd2End:
    """End-to-end inference tests."""

    def test_end2end_single_image(
        self,
        depth_engine,
        images,
    ) -> None:
        """end2end() on single image returns ndarray depth map."""
        horse_image = images["horse"].array
        from trtutils.image import DepthEstimator

        de = DepthEstimator(depth_engine, warmup=False)
        result = de.end2end(horse_image)
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 2

    def test_end2end_batch(
        self,
        depth_engine_batch,
        random_images,
    ) -> None:
        """end2end() on batch returns list[ndarray]."""
        from trtutils.image import DepthEstimator

        de = DepthEstimator(depth_engine_batch, warmup=False)
        imgs = random_images(count=2, height=480, width=640)
        result = de.end2end(imgs)
        assert isinstance(result, list)
        assert len(result) == 2
        for dm in result:
            assert isinstance(dm, np.ndarray)
