# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="misc"
"""Tests for the DepthEstimator class."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
# Depth estimator models are optional - skip if not available
DEPTH_ONNX = DATA_DIR / "onnx" / "depth_anything_v2_small.onnx"


@pytest.fixture(scope="module")
def depth_engine(build_test_engine) -> Path:
    """Build and cache a depth estimator engine."""
    if not DEPTH_ONNX.exists():
        pytest.skip("Depth estimator ONNX not available")
    return build_test_engine(DEPTH_ONNX)


class TestDepthEstimatorInference:
    """Test DepthEstimator inference."""

    def test_run_single_image(self, depth_engine: Path, images) -> None:
        """run() with single image returns outputs."""
        horse_image = images["horse"].array
        from trtutils.image import DepthEstimator

        model = DepthEstimator(depth_engine, warmup=False)
        results = model.run([horse_image], postprocess=False)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_run_batch(self, depth_engine: Path, test_images: list[np.ndarray]) -> None:
        """run() with batch returns outputs."""
        from trtutils.image import DepthEstimator

        model = DepthEstimator(depth_engine, warmup=False)
        results = model.run(test_images, postprocess=False)
        assert isinstance(results, list)

    def test_output_is_depth_map(self, depth_engine: Path, images) -> None:
        """end2end() returns depth maps with spatial dimensions."""
        horse_image = images["horse"].array
        from trtutils.image import DepthEstimator

        model = DepthEstimator(depth_engine, warmup=False)
        depth_maps = model.end2end([horse_image])
        assert isinstance(depth_maps, list)
        assert len(depth_maps) == 1
        depth = depth_maps[0]
        assert isinstance(depth, np.ndarray)
        assert depth.ndim >= 2  # spatial dimensions preserved

    def test_depth_values_positive(self, depth_engine: Path, images) -> None:
        """Depth values should be positive (distance from camera)."""
        horse_image = images["horse"].array
        from trtutils.image import DepthEstimator

        model = DepthEstimator(depth_engine, warmup=False)
        depth_maps = model.end2end([horse_image])
        depth = depth_maps[0]
        # Depth values should be >= 0
        assert depth.min() >= 0.0
