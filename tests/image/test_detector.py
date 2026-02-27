# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="misc"
"""Tests for the Detector class."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
YOLOV10_ONNX = DATA_DIR / "yolov10" / "yolov10n_640.onnx"


@pytest.fixture(scope="module")
def yolov10_engine(build_test_engine) -> Path:
    """Build and cache a YOLOv10n engine for the test module."""
    if not YOLOV10_ONNX.exists():
        pytest.skip("yolov10n_640.onnx not available")
    return build_test_engine(YOLOV10_ONNX)


@pytest.mark.gpu
class TestDetectorInference:
    """Test Detector inference modes."""

    def test_run_returns_outputs(self, yolov10_engine: Path, horse_image: np.ndarray) -> None:
        """run() with postprocess=False returns list of raw output arrays."""
        from trtutils.models import YOLOv10

        det = YOLOv10(yolov10_engine, warmup=False)
        results = det.run([horse_image], postprocess=False)
        assert isinstance(results, list)
        assert len(results) > 0
        assert isinstance(results[0], np.ndarray)

    def test_run_with_postprocess(self, yolov10_engine: Path, horse_image: np.ndarray) -> None:
        """run() with postprocess=True returns postprocessed results."""
        from trtutils.models import YOLOv10

        det = YOLOv10(yolov10_engine, warmup=False)
        results = det.run([horse_image], postprocess=True)
        assert isinstance(results, list)

    @pytest.mark.parametrize("preprocessor", ["cpu", "cuda", "trt"])
    def test_preprocessor_variants(
        self, yolov10_engine: Path, horse_image: np.ndarray, preprocessor: str
    ) -> None:
        """All preprocessors produce valid outputs."""
        from trtutils.models import YOLOv10

        det = YOLOv10(yolov10_engine, preprocessor=preprocessor, warmup=False)
        results = det.run([horse_image], postprocess=False)
        assert isinstance(results, list)
        assert len(results) > 0


@pytest.mark.gpu
class TestDetectorEnd2End:
    """Test Detector end2end pipeline."""

    def test_end2end_single(self, yolov10_engine: Path, horse_image: np.ndarray) -> None:
        """end2end() with single image returns list of detections."""
        from trtutils.models import YOLOv10

        det = YOLOv10(yolov10_engine, warmup=False)
        detections = det.end2end([horse_image])
        assert isinstance(detections, list)
        assert len(detections) == 1

    def test_end2end_returns_detections(self, yolov10_engine: Path, horse_image: np.ndarray) -> None:
        """end2end() returns list[list[tuple]] structure."""
        from trtutils.models import YOLOv10

        det = YOLOv10(yolov10_engine, warmup=False)
        detections = det.end2end([horse_image])
        # detections[0] is a list of (bbox, score, class_id) tuples
        assert isinstance(detections, list)
        for det_list in detections:
            assert isinstance(det_list, list)
            for d in det_list:
                assert len(d) == 3


@pytest.mark.gpu
class TestDetectorBatch:
    """Test Detector batch processing."""

    def test_batch_processing_single(self, yolov10_engine: Path, horse_image: np.ndarray) -> None:
        """Single-image batch inference runs correctly."""
        from trtutils.models import YOLOv10

        det = YOLOv10(yolov10_engine, warmup=False)
        results = det.run([horse_image], postprocess=False)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_batch_end2end(self, yolov10_engine: Path, horse_image: np.ndarray) -> None:
        """end2end returns one detection list per image."""
        from trtutils.models import YOLOv10

        det = YOLOv10(yolov10_engine, warmup=False)
        detections = det.end2end([horse_image])
        assert len(detections) == 1
