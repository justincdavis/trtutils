# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Detector output correctness tests (GPU required)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from trtutils.models import YOLOv10

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
YOLOV10_ONNX = DATA_DIR / "yolov10" / "yolov10n_640.onnx"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def yolov10_engine(build_test_engine) -> Path:
    """Build and cache a YOLOv10n engine for the test module."""
    if not YOLOV10_ONNX.exists():
        pytest.skip("yolov10n_640.onnx not available")
    return build_test_engine(YOLOV10_ONNX)


@pytest.fixture(scope="module")
def yolov10_engine_batch(build_test_engine) -> Path:
    """Build and cache a batch-capable YOLOv10n engine."""
    if not YOLOV10_ONNX.exists():
        pytest.skip("yolov10n_640.onnx not available")
    try:
        return build_test_engine(YOLOV10_ONNX, batch_size=2)
    except Exception as exc:
        pytest.skip(f"Batch YOLOv10 engine unavailable: {exc}")


@pytest.fixture(scope="module")
def yolov10_detector(yolov10_engine) -> YOLOv10:
    """Instantiate a YOLOv10 detector for the module."""
    from trtutils.compat._libs import cudart
    from trtutils.models import YOLOv10

    if not hasattr(cudart, "cudaStreamCreate"):
        pytest.skip("No CUDA runtime available")
    return YOLOv10(yolov10_engine, warmup=False)


@pytest.fixture(scope="module")
def yolov10_detector_batch(yolov10_engine_batch) -> YOLOv10:
    """Instantiate a batch-capable YOLOv10 detector for batched tests."""
    from trtutils.compat._libs import cudart
    from trtutils.models import YOLOv10

    if not hasattr(cudart, "cudaStreamCreate"):
        pytest.skip("No CUDA runtime available")
    return YOLOv10(yolov10_engine_batch, warmup=False)


# ---------------------------------------------------------------------------
# Single-image tests
# ---------------------------------------------------------------------------
class TestDetectorSingleImage:
    """Correctness tests for detector with a single image."""

    @pytest.mark.gpu
    def test_run_returns_list(
        self,
        yolov10_detector,
        horse_image,
    ) -> None:
        """run() with postprocess should return a list of ndarrays."""
        outputs = yolov10_detector.run(horse_image)
        assert isinstance(outputs, list)
        for arr in outputs:
            assert isinstance(arr, np.ndarray)

    @pytest.mark.gpu
    def test_run_raw_returns_list_of_ndarrays(
        self,
        yolov10_detector,
        horse_image,
    ) -> None:
        """run() with postprocess=False returns raw output tensors."""
        outputs = yolov10_detector.run(
            horse_image,
            postprocess=False,
        )
        assert isinstance(outputs, list)
        for arr in outputs:
            assert isinstance(arr, np.ndarray)

    @pytest.mark.gpu
    def test_get_detections_format(
        self,
        yolov10_detector,
        horse_image,
    ) -> None:
        """get_detections output should be list of (bbox, score, cls)."""
        outputs = yolov10_detector.run(horse_image)
        dets = yolov10_detector.get_detections(outputs)
        assert isinstance(dets, list)
        for det in dets:
            assert isinstance(det, tuple)
            assert len(det) == 3
            bbox, score, cls_id = det
            # bbox is (x1, y1, x2, y2) ints
            assert isinstance(bbox, tuple)
            assert len(bbox) == 4
            for coord in bbox:
                assert isinstance(coord, (int, np.integer))
            # score is a float
            assert isinstance(score, (float, np.floating))
            assert 0.0 <= float(score) <= 1.0
            # class id is an int
            assert isinstance(cls_id, (int, np.integer))
            assert int(cls_id) >= 0

    @pytest.mark.gpu
    def test_end2end_format(
        self,
        yolov10_detector,
        horse_image,
    ) -> None:
        """end2end() should return detections in the same format."""
        dets = yolov10_detector.end2end(horse_image)
        assert isinstance(dets, list)
        for det in dets:
            assert isinstance(det, tuple)
            assert len(det) == 3

    @pytest.mark.gpu
    def test_callable_matches_run(
        self,
        yolov10_detector,
        horse_image,
    ) -> None:
        """__call__ should produce the same result as run()."""
        out_run = yolov10_detector.run(horse_image)
        out_call = yolov10_detector(horse_image)
        assert len(out_run) == len(out_call)


# ---------------------------------------------------------------------------
# Batch tests
# ---------------------------------------------------------------------------
class TestDetectorBatch:
    """Correctness tests for detector with batched images."""

    @pytest.mark.gpu
    def test_batch_run_returns_nested_lists(
        self,
        yolov10_detector_batch,
        random_images,
    ) -> None:
        """Batch run with postprocess returns list[list[ndarray]]."""
        imgs = random_images(count=2, height=480, width=640)
        outputs = yolov10_detector_batch.run(imgs)
        assert isinstance(outputs, list)
        assert len(outputs) == 2
        for per_image in outputs:
            assert isinstance(per_image, list)
            for arr in per_image:
                assert isinstance(arr, np.ndarray)

    @pytest.mark.gpu
    def test_batch_get_detections(
        self,
        yolov10_detector_batch,
        random_images,
    ) -> None:
        """Batch get_detections returns list[list[tuple]]."""
        imgs = random_images(count=2, height=480, width=640)
        outputs = yolov10_detector_batch.run(imgs)
        dets = yolov10_detector_batch.get_detections(outputs)
        assert isinstance(dets, list)
        assert len(dets) == 2
        for per_image_dets in dets:
            assert isinstance(per_image_dets, list)


# ---------------------------------------------------------------------------
# Confidence threshold tests
# ---------------------------------------------------------------------------
class TestDetectorConfThreshold:
    """Test that confidence threshold filtering works."""

    @pytest.mark.gpu
    def test_high_threshold_fewer_detections(
        self,
        yolov10_detector,
        horse_image,
    ) -> None:
        """Higher conf threshold should yield fewer or equal detections."""
        outputs = yolov10_detector.run(horse_image)
        dets_low = yolov10_detector.get_detections(
            outputs,
            conf_thres=0.01,
        )
        dets_high = yolov10_detector.get_detections(
            outputs,
            conf_thres=0.9,
        )
        assert len(dets_high) <= len(dets_low)

    @pytest.mark.gpu
    def test_all_scores_above_threshold(
        self,
        yolov10_detector,
        horse_image,
    ) -> None:
        """All returned detections should have score >= conf_thres."""
        threshold = 0.5
        outputs = yolov10_detector.run(horse_image)
        dets = yolov10_detector.get_detections(
            outputs,
            conf_thres=threshold,
        )
        for _bbox, score, _cls_id in dets:
            assert float(score) >= threshold - 1e-6


# ---------------------------------------------------------------------------
# Bbox coordinate validity
# ---------------------------------------------------------------------------
class TestDetectorBboxValidity:
    """Test that bounding box coordinates are valid."""

    @pytest.mark.gpu
    def test_bbox_x2_ge_x1(
        self,
        yolov10_detector,
        horse_image,
    ) -> None:
        """x2 should be >= x1 for all detections."""
        outputs = yolov10_detector.run(horse_image)
        dets = yolov10_detector.get_detections(outputs)
        for (x1, y1, x2, y2), _score, _cls_id in dets:
            assert int(x2) >= int(x1)
            assert int(y2) >= int(y1)
