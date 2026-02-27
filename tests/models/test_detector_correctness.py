# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Detector output correctness tests (GPU required)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from .conftest import (
    DETECTOR_EXPECTED,
    DETECTOR_MODELS,
    _resolve_model_class,
    build_model_engine,
)

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


# ---------------------------------------------------------------------------
# Data-driven multi-model correctness tests
# ---------------------------------------------------------------------------
_DETECTOR_MODEL_IDS = list(DETECTOR_MODELS.keys())
_INFERENCE_MODES = ["end2end", "run"]


def _make_expected_ids() -> list[str]:
    """Create human-readable IDs for DETECTOR_EXPECTED entries."""
    return [Path(e["image"]).stem for e in DETECTOR_EXPECTED]


_EXPECTED_IDS = _make_expected_ids()


def _run_detector_inference(
    detector: object,
    image: np.ndarray,
    mode: str,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    """Run inference with the given mode and return detections."""
    if mode == "end2end":
        return detector.end2end(image)  # type: ignore[union-attr]
    if mode == "run":
        outputs = detector.run(image)  # type: ignore[union-attr]
        return detector.get_detections(outputs)  # type: ignore[union-attr]
    err_msg = f"Unknown inference mode: {mode}"
    raise ValueError(err_msg)


@pytest.mark.gpu
@pytest.mark.correctness
@pytest.mark.download
class TestDetectorImageCorrectness:
    """Data-driven correctness: every detector must find expected objects."""

    @pytest.mark.parametrize("expected_entry", DETECTOR_EXPECTED, ids=_EXPECTED_IDS)
    @pytest.mark.parametrize("model_id", _DETECTOR_MODEL_IDS)
    @pytest.mark.parametrize("mode", _INFERENCE_MODES)
    def test_image_correctness(
        self,
        expected_entry: dict,
        model_id: str,
        mode: str,
    ) -> None:
        """Detector finds expected classes with sufficient detections."""
        import cv2

        cls_name, model_name, imgsz = DETECTOR_MODELS[model_id]
        engine_path = build_model_engine(cls_name, model_name, imgsz)

        model_class = _resolve_model_class(cls_name)
        detector = model_class(
            engine_path,
            preprocessor="cpu",
            warmup=False,
            no_warn=True,
        )

        image_path = str(BASE_DIR / expected_entry["image"])
        image = cv2.imread(image_path)
        if image is None:
            pytest.skip(f"Image not found: {image_path}")

        dets = _run_detector_inference(detector, image, mode)

        # At least min_detections
        assert len(dets) >= expected_entry["min_detections"], (
            f"{model_id}/{mode}: expected >= {expected_entry['min_detections']} "
            f"detections, got {len(dets)}"
        )

        # At least one detection matches an expected class
        detected_classes = [int(d[2]) for d in dets]
        has_match = any(c in expected_entry["expected_classes"] for c in detected_classes)
        assert has_match, (
            f"{model_id}/{mode}: none of {detected_classes} match "
            f"expected {expected_entry['expected_classes']}"
        )

        # All scores above threshold
        for _bbox, score, _cls_id in dets:
            assert float(score) >= expected_entry["conf_thres"] - 1e-6, (
                f"{model_id}/{mode}: score {score} < {expected_entry['conf_thres']}"
            )

        del detector


@pytest.mark.gpu
@pytest.mark.correctness
@pytest.mark.download
class TestDetectorOutputValidityMultiModel:
    """Validate bbox/score/class constraints across all detector models."""

    @pytest.mark.parametrize("expected_entry", DETECTOR_EXPECTED, ids=_EXPECTED_IDS)
    @pytest.mark.parametrize("model_id", _DETECTOR_MODEL_IDS)
    def test_output_validity(
        self,
        expected_entry: dict,
        model_id: str,
    ) -> None:
        """All detections have valid bbox, score, and class_id."""
        import cv2

        cls_name, model_name, imgsz = DETECTOR_MODELS[model_id]
        engine_path = build_model_engine(cls_name, model_name, imgsz)

        model_class = _resolve_model_class(cls_name)
        detector = model_class(
            engine_path,
            preprocessor="cpu",
            warmup=False,
            no_warn=True,
        )

        image_path = str(BASE_DIR / expected_entry["image"])
        image = cv2.imread(image_path)
        if image is None:
            pytest.skip(f"Image not found: {image_path}")

        dets = _run_detector_inference(detector, image, "end2end")

        for bbox, score, class_id in dets:
            x1, y1, x2, y2 = bbox
            assert int(x2) > int(x1), f"{model_id}: invalid bbox width x1={x1}, x2={x2}"
            assert int(y2) > int(y1), f"{model_id}: invalid bbox height y1={y1}, y2={y2}"
            assert 0.0 <= float(score) <= 1.0, f"{model_id}: invalid score {score}"
            assert int(class_id) >= 0, f"{model_id}: invalid class_id {class_id}"

        del detector


@pytest.mark.gpu
@pytest.mark.correctness
@pytest.mark.download
class TestPreprocessorConsistencyMultiModel:
    """Verify cpu/cuda/trt preprocessors give consistent results."""

    @pytest.mark.parametrize("expected_entry", DETECTOR_EXPECTED, ids=_EXPECTED_IDS)
    @pytest.mark.parametrize("model_id", _DETECTOR_MODEL_IDS)
    def test_preprocessor_consistency(
        self,
        expected_entry: dict,
        model_id: str,
    ) -> None:
        """All preprocessors produce at least 1 detection, count variance <= 2."""
        import cv2

        cls_name, model_name, imgsz = DETECTOR_MODELS[model_id]
        engine_path = build_model_engine(cls_name, model_name, imgsz)

        model_class = _resolve_model_class(cls_name)

        image_path = str(BASE_DIR / expected_entry["image"])
        image = cv2.imread(image_path)
        if image is None:
            pytest.skip(f"Image not found: {image_path}")

        preprocessors = ["cpu", "cuda", "trt"]
        counts: dict[str, int] = {}

        for preproc in preprocessors:
            detector = model_class(
                engine_path,
                preprocessor=preproc,
                warmup=False,
                no_warn=True,
            )
            dets = detector.end2end(image)
            counts[preproc] = len(dets)
            del detector

        min_count = min(counts.values())
        max_count = max(counts.values())

        # Every preprocessor should produce at least 1 detection
        for preproc, count in counts.items():
            assert count >= 1, f"{model_id}/{preproc}: expected >= 1 detection, got {count}"

        # Detection count variance across preprocessors should be small
        assert max_count - min_count <= 2, f"{model_id}: detection count variance too high: {counts}"
