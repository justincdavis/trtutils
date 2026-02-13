# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="misc,no-any-return"
"""Comprehensive inference mode tests for Classifier and Detector."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest

import trtutils.builder
from trtutils.image import Classifier

from .models.common import DETECTOR_CONFIG, build_detector

# Test configuration
PREPROCESSORS = ["cpu", "cuda", "trt"]
BATCH_SIZES = [1, 2]
MEMORY_CONFIGS = [
    {"pagelocked_mem": True},
    {"pagelocked_mem": False},
]

# Use a subset of detector models for comprehensive testing
DETECTOR_MODELS = ["yolov8", "rtdetrv2"]

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
HORSE_IMAGE_PATH = DATA_DIR / "horse.jpg"
PEOPLE_IMAGE_PATH = DATA_DIR / "people.jpeg"
IMAGE_PATHS = [HORSE_IMAGE_PATH, PEOPLE_IMAGE_PATH]

# Classifier paths
CLASSIFIER_ONNX_PATH = DATA_DIR / "onnx" / "resnet18.onnx"
CLASSIFIER_ENGINE_PATH = DATA_DIR / "engines" / "classifier" / "resnet18.engine"


def _read_image(path: Path) -> np.ndarray:
    """
    Read an image from disk.

    Returns
    -------
    np.ndarray
        The loaded image.

    Raises
    ------
    FileNotFoundError
        If the image could not be read.

    """
    img = cv2.imread(str(path))
    if img is None:
        err_msg = f"Failed to read image: {path}"
        raise FileNotFoundError(err_msg)
    return img


def _get_test_images(count: int = 2) -> list[np.ndarray]:
    """
    Get test images, duplicating if needed for larger batch sizes.

    Returns
    -------
    list[np.ndarray]
        The list of test images.

    """
    images = [_read_image(p) for p in IMAGE_PATHS[:count]]
    while len(images) < count:
        images.append(images[0].copy())
    return images[:count]


Detection = tuple[tuple[int, int, int, int], float, int]


def _validate_detection_result(result: list[Detection]) -> None:
    """Validate detection result structure."""
    assert isinstance(result, list)
    for det in result:
        bbox, score, class_id = det
        assert len(bbox) == 4
        assert isinstance(score, float)
        assert isinstance(class_id, int)


def _validate_classification_result(result: list) -> None:
    """Validate classification result structure."""
    assert result is not None
    assert isinstance(result, list)
    for r in result:
        # Each result should have scores and optionally class indices
        assert r is not None


def _assert_detection_parity(
    results1: list[list[Detection]],
    results2: list[list[Detection]],
    atol: float = 5.0,
) -> None:
    """Assert two detection results are approximately equal."""
    assert len(results1) == len(results2), f"Batch size mismatch: {len(results1)} vs {len(results2)}"

    for r1, r2 in zip(results1, results2):
        # Compare number of detections (may vary slightly due to numerical differences)
        assert abs(len(r1) - len(r2)) <= 2, f"Detection count mismatch: {len(r1)} vs {len(r2)}"

        # If same count and non-empty, compare top detections
        if len(r1) == len(r2) and len(r1) > 0:
            # Sort by confidence and compare top detections
            r1_sorted = sorted(r1, key=lambda x: -x[1])
            r2_sorted = sorted(r2, key=lambda x: -x[1])
            for d1, d2 in zip(r1_sorted[:3], r2_sorted[:3]):
                np.testing.assert_allclose(d1[0], d2[0], atol=atol)


# =============================================================================
# Detector Tests - Single Image
# =============================================================================


class TestDetectorSingleImage:
    """Test Detector with single images."""

    @pytest.mark.parametrize("model_id", DETECTOR_MODELS)
    @pytest.mark.parametrize("preprocessor", PREPROCESSORS)
    @pytest.mark.parametrize("mem_config", MEMORY_CONFIGS)
    def test_end2end(self, model_id: str, preprocessor: str, mem_config: dict) -> None:
        """Test Detector.end2end() with single image."""
        engine_path = build_detector(model_id)
        config = DETECTOR_CONFIG[model_id]
        model_class = config["model_class"]

        detector = model_class(
            engine_path,
            preprocessor=preprocessor,
            warmup=False,
            no_warn=True,
            **mem_config,
        )

        images = _get_test_images(1)
        result = detector.end2end(images)

        assert result is not None
        assert len(result) == 1
        _validate_detection_result(result[0])

        del detector

    @pytest.mark.parametrize("model_id", DETECTOR_MODELS)
    @pytest.mark.parametrize("preprocessor", PREPROCESSORS)
    @pytest.mark.parametrize("mem_config", MEMORY_CONFIGS)
    def test_pipeline(self, model_id: str, preprocessor: str, mem_config: dict) -> None:
        """Test preprocess -> run -> postprocess pipeline with single image."""
        engine_path = build_detector(model_id)
        config = DETECTOR_CONFIG[model_id]
        model_class = config["model_class"]

        detector = model_class(
            engine_path,
            preprocessor=preprocessor,
            warmup=False,
            no_warn=True,
            **mem_config,
        )

        images = _get_test_images(1)

        # Step-by-step pipeline
        tensor, ratios, padding = detector.preprocess(images)
        assert tensor is not None
        assert len(ratios) == 1
        assert len(padding) == 1

        outputs = cast(
            "list[np.ndarray]",
            detector.run(
                [tensor],
                ratios=ratios,
                padding=padding,
                preprocessed=True,
                postprocess=False,
            ),
        )
        assert outputs is not None

        postprocessed = detector.postprocess(outputs, ratios, padding)
        assert postprocessed is not None
        assert len(postprocessed) == 1
        detections = detector.get_detections(postprocessed)
        _validate_detection_result(detections[0])

        del detector

    @pytest.mark.parametrize("model_id", DETECTOR_MODELS)
    @pytest.mark.parametrize("preprocessor", PREPROCESSORS)
    def test_run_with_postprocess(self, model_id: str, preprocessor: str) -> None:
        """Test run() with automatic postprocessing."""
        engine_path = build_detector(model_id)
        config = DETECTOR_CONFIG[model_id]
        model_class = config["model_class"]

        detector = model_class(
            engine_path,
            preprocessor=preprocessor,
            warmup=False,
            no_warn=True,
        )

        images = _get_test_images(1)

        # Run with postprocess=True (default)
        postprocessed = detector.run(images)

        assert postprocessed is not None
        assert len(postprocessed) == 1
        detections = detector.get_detections(cast("list[list[np.ndarray]]", postprocessed))
        _validate_detection_result(detections[0])

        del detector


# =============================================================================
# Detector Tests - Batched
# =============================================================================


class TestDetectorBatched:
    """Test Detector with batched images."""

    @pytest.mark.parametrize("model_id", DETECTOR_MODELS)
    @pytest.mark.parametrize("preprocessor", PREPROCESSORS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_end2end_batched(self, model_id: str, preprocessor: str, batch_size: int) -> None:
        """Test Detector.end2end() with batched images."""
        engine_path = build_detector(model_id)
        config = DETECTOR_CONFIG[model_id]
        model_class = config["model_class"]

        detector = model_class(
            engine_path,
            preprocessor=preprocessor,
            warmup=False,
            no_warn=True,
        )

        images = _get_test_images(batch_size)
        result = detector.end2end(images)

        assert result is not None
        assert len(result) == batch_size
        for r in result:
            _validate_detection_result(r)

        del detector

    @pytest.mark.parametrize("model_id", DETECTOR_MODELS)
    @pytest.mark.parametrize("preprocessor", PREPROCESSORS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_pipeline_batched(self, model_id: str, preprocessor: str, batch_size: int) -> None:
        """Test manual pipeline with batched images."""
        engine_path = build_detector(model_id)
        config = DETECTOR_CONFIG[model_id]
        model_class = config["model_class"]

        detector = model_class(
            engine_path,
            preprocessor=preprocessor,
            warmup=False,
            no_warn=True,
        )

        images = _get_test_images(batch_size)

        # Preprocess
        tensor, ratios, padding = detector.preprocess(images)
        assert tensor.shape[0] == batch_size
        assert len(ratios) == batch_size
        assert len(padding) == batch_size

        # Run inference
        outputs = cast(
            "list[np.ndarray]",
            detector.run(
                [tensor],
                ratios=ratios,
                padding=padding,
                preprocessed=True,
                postprocess=False,
            ),
        )
        assert outputs is not None

        # Postprocess
        postprocessed = detector.postprocess(outputs, ratios, padding)
        assert len(postprocessed) == batch_size
        detections = detector.get_detections(postprocessed)
        for r in detections:
            _validate_detection_result(r)

        del detector

    @pytest.mark.parametrize("model_id", DETECTOR_MODELS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_run_batched_with_postprocess(self, model_id: str, batch_size: int) -> None:
        """Test run() with batched images and automatic postprocessing."""
        engine_path = build_detector(model_id)
        config = DETECTOR_CONFIG[model_id]
        model_class = config["model_class"]

        detector = model_class(
            engine_path,
            preprocessor="cpu",
            warmup=False,
            no_warn=True,
        )

        images = _get_test_images(batch_size)
        postprocessed = detector.run(images)

        assert postprocessed is not None
        assert len(postprocessed) == batch_size
        detections = detector.get_detections(cast("list[list[np.ndarray]]", postprocessed))
        for r in detections:
            _validate_detection_result(r)

        del detector


# =============================================================================
# Detector Parity Tests
# =============================================================================


class TestDetectorParity:
    """Test output parity across different inference modes."""

    @pytest.mark.parametrize("model_id", DETECTOR_MODELS)
    def test_preprocessor_parity(self, model_id: str) -> None:
        """Verify all preprocessors produce equivalent detection results."""
        engine_path = build_detector(model_id)
        config = DETECTOR_CONFIG[model_id]
        model_class = config["model_class"]

        images = _get_test_images(1)
        results = {}

        for preproc in PREPROCESSORS:
            detector = model_class(
                engine_path,
                preprocessor=preproc,
                warmup=False,
                no_warn=True,
            )
            results[preproc] = detector.end2end(images)
            del detector

        # Compare CPU vs CUDA vs TRT
        _assert_detection_parity(results["cpu"], results["cuda"])
        _assert_detection_parity(results["cpu"], results["trt"])

    @pytest.mark.parametrize("model_id", DETECTOR_MODELS)
    def test_pipeline_vs_end2end_parity(self, model_id: str) -> None:
        """Verify manual pipeline matches end2end results."""
        engine_path = build_detector(model_id)
        config = DETECTOR_CONFIG[model_id]
        model_class = config["model_class"]

        detector = model_class(
            engine_path,
            preprocessor="cpu",
            warmup=False,
            no_warn=True,
        )

        images = _get_test_images(1)

        # end2end
        result_e2e = detector.end2end(images)

        # manual pipeline
        tensor, ratios, padding = detector.preprocess(images)
        outputs = cast(
            "list[np.ndarray]",
            detector.run(
                [tensor],
                ratios=ratios,
                padding=padding,
                preprocessed=True,
                postprocess=False,
            ),
        )
        postprocessed = detector.postprocess(outputs, ratios, padding)
        result_manual = detector.get_detections(postprocessed)

        _assert_detection_parity(result_e2e, result_manual)

        del detector

    @pytest.mark.parametrize("model_id", DETECTOR_MODELS)
    def test_memory_mode_parity(self, model_id: str) -> None:
        """Verify memory modes don't affect results."""
        engine_path = build_detector(model_id)
        config = DETECTOR_CONFIG[model_id]
        model_class = config["model_class"]

        images = _get_test_images(1)
        results = []

        for pagelocked in [True, False]:
            detector = model_class(
                engine_path,
                preprocessor="cpu",
                warmup=False,
                no_warn=True,
                pagelocked_mem=pagelocked,
            )
            results.append(detector.end2end(images))
            del detector

        _assert_detection_parity(results[0], results[1])


# =============================================================================
# Classifier Tests (conditional on engine availability)
# =============================================================================


def _build_classifier_engine() -> Path | None:
    """
    Build classifier engine if ONNX available, otherwise return None.

    Returns
    -------
    Path | None
        The engine path if available.

    """
    if not CLASSIFIER_ONNX_PATH.exists():
        return None

    if CLASSIFIER_ENGINE_PATH.exists():
        return CLASSIFIER_ENGINE_PATH

    CLASSIFIER_ENGINE_PATH.parent.mkdir(parents=True, exist_ok=True)

    trtutils.builder.build_engine(
        CLASSIFIER_ONNX_PATH,
        CLASSIFIER_ENGINE_PATH,
        optimization_level=1,
    )

    return CLASSIFIER_ENGINE_PATH


@pytest.fixture(scope="module")
def classifier_engine_path() -> Path | None:
    """
    Get classifier engine path, building if needed.

    Returns
    -------
    Path | None
        The engine path if available.

    """
    return _build_classifier_engine()


def _require_classifier_engine(engine_path: Path | None) -> Path:
    if engine_path is None:
        pytest.skip(reason="Classifier ONNX not available")
    assert engine_path is not None
    return engine_path


class TestClassifierSingleImage:
    """Test Classifier with single images."""

    @pytest.mark.parametrize("preprocessor", PREPROCESSORS)
    @pytest.mark.parametrize("mem_config", MEMORY_CONFIGS)
    def test_end2end(
        self, classifier_engine_path: Path | None, preprocessor: str, mem_config: dict
    ) -> None:
        """Test Classifier.end2end() with single image."""
        engine_path = _require_classifier_engine(classifier_engine_path)
        classifier = Classifier(
            engine_path,
            preprocessor=preprocessor,
            warmup=False,
            no_warn=True,
            **mem_config,
        )

        images = _get_test_images(1)
        result = classifier.end2end(images)

        assert result is not None
        _validate_classification_result(result)

        del classifier

    @pytest.mark.parametrize("preprocessor", PREPROCESSORS)
    @pytest.mark.parametrize("mem_config", MEMORY_CONFIGS)
    def test_pipeline(
        self, classifier_engine_path: Path | None, preprocessor: str, mem_config: dict
    ) -> None:
        """Test preprocess -> run -> postprocess pipeline with single image."""
        engine_path = _require_classifier_engine(classifier_engine_path)
        classifier = Classifier(
            engine_path,
            preprocessor=preprocessor,
            warmup=False,
            no_warn=True,
            **mem_config,
        )

        images = _get_test_images(1)

        # Step-by-step pipeline
        tensor, ratios, padding = classifier.preprocess(images)
        assert tensor is not None
        assert len(ratios) == 1
        assert len(padding) == 1

        outputs = cast(
            "list[np.ndarray]",
            classifier.run(
                [tensor],
                preprocessed=True,
                postprocess=False,
            ),
        )
        assert outputs is not None

        result = classifier.postprocess(outputs)
        assert result is not None
        _validate_classification_result(result)

        del classifier


class TestClassifierBatched:
    """Test Classifier with batched images."""

    @pytest.mark.parametrize("preprocessor", PREPROCESSORS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_end2end_batched(
        self, classifier_engine_path: Path | None, preprocessor: str, batch_size: int
    ) -> None:
        """Test Classifier.end2end() with batched images."""
        engine_path = _require_classifier_engine(classifier_engine_path)
        classifier = Classifier(
            engine_path,
            preprocessor=preprocessor,
            warmup=False,
            no_warn=True,
        )

        images = _get_test_images(batch_size)
        result = classifier.end2end(images)

        assert result is not None
        assert len(result) == batch_size
        _validate_classification_result(result)

        del classifier

    @pytest.mark.parametrize("preprocessor", PREPROCESSORS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_pipeline_batched(
        self, classifier_engine_path: Path | None, preprocessor: str, batch_size: int
    ) -> None:
        """Test manual pipeline with batched images."""
        engine_path = _require_classifier_engine(classifier_engine_path)
        classifier = Classifier(
            engine_path,
            preprocessor=preprocessor,
            warmup=False,
            no_warn=True,
        )

        images = _get_test_images(batch_size)

        # Preprocess
        tensor, ratios, padding = classifier.preprocess(images)
        assert tensor.shape[0] == batch_size
        assert len(ratios) == batch_size
        assert len(padding) == batch_size

        # Run inference
        outputs = cast(
            "list[np.ndarray]",
            classifier.run(
                [tensor],
                preprocessed=True,
                postprocess=False,
            ),
        )
        assert outputs is not None

        # Postprocess
        result = classifier.postprocess(outputs)
        assert len(result) == batch_size
        _validate_classification_result(result)

        del classifier


class TestClassifierParity:
    """Test classifier output parity across different inference modes."""

    def test_preprocessor_parity(self, classifier_engine_path: Path | None) -> None:
        """Verify all preprocessors produce equivalent classification results."""
        engine_path = _require_classifier_engine(classifier_engine_path)
        images = _get_test_images(1)
        results = {}

        for preproc in PREPROCESSORS:
            classifier = Classifier(
                engine_path,
                preprocessor=preproc,
                warmup=False,
                no_warn=True,
            )
            results[preproc] = classifier.end2end(images)
            del classifier

        # Compare results - top predicted class should match
        cpu_top = np.argmax(results["cpu"][0][0])
        cuda_top = np.argmax(results["cuda"][0][0])
        trt_top = np.argmax(results["trt"][0][0])

        assert cpu_top == cuda_top, f"CPU top class {cpu_top} != CUDA top class {cuda_top}"
        assert cpu_top == trt_top, f"CPU top class {cpu_top} != TRT top class {trt_top}"

    def test_pipeline_vs_end2end_parity(self, classifier_engine_path: Path | None) -> None:
        """Verify manual pipeline matches end2end results."""
        engine_path = _require_classifier_engine(classifier_engine_path)
        classifier = Classifier(
            engine_path,
            preprocessor="cpu",
            warmup=False,
            no_warn=True,
        )

        images = _get_test_images(1)

        # end2end
        result_e2e = classifier.end2end(images)

        # manual pipeline
        tensor, _ratios, _padding = classifier.preprocess(images)
        outputs = cast(
            "list[np.ndarray]",
            classifier.run(
                [tensor],
                preprocessed=True,
                postprocess=False,
            ),
        )
        result_manual = classifier.postprocess(outputs)

        # Compare outputs
        np.testing.assert_allclose(
            result_e2e[0][0],
            result_manual[0][0],
            rtol=1e-5,
            atol=1e-5,
        )

        del classifier

    def test_memory_mode_parity(self, classifier_engine_path: Path | None) -> None:
        """Verify memory modes don't affect results."""
        engine_path = _require_classifier_engine(classifier_engine_path)
        images = _get_test_images(1)
        results = []

        for pagelocked in [True, False]:
            classifier = Classifier(
                engine_path,
                preprocessor="cpu",
                warmup=False,
                no_warn=True,
                pagelocked_mem=pagelocked,
            )
            results.append(classifier.end2end(images))
            del classifier

        # Results should be identical
        np.testing.assert_allclose(
            results[0][0][0],
            results[1][0][0],
            rtol=1e-5,
            atol=1e-5,
        )
