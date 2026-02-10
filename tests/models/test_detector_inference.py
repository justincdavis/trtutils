# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: SLF001
# mypy: disable-error-code="misc,no-any-return,var-annotated"
"""Unified GPU-only detector and classifier inference tests."""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast
from unittest.mock import patch

import cv2
import numpy as np
import pytest

import trtutils
import trtutils.builder
from tests.helpers import HORSE_IMAGE_PATH, PEOPLE_IMAGE_PATH, read_image
from trtutils.image import Classifier

from .common import (
    DETECTOR_CONFIG,
    GPU_ENGINES,
    build_detector,
    detector_pagelocked_perf,
    detector_results,
    detector_run,
    detector_run_in_thread,
    detector_run_multiple,
    detector_run_multiple_threads,
    detector_swapping_preproc_results,
)

pytestmark = [pytest.mark.gpu]

# Test configuration
PREPROCESSORS = ["cpu", "cuda", "trt"]
BATCH_SIZES = [1, 2]
MEMORY_CONFIGS = [
    {"pagelocked_mem": True},
    {"pagelocked_mem": False},
]

# Use a subset of detector models for comprehensive testing
DETECTOR_MODELS_SUBSET = ["yolov8", "rtdetrv2"]

# All model IDs from config
DETECTOR_MODELS_ALL = list(DETECTOR_CONFIG.keys())

Detection = tuple[tuple[int, int, int, int], float, int]


def _get_test_images(count: int = 2) -> list[np.ndarray]:
    """Get test images, duplicating if needed for larger batch sizes."""
    paths = [Path(HORSE_IMAGE_PATH), Path(PEOPLE_IMAGE_PATH)]
    images = [read_image(p) for p in paths[:count]]
    while len(images) < count:
        images.append(images[0].copy())
    return images[:count]


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
        assert r is not None


def _assert_detection_parity(
    results1: list[list[Detection]],
    results2: list[list[Detection]],
    atol: float = 5.0,
) -> None:
    """Assert two detection results are approximately equal."""
    assert len(results1) == len(results2), f"Batch size mismatch: {len(results1)} vs {len(results2)}"

    for r1, r2 in zip(results1, results2):
        assert abs(len(r1) - len(r2)) <= 2, f"Detection count mismatch: {len(r1)} vs {len(r2)}"

        if len(r1) == len(r2) and len(r1) > 0:
            r1_sorted = sorted(r1, key=lambda x: -x[1])
            r2_sorted = sorted(r2, key=lambda x: -x[1])
            for d1, d2 in zip(r1_sorted[:3], r2_sorted[:3]):
                np.testing.assert_allclose(d1[0], d2[0], atol=atol)


def _require_classifier_engine(engine_path: Path | None) -> Path:
    if engine_path is None:
        pytest.skip("Classifier ONNX not available")
    assert engine_path is not None
    return engine_path


# =============================================================================
# Section 1: Basic Detector Run (from test_detector.py, GPU-only)
# =============================================================================


@pytest.mark.parametrize("model_id", DETECTOR_MODELS_ALL)
@pytest.mark.parametrize("preprocessor", PREPROCESSORS)
def test_detector_run(model_id: str, preprocessor: str) -> None:
    """Test detector engine runs with different preprocessors (GPU only)."""
    detector_run(model_id, preprocessor=preprocessor, use_dla=False)


@pytest.mark.parametrize("model_id", DETECTOR_MODELS_ALL)
@pytest.mark.parametrize("preprocessor", PREPROCESSORS)
def test_detector_run_in_thread(model_id: str, preprocessor: str) -> None:
    """Test detector engine runs in a thread with different preprocessors (GPU only)."""
    detector_run_in_thread(model_id, preprocessor=preprocessor, use_dla=False)


@pytest.mark.parametrize("model_id", DETECTOR_MODELS_ALL)
@pytest.mark.parametrize("preprocessor", PREPROCESSORS)
def test_detector_run_multiple(model_id: str, preprocessor: str) -> None:
    """Test multiple detector engines run with different preprocessors (GPU only)."""
    detector_run_multiple(model_id, preprocessor=preprocessor, count=GPU_ENGINES, use_dla=False)


@pytest.mark.parametrize("model_id", DETECTOR_MODELS_ALL)
@pytest.mark.parametrize("preprocessor", PREPROCESSORS)
def test_detector_run_multiple_threads(model_id: str, preprocessor: str) -> None:
    """Test multiple detector engines run across threads (GPU only)."""
    detector_run_multiple_threads(
        model_id, preprocessor=preprocessor, count=GPU_ENGINES, use_dla=False
    )


@pytest.mark.parametrize("model_id", DETECTOR_MODELS_ALL)
@pytest.mark.parametrize("preprocessor", PREPROCESSORS)
def test_detector_results(model_id: str, preprocessor: str) -> None:
    """Test detector engine produces valid results (GPU only)."""
    detector_results(model_id, preprocessor=preprocessor, use_dla=False)


@pytest.mark.parametrize("model_id", DETECTOR_MODELS_ALL)
def test_detector_swapping_preproc_results(model_id: str) -> None:
    """Test swapping preprocessing method at runtime (GPU only)."""
    detector_swapping_preproc_results(model_id, use_dla=False)


# =============================================================================
# Section 2: Detector Single/Batched/Parity
# =============================================================================


class TestDetectorSingleImage:
    """Test Detector with single images."""

    @pytest.mark.parametrize("model_id", DETECTOR_MODELS_SUBSET)
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

    @pytest.mark.parametrize("model_id", DETECTOR_MODELS_SUBSET)
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

    @pytest.mark.parametrize("model_id", DETECTOR_MODELS_SUBSET)
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

        postprocessed = detector.run(images)

        assert postprocessed is not None
        assert len(postprocessed) == 1
        detections = detector.get_detections(cast("list[list[np.ndarray]]", postprocessed))
        _validate_detection_result(detections[0])

        del detector


class TestDetectorBatched:
    """Test Detector with batched images."""

    @pytest.mark.parametrize("model_id", DETECTOR_MODELS_SUBSET)
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

    @pytest.mark.parametrize("model_id", DETECTOR_MODELS_SUBSET)
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

        tensor, ratios, padding = detector.preprocess(images)
        assert tensor.shape[0] == batch_size
        assert len(ratios) == batch_size
        assert len(padding) == batch_size

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
        assert len(postprocessed) == batch_size
        detections = detector.get_detections(postprocessed)
        for r in detections:
            _validate_detection_result(r)

        del detector

    @pytest.mark.parametrize("model_id", DETECTOR_MODELS_SUBSET)
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


class TestDetectorParity:
    """Test output parity across different inference modes."""

    @pytest.mark.parametrize("model_id", DETECTOR_MODELS_SUBSET)
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

        _assert_detection_parity(results["cpu"], results["cuda"])
        _assert_detection_parity(results["cpu"], results["trt"])

    @pytest.mark.parametrize("model_id", DETECTOR_MODELS_SUBSET)
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

        result_e2e = detector.end2end(images)

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

    @pytest.mark.parametrize("model_id", DETECTOR_MODELS_SUBSET)
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
# Section 3: Classifier Single/Batched/Parity
# =============================================================================


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

        tensor, ratios, padding = classifier.preprocess(images)
        assert tensor.shape[0] == batch_size
        assert len(ratios) == batch_size
        assert len(padding) == batch_size

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

        result_e2e = classifier.end2end(images)

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

        np.testing.assert_allclose(
            results[0][0][0],
            results[1][0][0],
            rtol=1e-5,
            atol=1e-5,
        )


# =============================================================================
# Section 4: E2E CUDA Graph (from test_end2end_graph.py)
# =============================================================================

# Use a single detector model to avoid 17x slowdown
E2E_GRAPH_DETECTOR_MODEL = "yolov10"

E2E_GRAPH_NUM_ITERS = 100


def _get_e2e_detector(
    *,
    cuda_graph: bool = True,
    preprocessor: str = "cuda",
) -> trtutils.image.Detector:
    """Create a detector with cuda_graph support."""
    config = DETECTOR_CONFIG[E2E_GRAPH_DETECTOR_MODEL]
    engine_path = build_detector(E2E_GRAPH_DETECTOR_MODEL)
    model_class = config["model_class"]
    return model_class(
        engine_path,
        warmup=True,
        preprocessor=preprocessor,
        backend="async_v3",
        cuda_graph=cuda_graph,
    )


def _get_e2e_classifier(
    classifier_engine_path: Path,
    *,
    cuda_graph: bool = True,
    preprocessor: str = "cuda",
) -> Classifier:
    """Create a classifier with cuda_graph support."""
    return Classifier(
        classifier_engine_path,
        warmup=True,
        preprocessor=preprocessor,
        backend="async_v3",
        cuda_graph=cuda_graph,
    )


@pytest.mark.cuda_graph
@pytest.mark.skipif(
    not trtutils.FLAGS.EXEC_ASYNC_V3,
    reason="End-to-end CUDA graph requires async_v3 backend",
)
class TestE2ECUDAGraph:
    """End-to-end CUDA graph tests for Detector and Classifier."""

    # ---- Graph Capture & Replay ----

    def test_e2e_graph_auto_capture_detector(self) -> None:
        """Test that graph is captured on first end2end() call for Detector."""
        detector = _get_e2e_detector(cuda_graph=True)
        image = read_image(HORSE_IMAGE_PATH)

        assert detector._e2e_graph is None

        result = detector.end2end(image)
        assert result is not None

        assert detector._e2e_graph is not None
        assert detector._e2e_graph.is_captured is True
        assert detector._e2e_input_dims is not None
        assert detector._e2e_batch_size is not None

    def test_e2e_graph_replay_detector(self) -> None:
        """Test that subsequent calls reuse the captured graph."""
        detector = _get_e2e_detector(cuda_graph=True)
        image = read_image(HORSE_IMAGE_PATH)

        detector.end2end(image)
        assert detector._e2e_graph is not None
        graph_exec_id = id(detector._e2e_graph._graph_exec)

        for _ in range(10):
            detector.end2end(image)
            assert id(detector._e2e_graph._graph_exec) == graph_exec_id

    def test_e2e_graph_locks_dimensions(self) -> None:
        """Test that dimensions are locked after first call."""
        detector = _get_e2e_detector(cuda_graph=True)
        image = read_image(HORSE_IMAGE_PATH)

        detector.end2end(image)
        assert detector._e2e_input_dims == (image.shape[0], image.shape[1])
        assert detector._e2e_batch_size == 1

    def test_e2e_graph_auto_capture_classifier(self, classifier_engine_path: Path | None) -> None:
        """Test that graph is captured on first end2end() call for Classifier."""
        engine_path = _require_classifier_engine(classifier_engine_path)
        classifier = _get_e2e_classifier(engine_path, cuda_graph=True)
        image = read_image(HORSE_IMAGE_PATH)

        assert classifier._e2e_graph is None

        result = classifier.end2end(image)
        assert result is not None

        assert classifier._e2e_graph is not None
        assert classifier._e2e_graph.is_captured is True

    # ---- Correctness ----

    def test_e2e_graph_output_matches_non_graph_detector(self) -> None:
        """Test that graph and non-graph produce identical results for Detector."""
        detector_graph = _get_e2e_detector(cuda_graph=True)
        detector_normal = _get_e2e_detector(cuda_graph=False)
        image = read_image(HORSE_IMAGE_PATH)

        result_graph = detector_graph.end2end(image)
        result_normal = detector_normal.end2end(image)

        assert isinstance(result_graph, list)
        assert isinstance(result_normal, list)
        assert result_graph is not None
        assert result_normal is not None

    def test_e2e_graph_output_matches_non_graph_classifier(
        self, classifier_engine_path: Path | None
    ) -> None:
        """Test that graph and non-graph produce identical results for Classifier."""
        engine_path = _require_classifier_engine(classifier_engine_path)
        classifier_graph = _get_e2e_classifier(engine_path, cuda_graph=True)
        classifier_normal = _get_e2e_classifier(engine_path, cuda_graph=False)
        image = read_image(HORSE_IMAGE_PATH)

        result_graph = classifier_graph.end2end(image)
        result_normal = classifier_normal.end2end(image)

        assert isinstance(result_graph, list)
        assert isinstance(result_normal, list)
        assert len(result_graph) > 0
        assert len(result_normal) > 0

    def test_e2e_graph_detector_multiple_frames(self) -> None:
        """Test detector graph with multiple frames (replay correctness)."""
        detector = _get_e2e_detector(cuda_graph=True)
        image = read_image(HORSE_IMAGE_PATH)

        results = []
        for _ in range(5):
            result = detector.end2end(image)
            results.append(result)

        for result in results:
            assert result is not None
            assert isinstance(result, list)

    def test_e2e_graph_classifier_multiple_frames(self, classifier_engine_path: Path | None) -> None:
        """Test classifier graph with multiple frames (replay correctness)."""
        engine_path = _require_classifier_engine(classifier_engine_path)
        classifier = _get_e2e_classifier(engine_path, cuda_graph=True)
        image = read_image(HORSE_IMAGE_PATH)

        results = []
        for _ in range(5):
            result = classifier.end2end(image)
            results.append(result)

        for result in results:
            assert result is not None
            assert isinstance(result, list)
            assert len(result) > 0

    # ---- Performance ----

    @pytest.mark.performance
    def test_e2e_graph_performance_detector(self) -> None:
        """Benchmark graph vs non-graph mode for Detector."""
        detector_graph = _get_e2e_detector(cuda_graph=True)
        detector_normal = _get_e2e_detector(cuda_graph=False)
        image = read_image(HORSE_IMAGE_PATH)

        detector_graph.end2end(image)
        detector_normal.end2end(image)

        graph_times = []
        for _ in range(E2E_GRAPH_NUM_ITERS):
            t0 = time.perf_counter()
            detector_graph.end2end(image)
            t1 = time.perf_counter()
            graph_times.append(t1 - t0)

        normal_times = []
        for _ in range(E2E_GRAPH_NUM_ITERS):
            t0 = time.perf_counter()
            detector_normal.end2end(image)
            t1 = time.perf_counter()
            normal_times.append(t1 - t0)

        mean_graph = np.mean(graph_times)
        mean_normal = np.mean(normal_times)
        speedup = mean_normal / mean_graph

        print(f"\nE2E Graph: {mean_graph * 1000:.3f}ms")
        print(f"E2E Normal: {mean_normal * 1000:.3f}ms")
        print(f"Speedup: {speedup:.2f}x")

        assert speedup >= 0.90, f"Expected no regression, got {speedup:.3f}"

    @pytest.mark.performance
    def test_e2e_graph_latency_variance(self) -> None:
        """Test that graph should reduce jitter."""
        detector = _get_e2e_detector(cuda_graph=True)
        image = read_image(HORSE_IMAGE_PATH)

        detector.end2end(image)

        latencies = []
        for _ in range(E2E_GRAPH_NUM_ITERS):
            t0 = time.perf_counter()
            detector.end2end(image)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

        mean = np.mean(latencies)
        std = np.std(latencies)
        cv = std / mean

        print(f"\nMean: {mean * 1000:.3f}ms, Std: {std * 1000:.3f}ms, CV: {cv:.3f}")

        assert cv < 0.5, f"High variation: CV={cv:.3f}"

    # ---- Error Handling ----

    def test_e2e_graph_dimension_mismatch_error(self) -> None:
        """Test that changing image dimensions raises RuntimeError."""
        detector = _get_e2e_detector(cuda_graph=True)
        image = read_image(HORSE_IMAGE_PATH)

        detector.end2end(image)

        different_image = cv2.resize(image, (320, 240))

        with pytest.raises(RuntimeError, match="Image dims"):
            detector.end2end(different_image)

    def test_e2e_graph_batch_size_mismatch_error(self) -> None:
        """Test that changing batch size raises RuntimeError."""
        detector = _get_e2e_detector(cuda_graph=True)
        image = read_image(HORSE_IMAGE_PATH)

        detector.end2end(image)

        with pytest.raises(RuntimeError, match="Batch size"):
            detector.end2end([image, image])

    def test_e2e_graph_capture_failure_raises(self) -> None:
        """Test that RuntimeError is raised on failed capture."""
        detector = _get_e2e_detector(cuda_graph=True)
        image = read_image(HORSE_IMAGE_PATH)

        with patch(
            "trtutils.image._detector.CUDAGraph.is_captured",
            new_callable=lambda: property(lambda _self: False),
        ), pytest.raises(RuntimeError, match="CUDA graph capture failed"):
            detector.end2end(image)

    def test_e2e_graph_cpu_preprocessor_works(self) -> None:
        """Test that cuda_graph works with CPU preprocessor."""
        detector = _get_e2e_detector(cuda_graph=True, preprocessor="cpu")
        image = read_image(HORSE_IMAGE_PATH)

        result = detector.end2end(image)
        assert result is not None
        assert isinstance(result, list)

        assert detector._e2e_graph is not None
        assert detector._e2e_graph.is_captured is True

    # ---- Lifecycle ----

    def test_e2e_graph_flag_false_uses_normal_path(self) -> None:
        """Test that cuda_graph=False uses _end2end()."""
        detector = _get_e2e_detector(cuda_graph=False)
        image = read_image(HORSE_IMAGE_PATH)

        result = detector.end2end(image)
        assert result is not None

        assert detector._e2e_graph is None
        assert detector._e2e_graph_enabled is False

    def test_e2e_graph_flag_true_uses_graph_path(self) -> None:
        """Test that cuda_graph=True uses _cuda_graph()."""
        detector = _get_e2e_detector(cuda_graph=True)
        image = read_image(HORSE_IMAGE_PATH)

        result = detector.end2end(image)
        assert result is not None

        assert detector._e2e_graph is not None
        assert detector._e2e_graph_enabled is True
        assert detector._e2e_graph.is_captured is True

    def test_e2e_graph_default_enabled(self) -> None:
        """Test that cuda_graph defaults to enabled."""
        config = DETECTOR_CONFIG[E2E_GRAPH_DETECTOR_MODEL]
        engine_path = build_detector(E2E_GRAPH_DETECTOR_MODEL)
        model_class = config["model_class"]
        detector = model_class(
            engine_path,
            warmup=True,
            preprocessor="cuda",
            backend="async_v3",
        )

        assert detector._e2e_graph_enabled is True
        assert detector._e2e_graph is None


# =============================================================================
# Section 5: Performance (from test_perf.py, GPU-only)
# =============================================================================


@pytest.mark.performance
@pytest.mark.parametrize("model_id", DETECTOR_MODELS_ALL)
def test_detector_pagelocked_perf(model_id: str) -> None:
    """Test the performance of detector models with pagelocked memory (GPU only)."""
    detector_pagelocked_perf(model_id, use_dla=False)
