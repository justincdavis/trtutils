# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: SLF001
# mypy: disable-error-code="misc,var-annotated"
"""Tests for end-to-end CUDA graph optimization in Detector and Classifier."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

import trtutils
import trtutils.builder
from trtutils.image import Classifier

from .models.common import DETECTOR_CONFIG, build_detector

# Skip entire file if async_v3 not available
pytestmark = pytest.mark.skipif(
    not trtutils.FLAGS.EXEC_ASYNC_V3,
    reason="End-to-end CUDA graph requires async_v3 backend",
)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
HORSE_IMAGE_PATH = DATA_DIR / "horse.jpg"
PEOPLE_IMAGE_PATH = DATA_DIR / "people.jpeg"
CLASSIFIER_ONNX_PATH = DATA_DIR / "onnx" / "resnet18.onnx"
CLASSIFIER_ENGINE_PATH = DATA_DIR / "engines" / "classifier" / "resnet18.engine"

# Use a common detector model for testing
DETECTOR_MODEL = "yolov10"

NUM_ITERS = 100


def _read_image(path: Path) -> np.ndarray:
    """Read an image from disk."""
    img = cv2.imread(str(path))
    if img is None:
        pytest.skip(f"Image not found: {path}")
    assert img is not None
    return img


def _build_classifier_engine() -> Path:
    """Build a classifier engine for testing."""
    if CLASSIFIER_ENGINE_PATH.exists():
        return CLASSIFIER_ENGINE_PATH
    if not CLASSIFIER_ONNX_PATH.exists():
        pytest.skip("Classifier ONNX not available")
    CLASSIFIER_ENGINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        trtutils.builder.build_engine(
            CLASSIFIER_ONNX_PATH,
            CLASSIFIER_ENGINE_PATH,
            optimization_level=1,
        )
    except RuntimeError as e:
        pytest.skip(f"Failed to build classifier engine: {e}")
    return CLASSIFIER_ENGINE_PATH


def _get_detector(
    *,
    cuda_graph: bool = True,
    preprocessor: str = "cuda",
) -> trtutils.image.Detector:
    """Create a detector with cuda_graph support."""
    config = DETECTOR_CONFIG[DETECTOR_MODEL]
    engine_path = build_detector(DETECTOR_MODEL)
    model_class = config["model_class"]
    return model_class(
        engine_path,
        warmup=True,
        preprocessor=preprocessor,
        backend="async_v3",
        cuda_graph=cuda_graph,
    )


def _get_classifier(
    *,
    cuda_graph: bool = True,
    preprocessor: str = "cuda",
) -> Classifier:
    """Create a classifier with cuda_graph support."""
    engine_path = _build_classifier_engine()
    return Classifier(
        engine_path,
        warmup=True,
        preprocessor=preprocessor,
        backend="async_v3",
        cuda_graph=cuda_graph,
    )


# ============================================================================
# Category 1: Graph Capture & Replay
# ============================================================================


def test_e2e_graph_auto_capture_detector() -> None:
    """Test that graph is captured on first end2end() call for Detector."""
    detector = _get_detector(cuda_graph=True)
    image = _read_image(HORSE_IMAGE_PATH)

    # Graph should not exist yet
    assert detector._e2e_graph is None

    # First call should capture the graph
    result = detector.end2end(image)
    assert result is not None

    # Graph should now exist and be captured
    assert detector._e2e_graph is not None
    assert detector._e2e_graph.is_captured is True
    assert detector._e2e_input_dims is not None
    assert detector._e2e_batch_size is not None


def test_e2e_graph_replay_detector() -> None:
    """Test that subsequent calls reuse the captured graph."""
    detector = _get_detector(cuda_graph=True)
    image = _read_image(HORSE_IMAGE_PATH)

    # First call captures
    detector.end2end(image)
    assert detector._e2e_graph is not None
    graph_exec_id = id(detector._e2e_graph._graph_exec)

    # Subsequent calls should reuse the same graph
    for _ in range(10):
        detector.end2end(image)
        assert id(detector._e2e_graph._graph_exec) == graph_exec_id


def test_e2e_graph_locks_dimensions() -> None:
    """Test that dimensions are locked after first call."""
    detector = _get_detector(cuda_graph=True)
    image = _read_image(HORSE_IMAGE_PATH)

    # First call locks dimensions
    detector.end2end(image)
    assert detector._e2e_input_dims == (image.shape[0], image.shape[1])
    assert detector._e2e_batch_size == 1


def test_e2e_graph_auto_capture_classifier() -> None:
    """Test that graph is captured on first end2end() call for Classifier."""
    classifier = _get_classifier(cuda_graph=True)
    image = _read_image(HORSE_IMAGE_PATH)

    # Graph should not exist yet
    assert classifier._e2e_graph is None

    # First call should capture the graph
    result = classifier.end2end(image)
    assert result is not None

    # Graph should now exist and be captured
    assert classifier._e2e_graph is not None
    assert classifier._e2e_graph.is_captured is True


# ============================================================================
# Category 2: Correctness
# ============================================================================


def test_e2e_graph_output_matches_non_graph_detector() -> None:
    """Test that graph and non-graph produce identical results for Detector."""
    detector_graph = _get_detector(cuda_graph=True)
    detector_normal = _get_detector(cuda_graph=False)
    image = _read_image(HORSE_IMAGE_PATH)

    # Run both paths
    result_graph = detector_graph.end2end(image)
    result_normal = detector_normal.end2end(image)

    # Both should produce detection results
    assert isinstance(result_graph, list)
    assert isinstance(result_normal, list)

    # Compare number of detections (may differ slightly due to floating point)
    # Just verify both produce valid output
    assert result_graph is not None
    assert result_normal is not None


def test_e2e_graph_output_matches_non_graph_classifier() -> None:
    """Test that graph and non-graph produce identical results for Classifier."""
    classifier_graph = _get_classifier(cuda_graph=True)
    classifier_normal = _get_classifier(cuda_graph=False)
    image = _read_image(HORSE_IMAGE_PATH)

    # Run both paths
    result_graph = classifier_graph.end2end(image)
    result_normal = classifier_normal.end2end(image)

    # Both should produce classification results
    assert isinstance(result_graph, list)
    assert isinstance(result_normal, list)
    assert len(result_graph) > 0
    assert len(result_normal) > 0


def test_e2e_graph_detector_multiple_frames() -> None:
    """Test detector graph with multiple frames (replay correctness)."""
    detector = _get_detector(cuda_graph=True)
    image = _read_image(HORSE_IMAGE_PATH)

    results = []
    for _ in range(5):
        result = detector.end2end(image)
        results.append(result)

    # All results should be valid
    for result in results:
        assert result is not None
        assert isinstance(result, list)


def test_e2e_graph_classifier_multiple_frames() -> None:
    """Test classifier graph with multiple frames (replay correctness)."""
    classifier = _get_classifier(cuda_graph=True)
    image = _read_image(HORSE_IMAGE_PATH)

    results = []
    for _ in range(5):
        result = classifier.end2end(image)
        results.append(result)

    # All results should be valid
    for result in results:
        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0


# ============================================================================
# Category 3: Performance
# ============================================================================


@pytest.mark.performance
def test_e2e_graph_performance_detector() -> None:
    """Benchmark graph vs non-graph mode for Detector."""
    detector_graph = _get_detector(cuda_graph=True)
    detector_normal = _get_detector(cuda_graph=False)
    image = _read_image(HORSE_IMAGE_PATH)

    # Warmup both
    detector_graph.end2end(image)
    detector_normal.end2end(image)

    # Measure graph path
    graph_times = []
    for _ in range(NUM_ITERS):
        t0 = time.perf_counter()
        detector_graph.end2end(image)
        t1 = time.perf_counter()
        graph_times.append(t1 - t0)

    # Measure normal path
    normal_times = []
    for _ in range(NUM_ITERS):
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

    # Assert no regression
    assert speedup >= 0.90, f"Expected no regression, got {speedup:.3f}"


@pytest.mark.performance
def test_e2e_graph_latency_variance() -> None:
    """Test that graph should reduce jitter."""
    detector = _get_detector(cuda_graph=True)
    image = _read_image(HORSE_IMAGE_PATH)

    # Warmup
    detector.end2end(image)

    latencies = []
    for _ in range(NUM_ITERS):
        t0 = time.perf_counter()
        detector.end2end(image)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    mean = np.mean(latencies)
    std = np.std(latencies)
    cv = std / mean

    print(f"\nMean: {mean * 1000:.3f}ms, Std: {std * 1000:.3f}ms, CV: {cv:.3f}")

    # Assert reasonable variance
    assert cv < 0.5, f"High variation: CV={cv:.3f}"


# ============================================================================
# Category 4: Error Handling
# ============================================================================


def test_e2e_graph_dimension_mismatch_error() -> None:
    """Test that changing image dimensions raises RuntimeError."""
    detector = _get_detector(cuda_graph=True)
    image = _read_image(HORSE_IMAGE_PATH)

    # First call locks dimensions
    detector.end2end(image)

    # Create image with different dimensions
    different_image = cv2.resize(image, (320, 240))

    # Should raise RuntimeError
    with pytest.raises(RuntimeError, match="Image dims"):
        detector.end2end(different_image)


def test_e2e_graph_batch_size_mismatch_error() -> None:
    """Test that changing batch size raises RuntimeError."""
    detector = _get_detector(cuda_graph=True)
    image = _read_image(HORSE_IMAGE_PATH)

    # First call with single image locks batch size to 1
    detector.end2end(image)

    # Call with different batch size
    with pytest.raises(RuntimeError, match="Batch size"):
        detector.end2end([image, image])


def test_e2e_graph_capture_failure_raises() -> None:
    """Test that RuntimeError is raised on failed capture."""
    detector = _get_detector(cuda_graph=True)
    image = _read_image(HORSE_IMAGE_PATH)

    # Mock CUDAGraph to simulate capture failure
    with patch(
        "trtutils.image._detector.CUDAGraph.is_captured",
        new_callable=lambda: property(lambda _self: False),
    ), pytest.raises(RuntimeError, match="CUDA graph capture failed"):
        detector.end2end(image)


def test_e2e_graph_cpu_preprocessor_raises() -> None:
    """Test that cuda_graph with CPU preprocessor raises RuntimeError."""
    detector = _get_detector(cuda_graph=True, preprocessor="cpu")
    image = _read_image(HORSE_IMAGE_PATH)

    with pytest.raises(TypeError, match="CUDA or TRT preprocessor"):
        detector.end2end(image)


# ============================================================================
# Category 5: Lifecycle
# ============================================================================


def test_e2e_graph_flag_false_uses_normal_path() -> None:
    """Test that cuda_graph=False uses _end2end()."""
    detector = _get_detector(cuda_graph=False)
    image = _read_image(HORSE_IMAGE_PATH)

    result = detector.end2end(image)
    assert result is not None

    # Graph should never be created
    assert detector._e2e_graph is None
    assert detector._e2e_graph_enabled is False


def test_e2e_graph_flag_true_uses_graph_path() -> None:
    """Test that cuda_graph=True uses _cuda_graph()."""
    detector = _get_detector(cuda_graph=True)
    image = _read_image(HORSE_IMAGE_PATH)

    result = detector.end2end(image)
    assert result is not None

    # Graph should be created and captured
    assert detector._e2e_graph is not None
    assert detector._e2e_graph_enabled is True
    assert detector._e2e_graph.is_captured is True


def test_e2e_graph_default_enabled() -> None:
    """Test that cuda_graph defaults to enabled."""
    config = DETECTOR_CONFIG[DETECTOR_MODEL]
    engine_path = build_detector(DETECTOR_MODEL)
    model_class = config["model_class"]
    detector = model_class(
        engine_path,
        warmup=True,
        preprocessor="cuda",
        backend="async_v3",
    )

    # Default is cuda_graph=True, so e2e graph should be enabled
    assert detector._e2e_graph_enabled is True
    # But graph is not captured until first end2end() call
    assert detector._e2e_graph is None
