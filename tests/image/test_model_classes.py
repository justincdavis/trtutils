# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Unit tests for model classes batch API signatures and basic processing."""
from __future__ import annotations

import numpy as np

from trtutils.image.postprocessors import (
    get_classifications,
    get_detections,
    postprocess_classifications,
    postprocess_yolov10,
)
from trtutils.image.preprocessors import CPUPreprocessor


def _make_random_images(num_images: int, height: int = 480, width: int = 640) -> list[np.ndarray]:
    """Create random images for testing."""
    return [
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        for _ in range(num_images)
    ]


def _make_yolov10_output(batch_size: int, num_dets: int = 10) -> list[np.ndarray]:
    """Create mock YOLOv10 output: (batch, 300, 6) - [x1, y1, x2, y2, score, class_id]."""
    output = np.zeros((batch_size, 300, 6), dtype=np.float32)
    for b in range(batch_size):
        for i in range(num_dets):
            offset = b * 50
            output[b, i] = [
                100 + i * 10 + offset,
                100 + i * 10 + offset,
                200 + i * 10 + offset,
                200 + i * 10 + offset,
                0.9 - i * 0.05,
                i % 10,
            ]
    return [output]


def _make_classification_output(batch_size: int, num_classes: int = 1000) -> list[np.ndarray]:
    """Create mock classification output: (batch, num_classes) logits."""
    output = np.random.randn(batch_size, num_classes).astype(np.float32)
    for b in range(batch_size):
        output[b, b % num_classes] = 10.0
        output[b, (b + 1) % num_classes] = 8.0
    return [output]


def _make_ratios_padding(
    batch_size: int,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Create mock ratios and padding for batch."""
    ratios = [(1.0, 1.0) for _ in range(batch_size)]
    padding = [(0.0, 0.0) for _ in range(batch_size)]
    return ratios, padding


def test_cpu_preproc_accepts_list_input() -> None:
    """Test that CPUPreprocessor accepts list[np.ndarray] input."""
    preproc = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    images = _make_random_images(3)

    result, ratios, padding = preproc.preprocess(images)

    assert isinstance(result, np.ndarray)
    assert isinstance(ratios, list)
    assert isinstance(padding, list)


def test_cpu_preproc_output_shapes_single() -> None:
    """Test CPUPreprocessor output shapes for single image."""
    preproc = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    images = _make_random_images(1)

    result, ratios, padding = preproc.preprocess(images)

    assert result.shape == (1, 3, 640, 640)
    assert len(ratios) == 1
    assert len(padding) == 1
    assert len(ratios[0]) == 2
    assert len(padding[0]) == 2


def test_cpu_preproc_output_shapes_batch() -> None:
    """Test CPUPreprocessor output shapes for batch."""
    preproc = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    batch_size = 4
    images = _make_random_images(batch_size)

    result, ratios, padding = preproc.preprocess(images)

    assert result.shape == (batch_size, 3, 640, 640)
    assert len(ratios) == batch_size
    assert len(padding) == batch_size


def test_cpu_preproc_ratio_padding_types() -> None:
    """Test that ratios and padding are lists of tuples."""
    preproc = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    images = _make_random_images(2)

    _, ratios, padding = preproc.preprocess(images)

    for ratio in ratios:
        assert isinstance(ratio, tuple)
        assert len(ratio) == 2
        assert all(isinstance(v, float) for v in ratio)

    for pad in padding:
        assert isinstance(pad, tuple)
        assert len(pad) == 2
        assert all(isinstance(v, float) for v in pad)


def test_postprocess_yolov10_returns_list_of_lists() -> None:
    """Test that postprocess_yolov10 returns list[list[np.ndarray]]."""
    batch_size = 3
    outputs = _make_yolov10_output(batch_size)
    ratios, padding = _make_ratios_padding(batch_size)

    results = postprocess_yolov10(outputs, ratios, padding)

    assert isinstance(results, list)
    assert len(results) == batch_size
    for result in results:
        assert isinstance(result, list)
        assert len(result) == 3  # bboxes, scores, class_ids


def test_postprocess_classifications_returns_list_of_lists() -> None:
    """Test that postprocess_classifications returns list[list[np.ndarray]]."""
    batch_size = 3
    outputs = _make_classification_output(batch_size)

    results = postprocess_classifications(outputs)

    assert isinstance(results, list)
    assert len(results) == batch_size
    for result in results:
        assert isinstance(result, list)


def test_get_detections_returns_list_of_lists() -> None:
    """Test that get_detections returns list[list[tuple]]."""
    batch_size = 3
    outputs = _make_yolov10_output(batch_size)
    ratios, padding = _make_ratios_padding(batch_size)

    postprocessed = postprocess_yolov10(outputs, ratios, padding)
    detections = get_detections(postprocessed)

    assert isinstance(detections, list)
    assert len(detections) == batch_size
    for image_dets in detections:
        assert isinstance(image_dets, list)
        for det in image_dets:
            assert isinstance(det, tuple)
            assert len(det) == 3  # bbox, score, class_id


def test_get_classifications_returns_list_of_lists() -> None:
    """Test that get_classifications returns list[list[tuple]]."""
    batch_size = 3
    outputs = _make_classification_output(batch_size)
    postprocessed = postprocess_classifications(outputs)

    classifications = get_classifications(postprocessed, top_k=5)

    assert isinstance(classifications, list)
    assert len(classifications) == batch_size
    for image_cls in classifications:
        assert isinstance(image_cls, list)
        for cls in image_cls:
            assert isinstance(cls, tuple)
            assert len(cls) == 2  # class_id, confidence


def test_preproc_batch_matches_individual() -> None:
    """Test that batch preprocessing matches individual preprocessing."""
    preproc = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))

    # Use same seed for reproducibility
    np.random.seed(42)
    images = _make_random_images(3)

    # Process as batch
    batch_result, batch_ratios, batch_padding = preproc.preprocess(images)

    # Process individually
    for i, img in enumerate(images):
        single_result, single_ratios, single_padding = preproc.preprocess([img])
        np.testing.assert_array_equal(batch_result[i], single_result[0])
        assert batch_ratios[i] == single_ratios[0]
        assert batch_padding[i] == single_padding[0]


def test_detection_output_structure() -> None:
    """Test detection output structure for batch."""
    batch_size = 2
    outputs = _make_yolov10_output(batch_size, num_dets=5)
    ratios, padding = _make_ratios_padding(batch_size)

    postprocessed = postprocess_yolov10(outputs, ratios, padding)

    for i, result in enumerate(postprocessed):
        bboxes, scores, class_ids = result
        assert isinstance(bboxes, np.ndarray)
        assert isinstance(scores, np.ndarray)
        assert isinstance(class_ids, np.ndarray)
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4
        assert len(scores) == len(bboxes)
        assert len(class_ids) == len(bboxes)


def test_classification_output_structure() -> None:
    """Test classification output structure for batch."""
    batch_size = 2
    outputs = _make_classification_output(batch_size)

    postprocessed = postprocess_classifications(outputs)

    for i, result in enumerate(postprocessed):
        assert len(result) >= 1
        probs = result[0]
        assert isinstance(probs, np.ndarray)
        # Probabilities should sum to ~1.0
        assert np.isclose(np.sum(probs), 1.0, rtol=1e-5)


def test_single_image_batch() -> None:
    """Test that single image in list works correctly."""
    preproc = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    images = _make_random_images(1)

    result, ratios, padding = preproc.preprocess(images)

    assert result.shape[0] == 1
    assert len(ratios) == 1
    assert len(padding) == 1


def test_get_detections_empty_detections() -> None:
    """Test get_detections with no valid detections."""
    batch_size = 2
    # Create outputs with very low scores
    outputs = [np.zeros((batch_size, 300, 6), dtype=np.float32)]
    ratios, padding = _make_ratios_padding(batch_size)

    postprocessed = postprocess_yolov10(outputs, ratios, padding, conf_thres=0.5)
    detections = get_detections(postprocessed)

    assert len(detections) == batch_size
    for image_dets in detections:
        assert isinstance(image_dets, list)
        assert len(image_dets) == 0


def test_different_image_sizes_in_batch() -> None:
    """Test preprocessing with different sized images in batch."""
    preproc = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    images = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),
        np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8),
    ]

    result, ratios, padding = preproc.preprocess(images)

    # All should be resized to same output shape
    assert result.shape == (3, 3, 640, 640)
    # Ratios/padding should differ per image
    assert len(set(ratios)) > 1 or len(set(padding)) > 1


def test_preproc_output_dtype() -> None:
    """Test that preprocessor outputs correct dtype."""
    preproc = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    images = _make_random_images(2)

    result, _, _ = preproc.preprocess(images)

    assert result.dtype == np.float32


def test_preproc_output_range() -> None:
    """Test that preprocessor outputs values in correct range."""
    preproc = CPUPreprocessor((640, 640), (0.0, 1.0), np.dtype(np.float32))
    images = _make_random_images(2)

    result, _, _ = preproc.preprocess(images)

    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_detection_bbox_types() -> None:
    """Test that detection bboxes have correct int types in final output."""
    outputs = _make_yolov10_output(1, num_dets=3)
    ratios, padding = _make_ratios_padding(1)

    postprocessed = postprocess_yolov10(outputs, ratios, padding)
    detections = get_detections(postprocessed)

    for det in detections[0]:
        bbox, score, class_id = det
        assert all(isinstance(coord, int) for coord in bbox)
        assert isinstance(score, float)
        assert isinstance(class_id, int)


def test_classification_output_types() -> None:
    """Test that classification outputs have correct types."""
    outputs = _make_classification_output(1)
    postprocessed = postprocess_classifications(outputs)
    classifications = get_classifications(postprocessed, top_k=5)

    for class_id, confidence in classifications[0]:
        assert isinstance(class_id, int)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
