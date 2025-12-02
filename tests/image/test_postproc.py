# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import numpy as np

from trtutils.image.postprocessors import (
    get_classifications,
    get_detections,
    postprocess_classifications,
    postprocess_detr,
    postprocess_efficient_nms,
    postprocess_rfdetr,
    postprocess_yolov10,
)


def _make_yolov10_output(batch_size: int, num_dets: int = 10) -> list[np.ndarray]:
    """Create mock YOLOv10 output: (batch, 300, 6) - [x1, y1, x2, y2, score, class_id]."""
    output = np.zeros((batch_size, 300, 6), dtype=np.float32)
    for b in range(batch_size):
        for i in range(num_dets):
            # Create bboxes with different values per batch item
            offset = b * 50
            output[b, i] = [
                100 + i * 10 + offset,  # x1
                100 + i * 10 + offset,  # y1
                200 + i * 10 + offset,  # x2
                200 + i * 10 + offset,  # y2
                0.9 - i * 0.05,  # score (decreasing)
                i % 10,  # class_id
            ]
    return [output]


def _make_efficient_nms_output(batch_size: int, num_dets: int = 10) -> list[np.ndarray]:
    """Create mock EfficientNMS output: [num_dets, bboxes, scores, class_ids]."""
    max_dets = 100
    num_dets_arr = np.full((batch_size,), num_dets, dtype=np.int32)
    bboxes = np.zeros((batch_size, max_dets, 4), dtype=np.float32)
    scores = np.zeros((batch_size, max_dets), dtype=np.float32)
    class_ids = np.zeros((batch_size, max_dets), dtype=np.float32)

    for b in range(batch_size):
        offset = b * 50
        for i in range(num_dets):
            bboxes[b, i] = [
                100 + i * 10 + offset,
                100 + i * 10 + offset,
                200 + i * 10 + offset,
                200 + i * 10 + offset,
            ]
            scores[b, i] = 0.9 - i * 0.05
            class_ids[b, i] = i % 10

    return [num_dets_arr, bboxes, scores, class_ids]


def _make_rfdetr_output(
    batch_size: int, num_queries: int = 300, num_classes: int = 80, num_dets: int = 10
) -> list[np.ndarray]:
    """Create mock RF-DETR output: [dets, labels]."""
    # dets: (batch, num_queries, 4) - normalized cxcywh
    # labels: (batch, num_queries, num_classes) - raw logits
    dets = np.zeros((batch_size, num_queries, 4), dtype=np.float32)
    labels = np.full((batch_size, num_queries, num_classes), -10.0, dtype=np.float32)

    for b in range(batch_size):
        for i in range(num_dets):
            # Normalized center coords (0-1 range for 640x640 input)
            cx = (150 + i * 10 + b * 30) / 640.0
            cy = (150 + i * 10 + b * 30) / 640.0
            w = 100 / 640.0
            h = 100 / 640.0
            dets[b, i] = [cx, cy, w, h]
            # Set high logit for one class per detection
            class_idx = i % num_classes
            labels[b, i, class_idx] = 5.0 - i * 0.3  # High logit (will become high prob)

    return [dets, labels]


def _make_detr_output(
    batch_size: int, num_queries: int = 300, num_dets: int = 10
) -> list[np.ndarray]:
    """Create mock DETR output: [scores, labels, boxes]."""
    # scores: (batch, num_queries)
    # labels: (batch, num_queries)
    # boxes: (batch, num_queries, 4) - xyxy in original coords
    scores = np.zeros((batch_size, num_queries), dtype=np.float32)
    labels = np.zeros((batch_size, num_queries), dtype=np.float32)
    boxes = np.zeros((batch_size, num_queries, 4), dtype=np.float32)

    for b in range(batch_size):
        offset = b * 50
        for i in range(num_dets):
            scores[b, i] = 0.9 - i * 0.05
            labels[b, i] = i % 10
            boxes[b, i] = [
                100 + i * 10 + offset,
                100 + i * 10 + offset,
                200 + i * 10 + offset,
                200 + i * 10 + offset,
            ]

    return [scores, labels, boxes]


def _make_classification_output(
    batch_size: int, num_classes: int = 1000
) -> list[np.ndarray]:
    """Create mock classification output: (batch, num_classes) logits."""
    output = np.random.randn(batch_size, num_classes).astype(np.float32)
    # Make some classes have higher values for predictable results
    for b in range(batch_size):
        output[b, b % num_classes] = 10.0  # High logit for one class per batch
        output[b, (b + 1) % num_classes] = 8.0  # Second highest
    return [output]


def _make_ratios_padding(
    batch_size: int,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Create mock ratios and padding for batch."""
    ratios = [(1.0, 1.0) for _ in range(batch_size)]
    padding = [(0.0, 0.0) for _ in range(batch_size)]
    return ratios, padding


def test_yolov10_single_image() -> None:
    """Test YOLOv10 postprocessing with single image."""
    outputs = _make_yolov10_output(batch_size=1, num_dets=5)
    ratios, padding = _make_ratios_padding(1)

    results = postprocess_yolov10(outputs, ratios, padding)

    assert len(results) == 1
    assert len(results[0]) == 3  # bboxes, scores, class_ids
    assert results[0][0].shape[1] == 4  # bboxes have 4 coords
    assert len(results[0][1]) == len(results[0][0])  # scores match bboxes
    assert len(results[0][2]) == len(results[0][0])  # class_ids match bboxes


def test_yolov10_batch() -> None:
    """Test YOLOv10 postprocessing with batch."""
    batch_size = 4
    outputs = _make_yolov10_output(batch_size=batch_size, num_dets=5)
    ratios, padding = _make_ratios_padding(batch_size)

    results = postprocess_yolov10(outputs, ratios, padding)

    assert len(results) == batch_size
    for result in results:
        assert len(result) == 3  # bboxes, scores, class_ids


def test_yolov10_batch_parity_with_single() -> None:
    """Test that batch results match concatenated single-image results."""
    batch_size = 3
    outputs = _make_yolov10_output(batch_size=batch_size, num_dets=5)
    ratios, padding = _make_ratios_padding(batch_size)

    # Process as batch
    batch_results = postprocess_yolov10(outputs, ratios, padding)

    # Process individually
    for i in range(batch_size):
        single_outputs = [out[i : i + 1] for out in outputs]
        single_results = postprocess_yolov10(single_outputs, [ratios[i]], [padding[i]])
        assert len(single_results) == 1
        # Compare results
        np.testing.assert_array_almost_equal(
            batch_results[i][0], single_results[0][0], decimal=5
        )
        np.testing.assert_array_almost_equal(
            batch_results[i][1], single_results[0][1], decimal=5
        )
        np.testing.assert_array_equal(batch_results[i][2], single_results[0][2])


def test_yolov10_conf_threshold() -> None:
    """Test YOLOv10 confidence threshold filtering."""
    outputs = _make_yolov10_output(batch_size=2, num_dets=10)
    ratios, padding = _make_ratios_padding(2)

    # With high threshold, should filter out many detections
    results_filtered = postprocess_yolov10(outputs, ratios, padding, conf_thres=0.8)
    results_unfiltered = postprocess_yolov10(outputs, ratios, padding, conf_thres=None)

    for i in range(2):
        assert len(results_filtered[i][0]) <= len(results_unfiltered[i][0])


def test_efficient_nms_single_image() -> None:
    """Test EfficientNMS postprocessing with single image."""
    outputs = _make_efficient_nms_output(batch_size=1, num_dets=5)
    ratios, padding = _make_ratios_padding(1)

    results = postprocess_efficient_nms(outputs, ratios, padding)

    assert len(results) == 1
    assert len(results[0]) == 3  # bboxes, scores, class_ids


def test_efficient_nms_batch() -> None:
    """Test EfficientNMS postprocessing with batch."""
    batch_size = 4
    outputs = _make_efficient_nms_output(batch_size=batch_size, num_dets=5)
    ratios, padding = _make_ratios_padding(batch_size)

    results = postprocess_efficient_nms(outputs, ratios, padding)

    assert len(results) == batch_size
    for result in results:
        assert len(result) == 3


def test_efficient_nms_batch_parity_with_single() -> None:
    """Test that batch results match concatenated single-image results."""
    batch_size = 3
    outputs = _make_efficient_nms_output(batch_size=batch_size, num_dets=5)
    ratios, padding = _make_ratios_padding(batch_size)

    # Process as batch
    batch_results = postprocess_efficient_nms(outputs, ratios, padding)

    # Process individually
    for i in range(batch_size):
        single_outputs = [
            outputs[0][i : i + 1],
            outputs[1][i : i + 1],
            outputs[2][i : i + 1],
            outputs[3][i : i + 1],
        ]
        single_results = postprocess_efficient_nms(
            single_outputs, [ratios[i]], [padding[i]]
        )
        assert len(single_results) == 1
        np.testing.assert_array_almost_equal(
            batch_results[i][0], single_results[0][0], decimal=5
        )


def test_rfdetr_single_image() -> None:
    """Test RF-DETR postprocessing with single image."""
    outputs = _make_rfdetr_output(batch_size=1, num_dets=5)
    ratios, padding = _make_ratios_padding(1)

    results = postprocess_rfdetr(outputs, ratios, padding, input_size=(640, 640))

    assert len(results) == 1
    assert len(results[0]) == 3  # bboxes, scores, class_ids


def test_rfdetr_batch() -> None:
    """Test RF-DETR postprocessing with batch."""
    batch_size = 4
    outputs = _make_rfdetr_output(batch_size=batch_size, num_dets=5)
    ratios, padding = _make_ratios_padding(batch_size)

    results = postprocess_rfdetr(outputs, ratios, padding, input_size=(640, 640))

    assert len(results) == batch_size
    for result in results:
        assert len(result) == 3


def test_rfdetr_batch_parity_with_single() -> None:
    """Test that batch results match concatenated single-image results."""
    batch_size = 3
    outputs = _make_rfdetr_output(batch_size=batch_size, num_dets=5)
    ratios, padding = _make_ratios_padding(batch_size)

    # Process as batch
    batch_results = postprocess_rfdetr(outputs, ratios, padding, input_size=(640, 640))

    # Process individually
    for i in range(batch_size):
        single_outputs = [out[i : i + 1] for out in outputs]
        single_results = postprocess_rfdetr(
            single_outputs, [ratios[i]], [padding[i]], input_size=(640, 640)
        )
        assert len(single_results) == 1
        np.testing.assert_array_almost_equal(
            batch_results[i][0], single_results[0][0], decimal=5
        )


def test_detr_single_image() -> None:
    """Test DETR postprocessing with single image."""
    outputs = _make_detr_output(batch_size=1, num_dets=5)
    ratios, padding = _make_ratios_padding(1)

    results = postprocess_detr(outputs, ratios, padding)

    assert len(results) == 1
    assert len(results[0]) == 3  # bboxes, scores, class_ids


def test_detr_batch() -> None:
    """Test DETR postprocessing with batch."""
    batch_size = 4
    outputs = _make_detr_output(batch_size=batch_size, num_dets=5)
    ratios, padding = _make_ratios_padding(batch_size)

    results = postprocess_detr(outputs, ratios, padding)

    assert len(results) == batch_size
    for result in results:
        assert len(result) == 3


def test_detr_batch_parity_with_single() -> None:
    """Test that batch results match concatenated single-image results."""
    batch_size = 3
    outputs = _make_detr_output(batch_size=batch_size, num_dets=5)
    ratios, padding = _make_ratios_padding(batch_size)

    # Process as batch
    batch_results = postprocess_detr(outputs, ratios, padding)

    # Process individually
    for i in range(batch_size):
        single_outputs = [out[i : i + 1] for out in outputs]
        single_results = postprocess_detr(single_outputs, [ratios[i]], [padding[i]])
        assert len(single_results) == 1
        np.testing.assert_array_almost_equal(
            batch_results[i][0], single_results[0][0], decimal=5
        )


def test_detr_conf_threshold() -> None:
    """Test DETR confidence threshold filtering."""
    outputs = _make_detr_output(batch_size=2, num_dets=10)
    ratios, padding = _make_ratios_padding(2)

    results_filtered = postprocess_detr(outputs, ratios, padding, conf_thres=0.8)
    results_unfiltered = postprocess_detr(outputs, ratios, padding, conf_thres=None)

    for i in range(2):
        assert len(results_filtered[i][0]) <= len(results_unfiltered[i][0])


def test_get_detections_single_image() -> None:
    """Test get_detections with single image."""
    outputs = _make_yolov10_output(batch_size=1, num_dets=5)
    ratios, padding = _make_ratios_padding(1)

    postprocessed = postprocess_yolov10(outputs, ratios, padding)
    detections = get_detections(postprocessed)

    assert len(detections) == 1
    assert isinstance(detections[0], list)
    for det in detections[0]:
        assert len(det) == 3  # bbox, score, class_id
        assert len(det[0]) == 4  # x1, y1, x2, y2


def test_get_detections_batch() -> None:
    """Test get_detections with batch."""
    batch_size = 4
    outputs = _make_yolov10_output(batch_size=batch_size, num_dets=5)
    ratios, padding = _make_ratios_padding(batch_size)

    postprocessed = postprocess_yolov10(outputs, ratios, padding)
    detections = get_detections(postprocessed)

    assert len(detections) == batch_size
    for image_dets in detections:
        assert isinstance(image_dets, list)


def test_get_detections_conf_threshold() -> None:
    """Test get_detections confidence threshold filtering."""
    batch_size = 2
    outputs = _make_yolov10_output(batch_size=batch_size, num_dets=10)
    ratios, padding = _make_ratios_padding(batch_size)

    postprocessed = postprocess_yolov10(outputs, ratios, padding)
    dets_filtered = get_detections(postprocessed, conf_thres=0.8)
    dets_unfiltered = get_detections(postprocessed, conf_thres=None)

    for i in range(batch_size):
        assert len(dets_filtered[i]) <= len(dets_unfiltered[i])


def test_classifications_single_image() -> None:
    """Test classification postprocessing with single image."""
    outputs = _make_classification_output(batch_size=1)

    results = postprocess_classifications(outputs)

    assert len(results) == 1
    assert len(results[0]) == 1  # One output tensor
    assert results[0][0].shape == (1, 1000)  # (1, num_classes)
    # Check probabilities sum to 1
    assert np.isclose(np.sum(results[0][0]), 1.0, rtol=1e-5)


def test_classifications_batch() -> None:
    """Test classification postprocessing with batch."""
    batch_size = 4
    outputs = _make_classification_output(batch_size=batch_size)

    results = postprocess_classifications(outputs)

    assert len(results) == batch_size
    for result in results:
        assert len(result) == 1
        assert np.isclose(np.sum(result[0]), 1.0, rtol=1e-5)


def test_classifications_batch_parity_with_single() -> None:
    """Test that batch results match concatenated single-image results."""
    batch_size = 3

    # Use fixed seed for reproducibility
    np.random.seed(42)
    outputs_batch = _make_classification_output(batch_size=batch_size)

    # Process as batch (make a copy since postprocess modifies in-place)
    batch_results = postprocess_classifications([out.copy() for out in outputs_batch])

    # Process individually (use same original outputs)
    for i in range(batch_size):
        single_outputs = [out[i : i + 1].copy() for out in outputs_batch]
        single_results = postprocess_classifications(single_outputs)
        assert len(single_results) == 1
        np.testing.assert_array_almost_equal(
            batch_results[i][0], single_results[0][0], decimal=5
        )


def test_get_classifications_single_image() -> None:
    """Test get_classifications with single image."""
    outputs = _make_classification_output(batch_size=1)
    postprocessed = postprocess_classifications(outputs)

    classifications = get_classifications(postprocessed, top_k=5)

    assert len(classifications) == 1
    assert len(classifications[0]) == 5  # top_k
    for class_id, confidence in classifications[0]:
        assert isinstance(class_id, int)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0


def test_get_classifications_batch() -> None:
    """Test get_classifications with batch."""
    batch_size = 4
    outputs = _make_classification_output(batch_size=batch_size)
    postprocessed = postprocess_classifications(outputs)

    classifications = get_classifications(postprocessed, top_k=5)

    assert len(classifications) == batch_size
    for image_classifications in classifications:
        assert len(image_classifications) == 5


def test_get_classifications_top_k() -> None:
    """Test get_classifications with different top_k values."""
    outputs = _make_classification_output(batch_size=2)
    postprocessed = postprocess_classifications(outputs)

    for top_k in [1, 3, 10]:
        classifications = get_classifications(postprocessed, top_k=top_k)
        for image_classifications in classifications:
            assert len(image_classifications) == top_k


def test_yolov10_empty_detections() -> None:
    """Test YOLOv10 with no valid detections (all zeros)."""
    outputs = [np.zeros((2, 300, 6), dtype=np.float32)]
    ratios, padding = _make_ratios_padding(2)

    results = postprocess_yolov10(outputs, ratios, padding, conf_thres=0.5)

    assert len(results) == 2
    for result in results:
        assert len(result[0]) == 0  # No bboxes pass threshold


def test_efficient_nms_zero_detections() -> None:
    """Test EfficientNMS with zero detections."""
    batch_size = 2
    num_dets_arr = np.zeros((batch_size,), dtype=np.int32)
    bboxes = np.zeros((batch_size, 100, 4), dtype=np.float32)
    scores = np.zeros((batch_size, 100), dtype=np.float32)
    class_ids = np.zeros((batch_size, 100), dtype=np.float32)
    outputs = [num_dets_arr, bboxes, scores, class_ids]
    ratios, padding = _make_ratios_padding(batch_size)

    results = postprocess_efficient_nms(outputs, ratios, padding)

    assert len(results) == batch_size
    for result in results:
        assert len(result[0]) == 0  # No detections


def test_different_ratios_per_image() -> None:
    """Test that different ratios/padding per image are applied correctly."""
    batch_size = 2
    outputs = _make_yolov10_output(batch_size=batch_size, num_dets=3)

    # Different ratios for each image
    ratios = [(1.0, 1.0), (2.0, 2.0)]
    padding = [(0.0, 0.0), (10.0, 10.0)]

    results = postprocess_yolov10(outputs, ratios, padding)

    assert len(results) == batch_size
    # Bboxes should be different due to different ratios/padding
    # Image 1 has ratio 1.0, image 2 has ratio 2.0 (boxes should be smaller)
    # This is a sanity check that per-image params are being used
    if len(results[0][0]) > 0 and len(results[1][0]) > 0:
        # The second image's boxes should be scaled differently
        assert not np.allclose(results[0][0], results[1][0])
