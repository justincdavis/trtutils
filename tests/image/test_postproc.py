# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="misc"
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from trtutils.image.postprocessors import (
    get_classifications,
    get_detections,
    postprocess_classifications,
    postprocess_detr,
    postprocess_efficient_nms,
    postprocess_rfdetr,
    postprocess_yolov10,
)

RatiosPaddingFactory = Callable[..., tuple[list[tuple[float, float]], list[tuple[float, float]]]]
YoloOutputFactory = Callable[..., list[np.ndarray]]
EfficientNmsOutputFactory = Callable[..., list[np.ndarray]]
RfdetrOutputFactory = Callable[..., list[np.ndarray]]
DetrOutputFactory = Callable[..., list[np.ndarray]]
ClassificationOutputFactory = Callable[..., list[np.ndarray]]


class TestYOLOv10:
    """Test YOLOv10 postprocessing helpers."""

    def test_single_image(
        self, make_yolov10_output: YoloOutputFactory, make_ratios_padding: RatiosPaddingFactory
    ) -> None:
        """Postprocess a single image output."""
        outputs = make_yolov10_output(batch_size=1, num_dets=5)
        ratios, padding = make_ratios_padding(1)
        results = postprocess_yolov10(outputs, ratios, padding)
        assert len(results) == 1
        assert len(results[0]) == 3
        assert results[0][0].shape[1] == 4
        assert len(results[0][1]) == len(results[0][0])
        assert len(results[0][2]) == len(results[0][0])

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_batch(
        self,
        make_yolov10_output: YoloOutputFactory,
        make_ratios_padding: RatiosPaddingFactory,
        batch_size: int,
    ) -> None:
        """Postprocess batch outputs."""
        outputs = make_yolov10_output(batch_size=batch_size, num_dets=5)
        ratios, padding = make_ratios_padding(batch_size)
        results = postprocess_yolov10(outputs, ratios, padding)
        assert len(results) == batch_size
        for result in results:
            assert len(result) == 3

    def test_batch_parity_with_single(
        self, make_yolov10_output: YoloOutputFactory, make_ratios_padding: RatiosPaddingFactory
    ) -> None:
        """Batch postprocess matches per-image postprocess."""
        batch_size = 3
        outputs = make_yolov10_output(batch_size=batch_size, num_dets=5)
        ratios, padding = make_ratios_padding(batch_size)
        batch_results = postprocess_yolov10(outputs, ratios, padding)
        for i in range(batch_size):
            single_outputs = [out[i : i + 1] for out in outputs]
            single_results = postprocess_yolov10(single_outputs, [ratios[i]], [padding[i]])
            assert len(single_results) == 1
            np.testing.assert_array_almost_equal(
                batch_results[i][0], single_results[0][0], decimal=5
            )
            np.testing.assert_array_almost_equal(
                batch_results[i][1], single_results[0][1], decimal=5
            )
            np.testing.assert_array_equal(batch_results[i][2], single_results[0][2])

    def test_conf_threshold(
        self, make_yolov10_output: YoloOutputFactory, make_ratios_padding: RatiosPaddingFactory
    ) -> None:
        """Confidence threshold filters detections."""
        outputs = make_yolov10_output(batch_size=2, num_dets=10)
        ratios, padding = make_ratios_padding(2)
        results_filtered = postprocess_yolov10(outputs, ratios, padding, conf_thres=0.8)
        results_unfiltered = postprocess_yolov10(outputs, ratios, padding, conf_thres=None)
        for i in range(2):
            assert len(results_filtered[i][0]) <= len(results_unfiltered[i][0])

    def test_empty_detections(self, make_ratios_padding: RatiosPaddingFactory) -> None:
        """Empty detections produce empty arrays."""
        outputs = [np.zeros((2, 300, 6), dtype=np.float32)]
        ratios, padding = make_ratios_padding(2)
        results = postprocess_yolov10(outputs, ratios, padding, conf_thres=0.5)
        assert len(results) == 2
        for result in results:
            assert len(result[0]) == 0


class TestEfficientNMS:
    """Test EfficientNMS postprocessing helpers."""

    def test_single_image(
        self,
        make_efficient_nms_output: EfficientNmsOutputFactory,
        make_ratios_padding: RatiosPaddingFactory,
    ) -> None:
        """Postprocess a single image output."""
        outputs = make_efficient_nms_output(batch_size=1, num_dets=5)
        ratios, padding = make_ratios_padding(1)
        results = postprocess_efficient_nms(outputs, ratios, padding)
        assert len(results) == 1
        assert len(results[0]) == 3

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_batch(
        self,
        make_efficient_nms_output: EfficientNmsOutputFactory,
        make_ratios_padding: RatiosPaddingFactory,
        batch_size: int,
    ) -> None:
        """Postprocess batch outputs."""
        outputs = make_efficient_nms_output(batch_size=batch_size, num_dets=5)
        ratios, padding = make_ratios_padding(batch_size)
        results = postprocess_efficient_nms(outputs, ratios, padding)
        assert len(results) == batch_size
        for result in results:
            assert len(result) == 3

    def test_batch_parity_with_single(
        self,
        make_efficient_nms_output: EfficientNmsOutputFactory,
        make_ratios_padding: RatiosPaddingFactory,
    ) -> None:
        """Batch postprocess matches per-image postprocess."""
        batch_size = 3
        outputs = make_efficient_nms_output(batch_size=batch_size, num_dets=5)
        ratios, padding = make_ratios_padding(batch_size)
        batch_results = postprocess_efficient_nms(outputs, ratios, padding)
        for i in range(batch_size):
            single_outputs = [
                outputs[0][i : i + 1],
                outputs[1][i : i + 1],
                outputs[2][i : i + 1],
                outputs[3][i : i + 1],
            ]
            single_results = postprocess_efficient_nms(single_outputs, [ratios[i]], [padding[i]])
            assert len(single_results) == 1
            np.testing.assert_array_almost_equal(
                batch_results[i][0], single_results[0][0], decimal=5
            )

    def test_zero_detections(self, make_ratios_padding: RatiosPaddingFactory) -> None:
        """Zero detections produce empty results."""
        batch_size = 2
        num_dets_arr = np.zeros((batch_size,), dtype=np.int32)
        bboxes = np.zeros((batch_size, 100, 4), dtype=np.float32)
        scores = np.zeros((batch_size, 100), dtype=np.float32)
        class_ids = np.zeros((batch_size, 100), dtype=np.float32)
        outputs = [num_dets_arr, bboxes, scores, class_ids]
        ratios, padding = make_ratios_padding(batch_size)
        results = postprocess_efficient_nms(outputs, ratios, padding)
        assert len(results) == batch_size
        for result in results:
            assert len(result[0]) == 0


class TestRFDETR:
    """Test RF-DETR postprocessing helpers."""

    def test_single_image(
        self, make_rfdetr_output: RfdetrOutputFactory, make_ratios_padding: RatiosPaddingFactory
    ) -> None:
        """Postprocess a single image output."""
        outputs = make_rfdetr_output(batch_size=1, num_dets=5)
        ratios, padding = make_ratios_padding(1)
        results = postprocess_rfdetr(outputs, ratios, padding, input_size=(640, 640))
        assert len(results) == 1
        assert len(results[0]) == 3

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_batch(
        self,
        make_rfdetr_output: RfdetrOutputFactory,
        make_ratios_padding: RatiosPaddingFactory,
        batch_size: int,
    ) -> None:
        """Postprocess batch outputs."""
        outputs = make_rfdetr_output(batch_size=batch_size, num_dets=5)
        ratios, padding = make_ratios_padding(batch_size)
        results = postprocess_rfdetr(outputs, ratios, padding, input_size=(640, 640))
        assert len(results) == batch_size
        for result in results:
            assert len(result) == 3

    def test_batch_parity_with_single(
        self, make_rfdetr_output: RfdetrOutputFactory, make_ratios_padding: RatiosPaddingFactory
    ) -> None:
        """Batch postprocess matches per-image postprocess."""
        batch_size = 3
        outputs = make_rfdetr_output(batch_size=batch_size, num_dets=5)
        ratios, padding = make_ratios_padding(batch_size)
        batch_results = postprocess_rfdetr(outputs, ratios, padding, input_size=(640, 640))
        for i in range(batch_size):
            single_outputs = [out[i : i + 1] for out in outputs]
            single_results = postprocess_rfdetr(
                single_outputs, [ratios[i]], [padding[i]], input_size=(640, 640)
            )
            assert len(single_results) == 1
            np.testing.assert_array_almost_equal(
                batch_results[i][0], single_results[0][0], decimal=5
            )


class TestDETR:
    """Test DETR postprocessing helpers."""

    def test_single_image(
        self, make_detr_output: DetrOutputFactory, make_ratios_padding: RatiosPaddingFactory
    ) -> None:
        """Postprocess a single image output."""
        outputs = make_detr_output(batch_size=1, num_dets=5)
        ratios, padding = make_ratios_padding(1)
        results = postprocess_detr(outputs, ratios, padding)
        assert len(results) == 1
        assert len(results[0]) == 3

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_batch(
        self,
        make_detr_output: DetrOutputFactory,
        make_ratios_padding: RatiosPaddingFactory,
        batch_size: int,
    ) -> None:
        """Postprocess batch outputs."""
        outputs = make_detr_output(batch_size=batch_size, num_dets=5)
        ratios, padding = make_ratios_padding(batch_size)
        results = postprocess_detr(outputs, ratios, padding)
        assert len(results) == batch_size
        for result in results:
            assert len(result) == 3

    def test_batch_parity_with_single(
        self, make_detr_output: DetrOutputFactory, make_ratios_padding: RatiosPaddingFactory
    ) -> None:
        """Batch postprocess matches per-image postprocess."""
        batch_size = 3
        outputs = make_detr_output(batch_size=batch_size, num_dets=5)
        ratios, padding = make_ratios_padding(batch_size)
        batch_results = postprocess_detr(outputs, ratios, padding)
        for i in range(batch_size):
            single_outputs = [out[i : i + 1] for out in outputs]
            single_results = postprocess_detr(single_outputs, [ratios[i]], [padding[i]])
            assert len(single_results) == 1
            np.testing.assert_array_almost_equal(
                batch_results[i][0], single_results[0][0], decimal=5
            )

    def test_conf_threshold(
        self, make_detr_output: DetrOutputFactory, make_ratios_padding: RatiosPaddingFactory
    ) -> None:
        """Confidence threshold filters detections."""
        outputs = make_detr_output(batch_size=2, num_dets=10)
        ratios, padding = make_ratios_padding(2)
        results_filtered = postprocess_detr(outputs, ratios, padding, conf_thres=0.8)
        results_unfiltered = postprocess_detr(outputs, ratios, padding, conf_thres=None)
        for i in range(2):
            assert len(results_filtered[i][0]) <= len(results_unfiltered[i][0])


class TestGetDetections:
    """Test get_detections helper."""

    def test_single_image(
        self, make_yolov10_output: YoloOutputFactory, make_ratios_padding: RatiosPaddingFactory
    ) -> None:
        """Get detections for a single image."""
        outputs = make_yolov10_output(batch_size=1, num_dets=5)
        ratios, padding = make_ratios_padding(1)
        postprocessed = postprocess_yolov10(outputs, ratios, padding)
        detections = get_detections(postprocessed)
        assert len(detections) == 1
        assert isinstance(detections[0], list)
        for det in detections[0]:
            assert len(det) == 3
            assert len(det[0]) == 4

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_batch(
        self,
        make_yolov10_output: YoloOutputFactory,
        make_ratios_padding: RatiosPaddingFactory,
        batch_size: int,
    ) -> None:
        """Get detections for a batch."""
        outputs = make_yolov10_output(batch_size=batch_size, num_dets=5)
        ratios, padding = make_ratios_padding(batch_size)
        postprocessed = postprocess_yolov10(outputs, ratios, padding)
        detections = get_detections(postprocessed)
        assert len(detections) == batch_size
        for image_dets in detections:
            assert isinstance(image_dets, list)

    def test_conf_threshold(
        self, make_yolov10_output: YoloOutputFactory, make_ratios_padding: RatiosPaddingFactory
    ) -> None:
        """Confidence threshold filters detections."""
        batch_size = 2
        outputs = make_yolov10_output(batch_size=batch_size, num_dets=10)
        ratios, padding = make_ratios_padding(batch_size)
        postprocessed = postprocess_yolov10(outputs, ratios, padding)
        dets_filtered = get_detections(postprocessed, conf_thres=0.8)
        dets_unfiltered = get_detections(postprocessed, conf_thres=None)
        for i in range(batch_size):
            assert len(dets_filtered[i]) <= len(dets_unfiltered[i])


class TestClassifications:
    """Test classification postprocessing helpers."""

    def test_single_image(self, make_classification_output: ClassificationOutputFactory) -> None:
        """Postprocess a single image output."""
        outputs = make_classification_output(batch_size=1)
        results = postprocess_classifications(outputs)
        assert len(results) == 1
        assert len(results[0]) == 1
        assert results[0][0].shape == (1, 1000)
        assert np.isclose(np.sum(results[0][0]), 1.0, rtol=1e-5)

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_batch(
        self, make_classification_output: ClassificationOutputFactory, batch_size: int
    ) -> None:
        """Postprocess a batch of outputs."""
        outputs = make_classification_output(batch_size=batch_size)
        results = postprocess_classifications(outputs)
        assert len(results) == batch_size
        for result in results:
            assert len(result) == 1
            assert np.isclose(np.sum(result[0]), 1.0, rtol=1e-5)

    def test_batch_parity_with_single(
        self, make_classification_output: ClassificationOutputFactory
    ) -> None:
        """Batch postprocess matches per-image postprocess."""
        batch_size = 3
        outputs_batch = make_classification_output(batch_size=batch_size)
        batch_results = postprocess_classifications([out.copy() for out in outputs_batch])
        for i in range(batch_size):
            single_outputs = [out[i : i + 1].copy() for out in outputs_batch]
            single_results = postprocess_classifications(single_outputs)
            assert len(single_results) == 1
            np.testing.assert_array_almost_equal(
                batch_results[i][0], single_results[0][0], decimal=5
            )


class TestGetClassifications:
    """Test get_classifications helper."""

    def test_single_image(self, make_classification_output: ClassificationOutputFactory) -> None:
        """Get classifications for a single image."""
        outputs = make_classification_output(batch_size=1)
        postprocessed = postprocess_classifications(outputs)
        classifications = get_classifications(postprocessed, top_k=5)
        assert len(classifications) == 1
        assert len(classifications[0]) == 5
        for class_id, confidence in classifications[0]:
            assert isinstance(class_id, int)
            assert isinstance(confidence, float)
            assert 0.0 <= confidence <= 1.0

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_batch(
        self, make_classification_output: ClassificationOutputFactory, batch_size: int
    ) -> None:
        """Get classifications for a batch."""
        outputs = make_classification_output(batch_size=batch_size)
        postprocessed = postprocess_classifications(outputs)
        classifications = get_classifications(postprocessed, top_k=5)
        assert len(classifications) == batch_size
        for image_classifications in classifications:
            assert len(image_classifications) == 5

    @pytest.mark.parametrize("top_k", [1, 3, 10])
    def test_top_k(
        self, make_classification_output: ClassificationOutputFactory, top_k: int
    ) -> None:
        """Top-k parameter controls number of results."""
        outputs = make_classification_output(batch_size=2)
        postprocessed = postprocess_classifications(outputs)
        classifications = get_classifications(postprocessed, top_k=top_k)
        for image_classifications in classifications:
            assert len(image_classifications) == top_k


class TestDifferentRatiosPerImage:
    """Test varying ratios and padding per image."""

    def test_different_ratios(self, make_yolov10_output: YoloOutputFactory) -> None:
        """Different ratios affect outputs."""
        batch_size = 2
        outputs = make_yolov10_output(batch_size=batch_size, num_dets=3)
        ratios = [(1.0, 1.0), (2.0, 2.0)]
        padding = [(0.0, 0.0), (10.0, 10.0)]
        results = postprocess_yolov10(outputs, ratios, padding)
        assert len(results) == batch_size
        if len(results[0][0]) > 0 and len(results[1][0]) > 0:
            assert not np.allclose(results[0][0], results[1][0])
