# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="misc"
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from trtutils.image.postprocessors import (
    get_classifications,
    get_detections,
    postprocess_classifications,
    postprocess_yolov10,
)

from trtutils.image.preprocessors import CPUPreprocessor

from .conftest import PREPROC_DTYPE, PREPROC_RANGE, PREPROC_SIZE

if TYPE_CHECKING:
    from collections.abc import Callable


class TestPreprocessorAPI:
    """Validate preprocessor API behavior."""

    def test_accepts_list_input(self, random_images: Callable[..., list[np.ndarray]]) -> None:
        """Preprocessor accepts list input and returns expected types."""
        preproc = CPUPreprocessor(PREPROC_SIZE, PREPROC_RANGE, PREPROC_DTYPE)
        images = random_images(3)
        result, ratios, padding = preproc.preprocess(images)
        assert isinstance(result, np.ndarray)
        assert isinstance(ratios, list)
        assert isinstance(padding, list)

    def test_output_shapes_single(self, random_images: Callable[..., list[np.ndarray]]) -> None:
        """Preprocessor returns correct shapes for single image."""
        preproc = CPUPreprocessor(PREPROC_SIZE, PREPROC_RANGE, PREPROC_DTYPE)
        images = random_images(1)
        result, ratios, padding = preproc.preprocess(images)
        assert result.shape == (1, 3, 640, 640)
        assert len(ratios) == 1
        assert len(padding) == 1
        assert len(ratios[0]) == 2
        assert len(padding[0]) == 2

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_output_shapes_batch(
        self, random_images: Callable[..., list[np.ndarray]], batch_size: int
    ) -> None:
        """Preprocessor returns correct shapes for batch input."""
        preproc = CPUPreprocessor(PREPROC_SIZE, PREPROC_RANGE, PREPROC_DTYPE)
        images = random_images(batch_size)
        result, ratios, padding = preproc.preprocess(images)
        assert result.shape == (batch_size, 3, 640, 640)
        assert len(ratios) == batch_size
        assert len(padding) == batch_size

    def test_ratio_padding_types(self, random_images: Callable[..., list[np.ndarray]]) -> None:
        """Preprocessor returns ratios and padding tuples."""
        preproc = CPUPreprocessor(PREPROC_SIZE, PREPROC_RANGE, PREPROC_DTYPE)
        images = random_images(2)
        _, ratios, padding = preproc.preprocess(images)
        for ratio in ratios:
            assert isinstance(ratio, tuple)
            assert len(ratio) == 2
            assert all(isinstance(v, float) for v in ratio)
        for pad in padding:
            assert isinstance(pad, tuple)
            assert len(pad) == 2
            assert all(isinstance(v, float) for v in pad)

    def test_batch_matches_individual(self, random_images: Callable[..., list[np.ndarray]]) -> None:
        """Batch preprocessing matches individual image preprocessing."""
        preproc = CPUPreprocessor(PREPROC_SIZE, PREPROC_RANGE, PREPROC_DTYPE)
        rng = np.random.default_rng(42)
        images = random_images(3)
        images = [rng.integers(0, 255, img.shape, dtype=np.uint8) for img in images]
        batch_result, batch_ratios, batch_padding = preproc.preprocess(images)
        for i, img in enumerate(images):
            single_result, single_ratios, single_padding = preproc.preprocess([img])
            np.testing.assert_array_equal(batch_result[i], single_result[0])
            assert batch_ratios[i] == single_ratios[0]
            assert batch_padding[i] == single_padding[0]

    def test_output_dtype(self, random_images: Callable[..., list[np.ndarray]]) -> None:
        """Preprocessor outputs float32 arrays."""
        preproc = CPUPreprocessor(PREPROC_SIZE, PREPROC_RANGE, PREPROC_DTYPE)
        images = random_images(2)
        result, _, _ = preproc.preprocess(images)
        assert result.dtype == np.float32

    def test_output_range(self, random_images: Callable[..., list[np.ndarray]]) -> None:
        """Preprocessor outputs normalized values in range."""
        preproc = CPUPreprocessor(PREPROC_SIZE, PREPROC_RANGE, PREPROC_DTYPE)
        images = random_images(2)
        result, _, _ = preproc.preprocess(images)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_different_image_sizes_in_batch(self) -> None:
        """Preprocessor handles mixed image sizes in batch."""
        preproc = CPUPreprocessor(PREPROC_SIZE, PREPROC_RANGE, PREPROC_DTYPE)
        rng = np.random.default_rng(123)
        images = [
            rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
            rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8),
            rng.integers(0, 255, (300, 400, 3), dtype=np.uint8),
        ]
        result, ratios, padding = preproc.preprocess(images)
        assert result.shape == (3, 3, 640, 640)
        assert len(set(ratios)) > 1 or len(set(padding)) > 1


class TestPostprocessorAPI:
    """Validate postprocessor API behavior."""

    def test_yolov10_returns_list_of_lists(
        self,
        make_yolov10_output: Callable[..., list[np.ndarray]],
        make_ratios_padding: Callable[
            ..., tuple[list[tuple[float, float]], list[tuple[float, float]]]
        ],
    ) -> None:
        """YOLOv10 postprocess returns list of detections per image."""
        batch_size = 3
        outputs = make_yolov10_output(batch_size)
        ratios, padding = make_ratios_padding(batch_size)
        results = postprocess_yolov10(outputs, ratios, padding)
        assert isinstance(results, list)
        assert len(results) == batch_size
        for result in results:
            assert isinstance(result, list)
            assert len(result) == 3

    def test_classifications_returns_list_of_lists(
        self, make_classification_output: Callable[..., list[np.ndarray]]
    ) -> None:
        """Classification postprocess returns list of results per image."""
        batch_size = 3
        outputs = make_classification_output(batch_size)
        results = postprocess_classifications(outputs)
        assert isinstance(results, list)
        assert len(results) == batch_size
        for result in results:
            assert isinstance(result, list)


class TestGetDetectionsAPI:
    """Validate get_detections helper behavior."""

    def test_returns_list_of_lists(
        self,
        make_yolov10_output: Callable[..., list[np.ndarray]],
        make_ratios_padding: Callable[
            ..., tuple[list[tuple[float, float]], list[tuple[float, float]]]
        ],
    ) -> None:
        """get_detections returns list of detections per image."""
        batch_size = 3
        outputs = make_yolov10_output(batch_size)
        ratios, padding = make_ratios_padding(batch_size)
        postprocessed = postprocess_yolov10(outputs, ratios, padding)
        detections = get_detections(postprocessed)
        assert isinstance(detections, list)
        assert len(detections) == batch_size
        for image_dets in detections:
            assert isinstance(image_dets, list)
            for det in image_dets:
                assert isinstance(det, tuple)
                assert len(det) == 3

    def test_empty_detections(
        self,
        make_ratios_padding: Callable[
            ..., tuple[list[tuple[float, float]], list[tuple[float, float]]]
        ],
    ) -> None:
        """get_detections handles empty output."""
        batch_size = 2
        outputs = [np.zeros((batch_size, 300, 6), dtype=np.float32)]
        ratios, padding = make_ratios_padding(batch_size)
        postprocessed = postprocess_yolov10(outputs, ratios, padding, conf_thres=0.5)
        detections = get_detections(postprocessed)
        assert len(detections) == batch_size
        for image_dets in detections:
            assert isinstance(image_dets, list)
            assert len(image_dets) == 0

    def test_bbox_types(
        self,
        make_yolov10_output: Callable[..., list[np.ndarray]],
        make_ratios_padding: Callable[
            ..., tuple[list[tuple[float, float]], list[tuple[float, float]]]
        ],
    ) -> None:
        """get_detections returns expected bbox types."""
        outputs = make_yolov10_output(1, num_dets=3)
        ratios, padding = make_ratios_padding(1)
        postprocessed = postprocess_yolov10(outputs, ratios, padding)
        detections = get_detections(postprocessed)
        for det in detections[0]:
            bbox, score, class_id = det
            assert all(isinstance(coord, int) for coord in bbox)
            assert isinstance(score, float)
            assert isinstance(class_id, int)


class TestGetClassificationsAPI:
    """Validate get_classifications helper behavior."""

    def test_returns_list_of_lists(
        self, make_classification_output: Callable[..., list[np.ndarray]]
    ) -> None:
        """get_classifications returns list of top-k outputs per image."""
        batch_size = 3
        outputs = make_classification_output(batch_size)
        postprocessed = postprocess_classifications(outputs)
        classifications = get_classifications(postprocessed, top_k=5)
        assert isinstance(classifications, list)
        assert len(classifications) == batch_size
        for image_cls in classifications:
            assert isinstance(image_cls, list)
            for cls in image_cls:
                assert isinstance(cls, tuple)
                assert len(cls) == 2

    def test_output_types(self, make_classification_output: Callable[..., list[np.ndarray]]) -> None:
        """get_classifications returns class id and confidence types."""
        outputs = make_classification_output(1)
        postprocessed = postprocess_classifications(outputs)
        classifications = get_classifications(postprocessed, top_k=5)
        for class_id, confidence in classifications[0]:
            assert isinstance(class_id, int)
            assert isinstance(confidence, float)
            assert 0.0 <= confidence <= 1.0


class TestOutputStructure:
    """Validate postprocessor output structures."""

    def test_detection_structure(
        self,
        make_yolov10_output: Callable[..., list[np.ndarray]],
        make_ratios_padding: Callable[
            ..., tuple[list[tuple[float, float]], list[tuple[float, float]]]
        ],
    ) -> None:
        """Postprocessed detections contain expected arrays."""
        batch_size = 2
        outputs = make_yolov10_output(batch_size, num_dets=5)
        ratios, padding = make_ratios_padding(batch_size)
        postprocessed = postprocess_yolov10(outputs, ratios, padding)
        for result in postprocessed:
            bboxes, scores, class_ids = result
            assert isinstance(bboxes, np.ndarray)
            assert isinstance(scores, np.ndarray)
            assert isinstance(class_ids, np.ndarray)
            assert bboxes.ndim == 2
            assert bboxes.shape[1] == 4
            assert len(scores) == len(bboxes)
            assert len(class_ids) == len(bboxes)

    def test_classification_structure(
        self, make_classification_output: Callable[..., list[np.ndarray]]
    ) -> None:
        """Postprocessed classifications contain normalized probabilities."""
        batch_size = 2
        outputs = make_classification_output(batch_size)
        postprocessed = postprocess_classifications(outputs)
        for result in postprocessed:
            assert len(result) >= 1
            probs = result[0]
            assert isinstance(probs, np.ndarray)
            assert np.isclose(np.sum(probs), 1.0, rtol=1e-5)
