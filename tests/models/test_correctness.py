# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: SLF001
# mypy: disable-error-code="misc,no-any-return"
"""Detection correctness and regression tests."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

import trtutils
from tests.ground_truth import HORSE_CLASS_ID, PERSON_CLASS_ID
from tests.helpers import HORSE_IMAGE_PATH, PEOPLE_IMAGE_PATH, read_image
from trtutils.image._schema import InputSchema
from trtutils.image.preprocessors import CUDAPreprocessor

from .common import DETECTOR_CONFIG, build_detector

pytestmark = [pytest.mark.gpu, pytest.mark.correctness]

# All detector models to test
ALL_MODELS = list(DETECTOR_CONFIG.keys())

# Inference modes to test
INFERENCE_MODES = [
    "end2end",  # end2end()
    "run",  # run() with postprocess=True
    "run_raw",  # run() with postprocess=False + get_detections()
]

# Preprocessor types
PREPROCESSORS = ["cpu", "cuda", "trt"]

# RTDETRv3 model ID
RTDETRV3_MODEL_ID = "rtdetrv3"

Detection = tuple[tuple[int, int, int, int], float, int]


def _run_inference(
    detector: object,
    images: list[np.ndarray],
    mode: str,
) -> list[list[Detection]]:
    """Run inference using the specified mode."""
    if mode == "end2end":
        result = detector.end2end(images)  # type: ignore[union-attr]
        if isinstance(result, list) and result and not isinstance(result[0], list):
            return [result]  # type: ignore[list-item]
        return result  # type: ignore[return-value]

    if mode == "run":
        result = detector.run(images)  # type: ignore[union-attr]
        dets = detector.get_detections(result)  # type: ignore[union-attr]
        if isinstance(dets, list) and dets and not isinstance(dets[0], list):
            return [dets]  # type: ignore[list-item]
        return dets  # type: ignore[return-value]

    if mode == "run_raw":
        raw = cast(
            "list[np.ndarray]",
            detector.run(images, postprocess=False),  # type: ignore[union-attr]
        )
        _tensor, ratios, padding = detector.preprocess(images)  # type: ignore[union-attr]
        postprocessed = detector.postprocess(raw, ratios, padding)  # type: ignore[union-attr]
        dets = detector.get_detections(postprocessed)  # type: ignore[union-attr]
        if isinstance(dets, list) and dets and not isinstance(dets[0], list):
            return [dets]  # type: ignore[list-item]
        return dets  # type: ignore[return-value]

    err_msg = f"Unknown mode: {mode}"
    raise ValueError(err_msg)


def _has_detection_with_class(detections: list[Detection], class_id: int) -> bool:
    """Check if any detection has the specified class ID."""
    return any(det[2] == class_id for det in detections)


def _skip_if_no_rtdetrv3() -> None:
    """Skip test if RTDETRv3 config is not available."""
    if RTDETRV3_MODEL_ID not in DETECTOR_CONFIG:
        pytest.skip(f"{RTDETRV3_MODEL_ID} not in DETECTOR_CONFIG")


# =============================================================================
# Section 1: Detection Correctness
# =============================================================================


class TestDetectorHorseDetection:
    """Test all detectors can find horse in horse.jpg."""

    @pytest.mark.parametrize("model_id", ALL_MODELS)
    @pytest.mark.parametrize("mode", INFERENCE_MODES)
    @pytest.mark.parametrize("preprocessor", PREPROCESSORS)
    def test_horse_detection(self, model_id: str, mode: str, preprocessor: str) -> None:
        """All models should detect horse in horse.jpg via all inference modes."""
        engine_path = build_detector(model_id)
        config = DETECTOR_CONFIG[model_id]
        model_class = config["model_class"]

        detector = model_class(
            engine_path,
            preprocessor=preprocessor,
            warmup=False,
            no_warn=True,
        )

        image = read_image(HORSE_IMAGE_PATH)
        results = _run_inference(detector, [image], mode)

        assert len(results) == 1, f"Expected 1 result batch, got {len(results)}"
        detections = results[0]
        assert len(detections) >= 1, f"{model_id}: Expected at least 1 detection"

        has_horse = _has_detection_with_class(detections, HORSE_CLASS_ID)
        if not has_horse:
            classes_detected = [det[2] for det in detections]
            pytest.fail(
                f"{model_id} did not detect horse (class {HORSE_CLASS_ID}). "
                f"Classes detected: {classes_detected}"
            )

        del detector


class TestDetectorPeopleDetection:
    """Test all detectors can find people in people.jpeg."""

    @pytest.mark.parametrize("model_id", ALL_MODELS)
    @pytest.mark.parametrize("mode", INFERENCE_MODES)
    def test_people_detection(self, model_id: str, mode: str) -> None:
        """All models should detect people in people.jpeg."""
        engine_path = build_detector(model_id)
        config = DETECTOR_CONFIG[model_id]
        model_class = config["model_class"]

        detector = model_class(
            engine_path,
            preprocessor="cpu",
            warmup=False,
            no_warn=True,
        )

        image = read_image(PEOPLE_IMAGE_PATH)
        results = _run_inference(detector, [image], mode)

        assert len(results) == 1, f"Expected 1 result batch, got {len(results)}"
        detections = results[0]
        assert len(detections) >= 1, f"{model_id}: Expected at least 1 detection"

        has_person = _has_detection_with_class(detections, PERSON_CLASS_ID)
        if not has_person:
            classes_detected = [det[2] for det in detections]
            pytest.fail(
                f"{model_id} did not detect person (class {PERSON_CLASS_ID}). "
                f"Classes detected: {classes_detected}"
            )

        del detector


class TestDetectorOutputValidity:
    """Test that detector outputs are valid."""

    @pytest.mark.parametrize("model_id", ALL_MODELS)
    def test_bbox_coordinates_valid(self, model_id: str) -> None:
        """Verify bounding box coordinates are valid."""
        engine_path = build_detector(model_id)
        config = DETECTOR_CONFIG[model_id]
        model_class = config["model_class"]

        detector = model_class(
            engine_path,
            preprocessor="cpu",
            warmup=False,
            no_warn=True,
        )

        image = read_image(HORSE_IMAGE_PATH)
        results = _run_inference(detector, [image], "end2end")

        for detections in results:
            for bbox, score, class_id in detections:
                x1, y1, x2, y2 = bbox

                assert x2 > x1, f"Invalid bbox width: x1={x1}, x2={x2}"
                assert y2 > y1, f"Invalid bbox height: y1={y1}, y2={y2}"

                assert 0.0 <= score <= 1.0, f"Invalid score: {score}"

                assert class_id >= 0, f"Invalid class_id: {class_id}"

        del detector

    @pytest.mark.parametrize("model_id", ALL_MODELS)
    def test_confidence_threshold(self, model_id: str) -> None:
        """Verify confidence threshold is respected."""
        engine_path = build_detector(model_id)
        config = DETECTOR_CONFIG[model_id]
        model_class = config["model_class"]

        conf_thres = 0.5

        detector = model_class(
            engine_path,
            preprocessor="cpu",
            conf_thres=conf_thres,
            warmup=False,
            no_warn=True,
        )

        image = read_image(HORSE_IMAGE_PATH)
        results = _run_inference(detector, [image], "end2end")

        for detections in results:
            for _bbox, score, _class_id in detections:
                assert score >= conf_thres, (
                    f"Detection with score {score} below threshold {conf_thres}"
                )

        del detector


class TestPreprocessorConsistency:
    """Test that all preprocessors produce consistent results."""

    @pytest.mark.parametrize("model_id", ALL_MODELS)
    def test_preprocessor_results_similar(self, model_id: str) -> None:
        """All preprocessors should produce similar detection results."""
        engine_path = build_detector(model_id)
        config = DETECTOR_CONFIG[model_id]
        model_class = config["model_class"]

        image = read_image(HORSE_IMAGE_PATH)
        results: dict[str, list[Detection]] = {}

        for preprocessor in PREPROCESSORS:
            detector = model_class(
                engine_path,
                preprocessor=preprocessor,
                warmup=False,
                no_warn=True,
            )
            inference_results = _run_inference(detector, [image], "end2end")
            results[preprocessor] = inference_results[0] if inference_results else []
            del detector

        counts = {k: len(v) for k, v in results.items()}
        min_count = min(counts.values())
        max_count = max(counts.values())

        assert max_count - min_count <= 2, f"{model_id}: Detection count variance too high: {counts}"

        for preprocessor, detections in results.items():
            assert len(detections) >= 1, (
                f"{model_id} with {preprocessor}: Expected at least 1 detection"
            )


# =============================================================================
# Section 2: RTDETRv3 Regression
# =============================================================================


@pytest.mark.regression
class TestRTDETRv3Regression:
    """Regression tests for RTDETRv3 dtype and input ordering bugs."""

    @pytest.mark.parametrize("preprocessor", ["cpu", "cuda", "trt"])
    def test_end2end_returns_nonzero_bboxes(self, preprocessor: str) -> None:
        """
        Verify end2end() returns actual bboxes, not zeros.

        Regression test for dtype mismatch bug where orig_size buffer
        was allocated as int32 but RTDETRv3 expects float32.
        """
        _skip_if_no_rtdetrv3()
        engine_path = build_detector(RTDETRV3_MODEL_ID)
        config = DETECTOR_CONFIG[RTDETRV3_MODEL_ID]
        model_class = config["model_class"]

        detector = model_class(
            engine_path,
            preprocessor=preprocessor,
            warmup=False,
            no_warn=True,
        )

        image = read_image(HORSE_IMAGE_PATH)
        result = detector.end2end(image)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) >= 1, "Expected at least one detection"

        for bbox, score, class_id in result:
            assert bbox != (0, 0, 0, 0), f"Got zero bbox with score {score}, class {class_id}"
            x1, y1, x2, y2 = bbox
            assert x2 > x1, f"Invalid bbox width: {bbox}"
            assert y2 > y1, f"Invalid bbox height: {bbox}"

        del detector

    @pytest.mark.parametrize("preprocessor", ["cpu", "cuda", "trt"])
    def test_end2end_matches_run(self, preprocessor: str) -> None:
        """
        Verify end2end() and run() produce equivalent results.

        Validates that the direct_preproc path (end2end) matches
        the standard preprocess/run/postprocess path.
        """
        _skip_if_no_rtdetrv3()
        engine_path = build_detector(RTDETRV3_MODEL_ID)
        config = DETECTOR_CONFIG[RTDETRV3_MODEL_ID]
        model_class = config["model_class"]

        detector = model_class(
            engine_path,
            preprocessor=preprocessor,
            warmup=False,
            no_warn=True,
        )

        image = read_image(HORSE_IMAGE_PATH)

        result_e2e = detector.end2end(image)

        result_run = detector.run(image)
        result_run_dets = detector.get_detections(result_run)

        assert abs(len(result_e2e) - len(result_run_dets)) <= 2, (
            f"Detection count mismatch: e2e={len(result_e2e)}, run={len(result_run_dets)}"
        )

        if len(result_e2e) > 0 and len(result_run_dets) > 0:
            e2e_sorted = sorted(result_e2e, key=lambda x: -x[1])
            run_sorted = sorted(result_run_dets, key=lambda x: -x[1])

            e2e_top_bbox = np.array(e2e_sorted[0][0])
            run_top_bbox = np.array(run_sorted[0][0])
            np.testing.assert_allclose(
                e2e_top_bbox,
                run_top_bbox,
                atol=5.0,
                err_msg=f"Top bbox mismatch: e2e={e2e_sorted[0]}, run={run_sorted[0]}",
            )

        del detector

    @pytest.mark.skipif(
        not trtutils.FLAGS.EXEC_ASYNC_V3,
        reason="CUDA graph requires async_v3 backend",
    )
    @pytest.mark.parametrize("preprocessor", ["cuda", "trt"])
    def test_cuda_graph_end2end(self, preprocessor: str) -> None:
        """
        Verify CUDA graph end2end works for RTDETRv3.

        Regression test for the input ordering bug where
        _end2end_graph_core built inputs as [image, im_shape, scale_factor]
        but RTDETRv3 expects [im_shape, image, scale_factor].
        """
        _skip_if_no_rtdetrv3()
        engine_path = build_detector(RTDETRV3_MODEL_ID)
        config = DETECTOR_CONFIG[RTDETRV3_MODEL_ID]
        model_class = config["model_class"]

        detector = model_class(
            engine_path,
            preprocessor=preprocessor,
            backend="async_v3",
            cuda_graph=True,
            warmup=True,
            no_warn=True,
        )

        image = read_image(HORSE_IMAGE_PATH)
        result = detector.end2end(image)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) >= 1, "Expected at least one detection with CUDA graph"

        for bbox, score, class_id in result:
            assert bbox != (0, 0, 0, 0), f"Got zero bbox with score {score}, class {class_id}"

        assert detector._e2e_graph is not None
        assert detector._e2e_graph.is_captured is True

        del detector

    @pytest.mark.skipif(
        not trtutils.FLAGS.EXEC_ASYNC_V3,
        reason="CUDA graph requires async_v3 backend",
    )
    @pytest.mark.parametrize("preprocessor", ["cuda", "trt"])
    def test_cuda_graph_matches_non_graph(self, preprocessor: str) -> None:
        """Verify CUDA graph and non-graph produce equivalent results."""
        _skip_if_no_rtdetrv3()
        engine_path = build_detector(RTDETRV3_MODEL_ID)
        config = DETECTOR_CONFIG[RTDETRV3_MODEL_ID]
        model_class = config["model_class"]

        detector_graph = model_class(
            engine_path,
            preprocessor=preprocessor,
            backend="async_v3",
            cuda_graph=True,
            warmup=True,
            no_warn=True,
        )
        detector_normal = model_class(
            engine_path,
            preprocessor=preprocessor,
            backend="async_v3",
            cuda_graph=False,
            warmup=True,
            no_warn=True,
        )

        image = read_image(HORSE_IMAGE_PATH)

        result_graph = detector_graph.end2end(image)
        result_normal = detector_normal.end2end(image)

        assert result_graph is not None
        assert result_normal is not None

        assert abs(len(result_graph) - len(result_normal)) <= 2

        del detector_graph
        del detector_normal


@pytest.mark.regression
class TestRTDETRv3InputSchema:
    """Tests verifying RTDETRv3 input schema handling."""

    def test_input_schema_is_rt_detr_v3(self) -> None:
        """Verify the detector correctly identifies RT_DETR_V3 input schema."""
        _skip_if_no_rtdetrv3()

        engine_path = build_detector(RTDETRV3_MODEL_ID)
        config = DETECTOR_CONFIG[RTDETRV3_MODEL_ID]
        model_class = config["model_class"]

        detector = model_class(
            engine_path,
            warmup=False,
            no_warn=True,
        )

        assert detector.input_schema == InputSchema.RT_DETR_V3
        assert detector._use_image_size is True
        assert detector._use_scale_factor is True

        del detector

    def test_preprocessor_orig_size_buffer_dtype(self) -> None:
        """Verify preprocessor allocates orig_size buffer with float32 dtype."""
        _skip_if_no_rtdetrv3()

        engine_path = build_detector(RTDETRV3_MODEL_ID)
        config = DETECTOR_CONFIG[RTDETRV3_MODEL_ID]
        model_class = config["model_class"]

        detector = model_class(
            engine_path,
            preprocessor="cuda",
            warmup=False,
            no_warn=True,
        )

        preproc = detector._preprocessor
        assert isinstance(preproc, CUDAPreprocessor)
        assert hasattr(preproc, "_orig_size_dtype")
        assert preproc._orig_size_dtype == np.dtype(np.float32)

        del detector
