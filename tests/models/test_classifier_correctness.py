# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Classifier output correctness tests (GPU required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from .conftest import (
    CLASSIFIER_EXPECTED,
    CLASSIFIER_MODELS,
    _resolve_model_class,
    build_model_engine,
)

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SIMPLE_ONNX = DATA_DIR / "simple.onnx"


# ---------------------------------------------------------------------------
# Helpers -- build a minimal classifier-shaped engine
# ---------------------------------------------------------------------------
def _find_classifier_onnx() -> Path:
    """Find an available classifier ONNX model in the data directory."""
    # simple.onnx is always available for basic tests
    if SIMPLE_ONNX.exists():
        return SIMPLE_ONNX
    pytest.skip("No classifier ONNX model available")
    # unreachable but keeps type checkers happy
    return SIMPLE_ONNX


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def classifier_engine(build_test_engine) -> Path:
    """Build and cache a classifier engine for the test module."""
    onnx_path = _find_classifier_onnx()
    return build_test_engine(onnx_path)


@pytest.fixture(scope="module")
def classifier_engine_batch(build_test_engine) -> Path:
    """Build a batch-capable classifier engine for batch tests."""
    onnx_path = _find_classifier_onnx()
    try:
        return build_test_engine(onnx_path, batch_size=2)
    except Exception as exc:
        pytest.skip(f"Batch classifier engine unavailable: {exc}")


# ---------------------------------------------------------------------------
# Classifier base class tests (using Classifier directly)
# ---------------------------------------------------------------------------
class TestClassifierOutput:
    """Tests for Classifier output format using a simple engine."""

    def test_classifier_run_returns_list(
        self,
        classifier_engine,
        images,
    ) -> None:
        """Classifier.run() should return a list of ndarrays."""
        horse_image = images["horse"].array
        from trtutils.image import Classifier

        cls = Classifier(classifier_engine, warmup=False)
        outputs = cls.run(horse_image)
        assert isinstance(outputs, list)
        for arr in outputs:
            assert isinstance(arr, np.ndarray)

    def test_classifier_run_raw(
        self,
        classifier_engine,
        images,
    ) -> None:
        """run(postprocess=False) returns raw output ndarrays."""
        horse_image = images["horse"].array
        from trtutils.image import Classifier

        cls = Classifier(classifier_engine, warmup=False)
        outputs = cls.run(horse_image, postprocess=False)
        assert isinstance(outputs, list)
        for arr in outputs:
            assert isinstance(arr, np.ndarray)

    def test_classifier_postprocess_format(
        self,
        classifier_engine,
        images,
    ) -> None:
        """postprocess() should return list[list[ndarray]]."""
        horse_image = images["horse"].array
        from trtutils.image import Classifier

        cls = Classifier(classifier_engine, warmup=False)
        raw = cls.run(horse_image, postprocess=False)
        postprocessed = cls.postprocess(raw)
        assert isinstance(postprocessed, list)
        for per_image in postprocessed:
            assert isinstance(per_image, list)
            for arr in per_image:
                assert isinstance(arr, np.ndarray)

    def test_get_classifications_format(
        self,
        classifier_engine,
        images,
    ) -> None:
        """get_classifications returns list[tuple[int, float]]."""
        horse_image = images["horse"].array
        from trtutils.image import Classifier

        cls = Classifier(classifier_engine, warmup=False)
        outputs = cls.run(horse_image)
        classifications = cls.get_classifications(outputs, top_k=5)
        assert isinstance(classifications, list)
        assert len(classifications) <= 5
        for entry in classifications:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            cls_id, score = entry
            assert isinstance(cls_id, (int, np.integer))
            assert isinstance(score, (float, np.floating))

    def test_get_classifications_top_k(
        self,
        classifier_engine,
        images,
    ) -> None:
        """top_k parameter should limit number of results."""
        horse_image = images["horse"].array
        from trtutils.image import Classifier

        cls = Classifier(classifier_engine, warmup=False)
        outputs = cls.run(horse_image)
        top1 = cls.get_classifications(outputs, top_k=1)
        top3 = cls.get_classifications(outputs, top_k=3)
        assert len(top1) <= 1
        assert len(top3) <= 3

    def test_get_classifications_scores_descending(
        self,
        classifier_engine,
        images,
    ) -> None:
        """Classification scores should be in descending order."""
        horse_image = images["horse"].array
        from trtutils.image import Classifier

        cls = Classifier(classifier_engine, warmup=False)
        outputs = cls.run(horse_image)
        classifications = cls.get_classifications(outputs, top_k=10)
        if len(classifications) > 1:
            scores = [float(s) for _, s in classifications]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1] - 1e-6

    def test_callable_matches_run(
        self,
        classifier_engine,
        images,
    ) -> None:
        """__call__ should produce the same result as run()."""
        horse_image = images["horse"].array
        from trtutils.image import Classifier

        cls = Classifier(classifier_engine, warmup=False)
        out_run = cls.run(horse_image)
        out_call = cls(horse_image)
        assert len(out_run) == len(out_call)


# ---------------------------------------------------------------------------
# Batch tests
# ---------------------------------------------------------------------------
class TestClassifierBatch:
    """Batch inference tests."""

    def test_batch_run_returns_nested(
        self,
        classifier_engine_batch,
        random_images,
    ) -> None:
        """Batch run with postprocess returns list[list[ndarray]]."""
        from trtutils.image import Classifier

        cls = Classifier(classifier_engine_batch, warmup=False)
        imgs = random_images(count=2, height=480, width=640)
        outputs = cls.run(imgs)
        assert isinstance(outputs, list)
        assert len(outputs) == 2
        for per_image in outputs:
            assert isinstance(per_image, list)

    def test_batch_get_classifications(
        self,
        classifier_engine_batch,
        random_images,
    ) -> None:
        """Batch get_classifications returns list[list[tuple]]."""
        from trtutils.image import Classifier

        cls = Classifier(classifier_engine_batch, warmup=False)
        imgs = random_images(count=2, height=480, width=640)
        outputs = cls.run(imgs)
        results = cls.get_classifications(outputs, top_k=3)
        assert isinstance(results, list)
        assert len(results) == 2
        for per_image_cls in results:
            assert isinstance(per_image_cls, list)


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------
class TestClassifierEnd2End:
    """End-to-end inference tests."""

    def test_end2end_single_image(
        self,
        classifier_engine,
        images,
    ) -> None:
        """end2end() on single image returns list[tuple[int, float]]."""
        horse_image = images["horse"].array
        from trtutils.image import Classifier

        cls = Classifier(classifier_engine, warmup=False)
        result = cls.end2end(horse_image, top_k=5)
        assert isinstance(result, list)
        for entry in result:
            assert isinstance(entry, tuple)
            assert len(entry) == 2


# ---------------------------------------------------------------------------
# Data-driven multi-model classifier correctness tests
# ---------------------------------------------------------------------------
_CLASSIFIER_MODEL_IDS = list(CLASSIFIER_MODELS.keys())


def _make_cls_expected_ids() -> list[str]:
    """Create human-readable IDs for CLASSIFIER_EXPECTED entries."""
    return [Path(e["image"]).stem for e in CLASSIFIER_EXPECTED]


_CLS_EXPECTED_IDS = _make_cls_expected_ids()


@pytest.mark.correctness
@pytest.mark.download
class TestClassifierImageCorrectness:
    """Data-driven correctness: classifiers must find expected classes."""

    @pytest.mark.parametrize(
        "expected_entry",
        CLASSIFIER_EXPECTED,
        ids=_CLS_EXPECTED_IDS,
    )
    @pytest.mark.parametrize("model_id", _CLASSIFIER_MODEL_IDS)
    def test_image_correctness(
        self,
        expected_entry: dict,
        model_id: str,
    ) -> None:
        """At least one expected class appears in top-k predictions."""
        import cv2

        cls_name, model_name, imgsz = CLASSIFIER_MODELS[model_id]
        engine_path = build_model_engine(cls_name, model_name, imgsz)

        model_class = _resolve_model_class(cls_name)
        classifier = model_class(
            engine_path,
            warmup=False,
            no_warn=True,
        )

        image_path = str(BASE_DIR / expected_entry["image"])
        image = cv2.imread(image_path)
        if image is None:
            pytest.skip(f"Image not found: {image_path}")

        top_k = expected_entry["top_k"]
        predictions = classifier.end2end(image, top_k=top_k)

        predicted_classes = [int(cls_id) for cls_id, _score in predictions]
        has_match = any(c in expected_entry["expected_top_k_classes"] for c in predicted_classes)
        assert has_match, (
            f"{model_id}: none of {predicted_classes} match "
            f"expected {expected_entry['expected_top_k_classes']}"
        )

        del classifier


@pytest.mark.correctness
@pytest.mark.download
class TestClassifierOutputFormat:
    """Validate classifier output format across all models."""

    @pytest.mark.parametrize(
        "expected_entry",
        CLASSIFIER_EXPECTED,
        ids=_CLS_EXPECTED_IDS,
    )
    @pytest.mark.parametrize("model_id", _CLASSIFIER_MODEL_IDS)
    def test_output_format(
        self,
        expected_entry: dict,
        model_id: str,
    ) -> None:
        """Output is list[tuple[int, float]], scores in [0,1], descending."""
        import cv2

        cls_name, model_name, imgsz = CLASSIFIER_MODELS[model_id]
        engine_path = build_model_engine(cls_name, model_name, imgsz)

        model_class = _resolve_model_class(cls_name)
        classifier = model_class(
            engine_path,
            warmup=False,
            no_warn=True,
        )

        image_path = str(BASE_DIR / expected_entry["image"])
        image = cv2.imread(image_path)
        if image is None:
            pytest.skip(f"Image not found: {image_path}")

        top_k = expected_entry["top_k"]
        predictions = classifier.end2end(image, top_k=top_k)

        # Must be a list
        assert isinstance(predictions, list)

        # Each entry is (int, float)
        for entry in predictions:
            assert isinstance(entry, tuple), f"Expected tuple, got {type(entry)}"
            assert len(entry) == 2
            cls_id, score = entry
            assert isinstance(cls_id, (int, np.integer))
            assert isinstance(score, (float, np.floating))
            assert 0.0 <= float(score) <= 1.0, f"{model_id}: score {score} out of [0, 1]"

        # Scores in descending order
        if len(predictions) > 1:
            scores = [float(s) for _, s in predictions]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1] - 1e-6, (
                    f"{model_id}: scores not descending at index {i}: {scores[i]} < {scores[i + 1]}"
                )

        del classifier


@pytest.mark.correctness
@pytest.mark.download
class TestClassifierPreprocessorConsistency:
    """Verify cpu/cuda/trt preprocessors agree on top-1 class."""

    @pytest.mark.parametrize(
        "expected_entry",
        CLASSIFIER_EXPECTED,
        ids=_CLS_EXPECTED_IDS,
    )
    @pytest.mark.parametrize("model_id", _CLASSIFIER_MODEL_IDS)
    def test_preprocessor_consistency(
        self,
        expected_entry: dict,
        model_id: str,
    ) -> None:
        """All preprocessors produce the same top-1 class."""
        import cv2

        cls_name, model_name, imgsz = CLASSIFIER_MODELS[model_id]
        engine_path = build_model_engine(cls_name, model_name, imgsz)

        model_class = _resolve_model_class(cls_name)

        image_path = str(BASE_DIR / expected_entry["image"])
        image = cv2.imread(image_path)
        if image is None:
            pytest.skip(f"Image not found: {image_path}")

        preprocessors = ["cpu", "cuda", "trt"]
        top1_classes: dict[str, int] = {}

        for preproc in preprocessors:
            classifier = model_class(
                engine_path,
                preprocessor=preproc,
                warmup=False,
                no_warn=True,
            )
            predictions = classifier.end2end(image, top_k=1)
            if predictions:
                top1_classes[preproc] = int(predictions[0][0])
            del classifier

        unique_classes = set(top1_classes.values())
        assert len(unique_classes) == 1, (
            f"{model_id}: preprocessors disagree on top-1: {top1_classes}"
        )
