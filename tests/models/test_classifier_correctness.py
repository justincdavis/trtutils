# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Classifier output correctness tests (GPU required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# Classifier base class tests (using Classifier directly)
# ---------------------------------------------------------------------------
class TestClassifierOutput:
    """Tests for Classifier output format using a simple engine."""

    @pytest.mark.gpu
    def test_classifier_run_returns_list(
        self,
        classifier_engine,
        horse_image,
    ) -> None:
        """Classifier.run() should return a list of ndarrays."""
        from trtutils.image import Classifier

        cls = Classifier(classifier_engine, warmup=False)
        outputs = cls.run(horse_image)
        assert isinstance(outputs, list)
        for arr in outputs:
            assert isinstance(arr, np.ndarray)

    @pytest.mark.gpu
    def test_classifier_run_raw(
        self,
        classifier_engine,
        horse_image,
    ) -> None:
        """run(postprocess=False) returns raw output ndarrays."""
        from trtutils.image import Classifier

        cls = Classifier(classifier_engine, warmup=False)
        outputs = cls.run(horse_image, postprocess=False)
        assert isinstance(outputs, list)
        for arr in outputs:
            assert isinstance(arr, np.ndarray)

    @pytest.mark.gpu
    def test_classifier_postprocess_format(
        self,
        classifier_engine,
        horse_image,
    ) -> None:
        """postprocess() should return list[list[ndarray]]."""
        from trtutils.image import Classifier

        cls = Classifier(classifier_engine, warmup=False)
        raw = cls.run(horse_image, postprocess=False)
        postprocessed = cls.postprocess(raw)
        assert isinstance(postprocessed, list)
        for per_image in postprocessed:
            assert isinstance(per_image, list)
            for arr in per_image:
                assert isinstance(arr, np.ndarray)

    @pytest.mark.gpu
    def test_get_classifications_format(
        self,
        classifier_engine,
        horse_image,
    ) -> None:
        """get_classifications returns list[tuple[int, float]]."""
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

    @pytest.mark.gpu
    def test_get_classifications_top_k(
        self,
        classifier_engine,
        horse_image,
    ) -> None:
        """top_k parameter should limit number of results."""
        from trtutils.image import Classifier

        cls = Classifier(classifier_engine, warmup=False)
        outputs = cls.run(horse_image)
        top1 = cls.get_classifications(outputs, top_k=1)
        top3 = cls.get_classifications(outputs, top_k=3)
        assert len(top1) <= 1
        assert len(top3) <= 3

    @pytest.mark.gpu
    def test_get_classifications_scores_descending(
        self,
        classifier_engine,
        horse_image,
    ) -> None:
        """Classification scores should be in descending order."""
        from trtutils.image import Classifier

        cls = Classifier(classifier_engine, warmup=False)
        outputs = cls.run(horse_image)
        classifications = cls.get_classifications(outputs, top_k=10)
        if len(classifications) > 1:
            scores = [float(s) for _, s in classifications]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1] - 1e-6

    @pytest.mark.gpu
    def test_callable_matches_run(
        self,
        classifier_engine,
        horse_image,
    ) -> None:
        """__call__ should produce the same result as run()."""
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

    @pytest.mark.gpu
    def test_batch_run_returns_nested(
        self,
        classifier_engine,
        random_images,
    ) -> None:
        """Batch run with postprocess returns list[list[ndarray]]."""
        from trtutils.image import Classifier

        cls = Classifier(classifier_engine, warmup=False)
        imgs = random_images(count=2, height=480, width=640)
        outputs = cls.run(imgs)
        assert isinstance(outputs, list)
        assert len(outputs) == 2
        for per_image in outputs:
            assert isinstance(per_image, list)

    @pytest.mark.gpu
    def test_batch_get_classifications(
        self,
        classifier_engine,
        random_images,
    ) -> None:
        """Batch get_classifications returns list[list[tuple]]."""
        from trtutils.image import Classifier

        cls = Classifier(classifier_engine, warmup=False)
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

    @pytest.mark.gpu
    def test_end2end_single_image(
        self,
        classifier_engine,
        horse_image,
    ) -> None:
        """end2end() on single image returns list[tuple[int, float]]."""
        from trtutils.image import Classifier

        cls = Classifier(classifier_engine, warmup=False)
        result = cls.end2end(horse_image, top_k=5)
        assert isinstance(result, list)
        for entry in result:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
