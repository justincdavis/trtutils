# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="misc"
"""Tests for the Classifier class."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import numpy as np

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
# Classifier models are optional - skip if not available
CLASSIFIER_ONNX = DATA_DIR / "onnx" / "resnet18.onnx"


@pytest.fixture(scope="module")
def classifier_engine(build_test_engine) -> Path:
    """Build and cache a classifier engine."""
    if not CLASSIFIER_ONNX.exists():
        pytest.skip("Classifier ONNX not available (resnet18.onnx)")
    return build_test_engine(CLASSIFIER_ONNX)


@pytest.mark.gpu
class TestClassifierInference:
    """Test Classifier inference."""

    def test_run_single_image(self, classifier_engine: Path, horse_image: np.ndarray) -> None:
        """run() with single image returns outputs."""
        from trtutils.image import Classifier

        clf = Classifier(classifier_engine, warmup=False)
        results = clf.run([horse_image], postprocess=False)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_run_batch(self, classifier_engine: Path, test_images: list[np.ndarray]) -> None:
        """run() with batch returns outputs for each image."""
        from trtutils.image import Classifier

        clf = Classifier(classifier_engine, warmup=False)
        results = clf.run(test_images, postprocess=False)
        assert isinstance(results, list)

    def test_end2end(self, classifier_engine: Path, horse_image: np.ndarray) -> None:
        """end2end() returns classification results."""
        from trtutils.image import Classifier

        clf = Classifier(classifier_engine, warmup=False)
        classifications = clf.end2end([horse_image])
        assert isinstance(classifications, list)
        assert len(classifications) == 1

    def test_get_classifications(self, classifier_engine: Path, horse_image: np.ndarray) -> None:
        """get_classifications() returns top-k results."""
        from trtutils.image import Classifier

        clf = Classifier(classifier_engine, warmup=False)
        postprocessed = clf.run([horse_image], postprocess=True)
        classifications = clf.get_classifications(postprocessed, top_k=5)
        assert len(classifications) >= 1


@pytest.mark.gpu
class TestClassifierPostprocessing:
    """Test Classifier postprocessing."""

    def test_postprocess_returns_probabilities(
        self, classifier_engine: Path, horse_image: np.ndarray
    ) -> None:
        """Postprocessed output probabilities sum to approximately 1."""
        from trtutils.image import Classifier

        clf = Classifier(classifier_engine, warmup=False)
        raw = clf.run([horse_image], postprocess=False)
        processed = clf.postprocess(raw)
        # Softmax probabilities should sum to ~1
        assert isinstance(processed, list)

    def test_top_k_limits(self, classifier_engine: Path, horse_image: np.ndarray) -> None:
        """top_k parameter controls number of results in end2end."""
        from trtutils.image import Classifier

        clf = Classifier(classifier_engine, warmup=False)
        results_5 = clf.end2end([horse_image], top_k=5)
        results_1 = clf.end2end([horse_image], top_k=1)
        assert isinstance(results_5, list)
        assert isinstance(results_1, list)
