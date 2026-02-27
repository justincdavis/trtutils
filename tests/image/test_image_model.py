# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="misc"
"""Tests for ImageModel base class functionality."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
YOLOV10_ONNX = DATA_DIR / "yolov10" / "yolov10n_640.onnx"


# ---------------------------------------------------------------------------
# GPU tests — require a real engine
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def yolov10_engine(build_test_engine) -> Path:
    """Build and cache a YOLOv10n engine for the test module."""
    if not YOLOV10_ONNX.exists():
        pytest.skip("yolov10n_640.onnx not available")
    return build_test_engine(YOLOV10_ONNX)


@pytest.mark.gpu
class TestImageModelInit:
    """Test ImageModel initialization with various options."""

    @pytest.mark.parametrize("preprocessor", ["cpu", "cuda", "trt"])
    def test_init_with_preprocessor_types(self, yolov10_engine: Path, preprocessor: str) -> None:
        """All 3 preprocessor backends initialize correctly."""
        from trtutils.models import YOLOv10

        model = YOLOv10(yolov10_engine, preprocessor=preprocessor, warmup=False)
        assert model is not None

    @pytest.mark.parametrize("resize_method", ["linear", "letterbox"])
    def test_init_with_resize_methods(self, yolov10_engine: Path, resize_method: str) -> None:
        """Both resize methods initialize correctly."""
        from trtutils.models import YOLOv10

        model = YOLOv10(yolov10_engine, resize_method=resize_method, warmup=False)
        assert model is not None

    @pytest.mark.parametrize("backend", ["auto"])
    def test_init_with_backends(self, yolov10_engine: Path, backend: str) -> None:
        """Supported execution backends initialize correctly."""
        from trtutils.models import YOLOv10

        model = YOLOv10(yolov10_engine, backend=backend, warmup=False)
        assert model is not None


@pytest.mark.gpu
class TestImageModelPreprocessing:
    """Test ImageModel preprocessing."""

    def test_preprocess_single_image(self, yolov10_engine: Path, horse_image: np.ndarray) -> None:
        """Preprocessing single np.ndarray input produces correct shape."""
        from trtutils.models import YOLOv10

        model = YOLOv10(yolov10_engine, warmup=False)
        result, _, _ = model.preprocess([horse_image])
        assert isinstance(result, np.ndarray)
        assert result.ndim == 4  # (batch, C, H, W)
        assert result.shape[0] == 1

    def test_preprocess_batch(self, yolov10_engine: Path, test_images: list[np.ndarray]) -> None:
        """Preprocessing list input produces batch output."""
        from trtutils.models import YOLOv10

        # Use TRT preprocessor with single image (engine has static batch=1)
        model = YOLOv10(yolov10_engine, preprocessor="cpu", warmup=False)
        images = test_images[:2]
        result, _, _ = model.preprocess(images)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(images)

    def test_preprocess_output_shape(self, yolov10_engine: Path, horse_image: np.ndarray) -> None:
        """Preprocessed output has correct (1, 3, 640, 640) shape."""
        from trtutils.models import YOLOv10

        model = YOLOv10(yolov10_engine, warmup=False)
        result, _, _ = model.preprocess([horse_image])
        assert result.shape == (1, 3, 640, 640)


@pytest.mark.gpu
class TestImageModelUtilities:
    """Test ImageModel utility methods."""

    def test_get_random_input(self, yolov10_engine: Path) -> None:
        """get_random_input generates valid random image tensors."""
        from trtutils.models import YOLOv10

        model = YOLOv10(yolov10_engine, warmup=False)
        rand_input = model.get_random_input()
        # Returns list[np.ndarray] (one per engine input)
        assert isinstance(rand_input, list)
        assert len(rand_input) > 0
        assert isinstance(rand_input[0], np.ndarray)

    def test_mock_run(self, yolov10_engine: Path) -> None:
        """Engine mock_execute runs without error."""
        from trtutils.models import YOLOv10

        model = YOLOv10(yolov10_engine, warmup=False)
        engine = model.engine  # public property
        engine.mock_execute()  # should not raise
