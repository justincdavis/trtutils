# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="misc,import-untyped"
"""
Tests for TRT preprocessing ONNX model engines.

Port from: tests/legacy/image/onnx/test_image_preproc.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from trtutils import TRTEngine
from trtutils.image.onnx_models import build_image_preproc, build_image_preproc_imagenet
from trtutils.image.preprocessors import preprocess

_TRT_VERSION: str | None = None
try:
    import tensorrt as _trt_module  # type: ignore[import-untyped]

    _TRT_VERSION = str(_trt_module.__version__)
except ImportError:
    _trt_module = None  # type: ignore[assignment]

_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
_HORSE_IMAGE_PATH = _DATA_DIR / "horse.jpg"


def _trt_available() -> bool:
    """Check if TensorRT is available."""
    return _TRT_VERSION is not None


@pytest.mark.gpu
class TestTRTPreprocEngine:
    """Tests for TRT preprocessing ONNX engines."""

    def test_trt_preproc_matches_cpu(self) -> None:
        """TRT preprocessing engine output matches CPU preprocessing."""
        if not _trt_available():
            pytest.skip("TensorRT not available")
        if not _HORSE_IMAGE_PATH.exists():
            pytest.skip("Horse test image not found")

        img = cv2.imread(str(_HORSE_IMAGE_PATH))
        if img is None:
            pytest.skip("Failed to read test image")

        output_shape = 640
        o_range = (0.0, 1.0)
        scale = o_range[1] / 255.0
        offset = o_range[0]

        img = cv2.resize(img, (output_shape, output_shape))  # type: ignore[arg-type]

        cpu_result, _, _ = preprocess(
            [img], (output_shape, output_shape), np.dtype(np.float32), input_range=o_range
        )
        cpu_result = cpu_result[0]

        engine_path = build_image_preproc(
            (output_shape, output_shape),
            np.dtype(np.float32),
            trt_version=str(_TRT_VERSION),
        )
        engine = TRTEngine(engine_path)
        engine.mock_execute()

        all_result = engine.execute(
            [
                img,
                np.array((scale,), dtype=np.float32),
                np.array((offset,), dtype=np.float32),
            ]
        )
        trt_result = all_result[0]
        if trt_result.ndim == 4:
            trt_result = trt_result[0]

        assert trt_result.shape == cpu_result.shape
        assert trt_result.dtype == cpu_result.dtype
        assert np.min(trt_result) >= 0.0
        assert np.max(trt_result) <= 1.0

        diff_mask = np.any(cpu_result != trt_result, axis=-1)
        avg_diff = np.mean(np.abs(cpu_result[diff_mask] - trt_result[diff_mask]))
        assert avg_diff < 0.0001, f"avg diff: {avg_diff}"
        assert np.allclose(trt_result, cpu_result, rtol=5e-4, atol=5e-4)

        del engine

    def test_trt_preproc_imagenet_matches_cpu(self) -> None:
        """TRT ImageNet preprocessing engine output matches CPU preprocessing."""
        if not _trt_available():
            pytest.skip("TensorRT not available")
        if not _HORSE_IMAGE_PATH.exists():
            pytest.skip("Horse test image not found")

        img = cv2.imread(str(_HORSE_IMAGE_PATH))
        if img is None:
            pytest.skip("Failed to read test image")

        output_shape = 640
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        img = cv2.resize(img, (output_shape, output_shape))  # type: ignore[arg-type]

        cpu_result, _, _ = preprocess(
            [img],
            (output_shape, output_shape),
            np.dtype(np.float32),
            input_range=(0.0, 1.0),
            mean=mean,
            std=std,
        )
        cpu_result = cpu_result[0]

        engine_path = build_image_preproc_imagenet(
            (output_shape, output_shape),
            np.dtype(np.float32),
            trt_version=str(_TRT_VERSION),
        )
        engine = TRTEngine(engine_path)
        engine.mock_execute()

        mean_array = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
        std_array = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)

        all_result = engine.execute([img, mean_array, std_array])
        trt_result = all_result[0]
        if trt_result.ndim == 4:
            trt_result = trt_result[0]

        assert trt_result.shape == cpu_result.shape
        assert trt_result.dtype == cpu_result.dtype

        # ImageNet normalization engine uses fp16, so use relaxed tolerance
        assert np.allclose(trt_result, cpu_result, rtol=2e-3, atol=2e-3)

        del engine

    def test_numerical_tolerance(self) -> None:
        """TRT preproc engine meets expected numerical tolerance bounds."""
        if not _trt_available():
            pytest.skip("TensorRT not available")
        if not _HORSE_IMAGE_PATH.exists():
            pytest.skip("Horse test image not found")

        img = cv2.imread(str(_HORSE_IMAGE_PATH))
        if img is None:
            pytest.skip("Failed to read test image")

        output_shape = 640
        o_range = (0.0, 1.0)
        scale = o_range[1] / 255.0
        offset = o_range[0]
        img = cv2.resize(img, (output_shape, output_shape))  # type: ignore[arg-type]

        cpu_result, _, _ = preprocess(
            [img], (output_shape, output_shape), np.dtype(np.float32), input_range=o_range
        )
        cpu_result = cpu_result[0]

        engine_path = build_image_preproc(
            (output_shape, output_shape),
            np.dtype(np.float32),
            trt_version=str(_TRT_VERSION),
        )
        engine = TRTEngine(engine_path)
        engine.mock_execute()

        all_result = engine.execute(
            [
                img,
                np.array((scale,), dtype=np.float32),
                np.array((offset,), dtype=np.float32),
            ]
        )
        trt_result = all_result[0]
        if trt_result.ndim == 4:
            trt_result = trt_result[0]

        cpu_mean = np.mean(cpu_result)
        trt_mean = np.mean(trt_result)
        assert cpu_mean * 0.99 <= trt_mean <= cpu_mean * 1.01, (
            f"CPU mean: {cpu_mean}, TRT mean: {trt_mean}"
        )

        del engine
