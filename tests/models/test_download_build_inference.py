# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="misc"
from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import pytest

from tests.helpers import HORSE_IMAGE_PATH

from .common import MODEL_DOWNLOAD_CONFIGS

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = [pytest.mark.gpu, pytest.mark.download, pytest.mark.slow]


@contextmanager
def _temporary_dir() -> Iterator[Path]:
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path)


@pytest.mark.parametrize(
    ("model_class", "model_name", "imgsz"),
    MODEL_DOWNLOAD_CONFIGS,
    ids=[cfg[1] for cfg in MODEL_DOWNLOAD_CONFIGS],
)
def test_download_build_inference(model_class: type, model_name: str, imgsz: int | None) -> None:
    """
    Test the complete workflow: download model, build engine, run inference.

    For each supported model, this test:
    1. Downloads the smallest variant to ONNX
    2. Builds a TensorRT engine
    3. Runs inference on the horse test image
    4. Asserts that at least one object was detected
    """
    with _temporary_dir() as temp_dir:
        onnx_path = temp_dir / f"{model_name}.onnx"
        engine_path = temp_dir / f"{model_name}.engine"

        # Download the model to ONNX
        model_class.download(  # type: ignore[attr-defined]
            model=model_name,
            output=onnx_path,
            imgsz=imgsz,
            accept=True,
        )
        assert onnx_path.exists(), f"ONNX file was not created: {onnx_path}"

        # Build the TensorRT engine
        model_class.build(  # type: ignore[attr-defined]
            onnx=onnx_path,
            output=engine_path,
            imgsz=imgsz,
            opt_level=1,
        )
        assert engine_path.exists(), f"Engine file was not created: {engine_path}"

        # Load the model and run inference
        detector = model_class(
            engine_path,
            warmup=False,
        )

        image = cv2.imread(HORSE_IMAGE_PATH)
        assert image is not None, f"Failed to read image: {HORSE_IMAGE_PATH}"

        detections = detector.end2end([image])

        # Assert at least 1 detection (the horse in the image)
        assert len(detections) == 1, "Expected detections for 1 image"
        assert len(detections[0]) >= 1, f"Expected at least 1 detection, got {len(detections[0])}"

        del detector
