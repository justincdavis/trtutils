# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import pytest

from trtutils.models import (
    DEIM,
    DFINE,
    RFDETR,
    YOLO3,
    YOLO5,
    YOLO7,
    YOLO8,
    YOLO9,
    YOLO10,
    YOLO11,
    YOLO12,
    YOLO13,
    YOLOX,
    DEIMv2,
    RTDETRv1,
    RTDETRv2,
    RTDETRv3,
)

from .paths import HORSE_IMAGE_PATH

# Each model config: (ModelClass, model_name, imgsz) - imgsz is None to use default
MODEL_CONFIGS = [
    # YOLO models (all use default imgsz=640)
    (YOLOX, "yoloxn", None),
    (YOLO3, "yolov3tu", None),
    (YOLO5, "yolov5nu", None),
    (YOLO7, "yolov7t", None),
    (YOLO8, "yolov8n", None),
    (YOLO9, "yolov9t", None),
    (YOLO10, "yolov10n", None),
    (YOLO11, "yolov11n", None),
    (YOLO12, "yolov12n", None),
    (YOLO13, "yolov13n", None),
    # DETR models (most use default imgsz=640)
    (RTDETRv1, "rtdetrv1_r18", None),
    (RTDETRv2, "rtdetrv2_r18", None),
    (RTDETRv3, "rtdetrv3_r18", None),
    (DFINE, "dfine_n", None),
    (DEIM, "deim_dfine_n", None),
    # Special cases - these models require specific imgsz values
    (DEIMv2, "deimv2_atto", 320),
    (RFDETR, "rfdetr_n", 384),
]


@pytest.mark.parametrize(
    ("model_class", "model_name", "imgsz"),
    MODEL_CONFIGS,
    ids=[cfg[1] for cfg in MODEL_CONFIGS],
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
    with tempfile.TemporaryDirectory() as temp_dir:
        onnx_path = Path(temp_dir) / f"{model_name}.onnx"
        engine_path = Path(temp_dir) / f"{model_name}.engine"

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
