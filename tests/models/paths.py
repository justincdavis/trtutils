# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Shared path definitions for all model tests."""
from __future__ import annotations

from pathlib import Path

BASE = Path(__file__).parent.parent.parent

# YOLO model paths
YOLO_ENGINE_PATHS: dict[int, Path] = {
    0: BASE / "data" / "engines" / "yolox" / "yoloxn.engine",
    7: BASE / "data" / "engines" / "yolov7" / "yolov7t.engine",
    8: BASE / "data" / "engines" / "yolov8" / "yolov8n.engine",
    9: BASE / "data" / "engines" / "yolov9" / "yolov9t.engine",
    10: BASE / "data" / "engines" / "yolov10" / "yolov10n.engine",
    11: BASE / "data" / "engines" / "yolov11" / "yolov11n.engine",
    12: BASE / "data" / "engines" / "yolov12" / "yolov12n.engine",
    13: BASE / "data" / "engines" / "yolov13" / "yolov13n.engine",
}

YOLO_ONNX_PATHS: dict[int, Path] = {
    0: BASE / "data" / "yolox" / "yoloxn.onnx",
    7: BASE / "data" / "yolov7" / "yolov7t.onnx",
    8: BASE / "data" / "yolov8" / "yolov8n.onnx",
    9: BASE / "data" / "yolov9" / "yolov9t.onnx",
    10: BASE / "data" / "yolov10" / "yolov10n.onnx",
    11: BASE / "data" / "yolov11" / "yolov11n.onnx",
    12: BASE / "data" / "yolov12" / "yolov12n.onnx",
    13: BASE / "data" / "yolov13" / "yolov13n.onnx",
}

# RT-DETR family model paths
RTDETR_ENGINE_PATHS: dict[str, Path] = {
    "rtdetrv1": BASE / "data" / "engines" / "rtdetr" / "rtdetrv1_r18.engine",
    "rtdetrv2": BASE / "data" / "engines" / "rtdetr" / "rtdetrv2_r18.engine",
    "rtdetrv3": BASE / "data" / "engines" / "rtdetr" / "rtdetrv3_r18.engine",
    "dfine": BASE / "data" / "engines" / "dfine" / "dfine_n.engine",
    "deim": BASE / "data" / "engines" / "deim" / "deim_dfine_n.engine",
    "deimv2": BASE / "data" / "engines" / "deimv2" / "deimv2_n.engine",
    "rfdetr": BASE / "data" / "engines" / "rfdetr" / "rfdetr_n.engine",
}

RTDETR_ONNX_PATHS: dict[str, Path] = {
    "rtdetrv1": BASE / "data" / "rtdetr" / "rtdetrv1_r18.onnx",
    "rtdetrv2": BASE / "data" / "rtdetr" / "rtdetrv2_r18.onnx",
    "rtdetrv3": BASE / "data" / "rtdetr" / "rtdetrv3_r18.onnx",
    "dfine": BASE / "data" / "dfine" / "dfine_n.onnx",
    "deim": BASE / "data" / "deim" / "deim_dfine_n.onnx",
    "deimv2": BASE / "data" / "deimv2" / "deimv2_n.onnx",
    "rfdetr": BASE / "data" / "rfdetr" / "rfdetr_n.onnx",
}

# Shared test images
HORSE_IMAGE_PATH: str = str(BASE / "data" / "horse.jpg")
PEOPLE_IMAGE_PATH: str = str(BASE / "data" / "people.jpeg")
IMAGE_PATHS: list[str] = [
    HORSE_IMAGE_PATH,
    PEOPLE_IMAGE_PATH,
]

# Ground truth detections per image
GROUND_TRUTHS: list[int] = [1, 4]

