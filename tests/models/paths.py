# Copyright (c) 2024-2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

BASE = Path(__file__).parent.parent.parent
# YOLO paths (version-based: 0=YOLOX, 3=YOLOv3, 5=YOLOv5, etc.)
YOLO_ENGINE_PATHS: dict[int, Path] = {
    0: BASE / "data" / "engines" / "yolox" / "yoloxn.engine",
    3: BASE / "data" / "engines" / "yolov3" / "yolov3.engine",
    5: BASE / "data" / "engines" / "yolov5" / "yolov5n.engine",
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
    3: BASE / "data" / "yolov3" / "yolov3.onnx",
    5: BASE / "data" / "yolov5" / "yolov5n.onnx",
    7: BASE / "data" / "yolov7" / "yolov7t.onnx",
    8: BASE / "data" / "yolov8" / "yolov8n.onnx",
    9: BASE / "data" / "yolov9" / "yolov9t.onnx",
    10: BASE / "data" / "yolov10" / "yolov10n.onnx",
    11: BASE / "data" / "yolov11" / "yolov11n.onnx",
    12: BASE / "data" / "yolov12" / "yolov12n.onnx",
    13: BASE / "data" / "yolov13" / "yolov13n.onnx",
}

# DETR paths (name-based)
DETR_ENGINE_PATHS: dict[str, Path] = {
    "rtdetrv1": BASE / "data" / "engines" / "rtdetrv1" / "rtdetrv1_r18.engine",
    "rtdetrv2": BASE / "data" / "engines" / "rtdetrv2" / "rtdetrv2_r18.engine",
    "rtdetrv3": BASE / "data" / "engines" / "rtdetrv3" / "rtdetrv3_r18.engine",
    "dfine": BASE / "data" / "engines" / "dfine" / "dfine_n.engine",
    "deim": BASE / "data" / "engines" / "deim" / "deim_dfine_n.engine",
    "deimv2": BASE / "data" / "engines" / "deimv2" / "deimv2_atto.engine",
    "rfdetr": BASE / "data" / "engines" / "rfdetr" / "rfdetr_n.engine",
}
DETR_ONNX_PATHS: dict[str, Path] = {
    "rtdetrv1": BASE / "data" / "rtdetrv1" / "rtdetrv1_r18.onnx",
    "rtdetrv2": BASE / "data" / "rtdetrv2" / "rtdetrv2_r18.onnx",
    "rtdetrv3": BASE / "data" / "rtdetrv3" / "rtdetrv3_r18.onnx",
    "dfine": BASE / "data" / "dfine" / "dfine_n.onnx",
    "deim": BASE / "data" / "deim" / "deim_dfine_n.onnx",
    "deimv2": BASE / "data" / "deimv2" / "deimv2_atto.onnx",
    "rfdetr": BASE / "data" / "rfdetr" / "rfdetr_n.onnx",
}

# Backward compatibility aliases
ENGINE_PATHS = YOLO_ENGINE_PATHS
ONNX_PATHS = YOLO_ONNX_PATHS

# Image paths
HORSE_IMAGE_PATH: str = str(BASE / "data" / "horse.jpg")
PEOPLE_IMAGE_PATH: str = str(BASE / "data" / "people.jpeg")
IMAGE_PATHS: list[str] = [
    HORSE_IMAGE_PATH,
    PEOPLE_IMAGE_PATH,
]
