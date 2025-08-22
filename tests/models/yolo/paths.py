# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

BASE = Path(__file__).parent.parent.parent.parent
ENGINE_PATHS: dict[int, Path] = {
    7: BASE / "data" / "engines" / "yolov7" / "yolov7t.engine",
    8: BASE / "data" / "engines" / "yolov8" / "yolov8n.engine",
    9: BASE / "data" / "engines" / "yolov9" / "yolov9t.engine",
    10: BASE / "data" / "engines" / "yolov10" / "yolov10n.engine",
    11: BASE / "data" / "engines" / "yolov11" / "yolov11n.engine",
    12: BASE / "data" / "engines" / "yolov12" / "yolov12n.engine",
    0: BASE / "data" / "engines" / "yolox" / "yoloxn.engine",
}
ONNX_PATHS: dict[int, Path] = {
    7: BASE / "data" / "yolov7" / "yolov7t.onnx",
    8: BASE / "data" / "yolov8" / "yolov8n.onnx",
    9: BASE / "data" / "yolov9" / "yolov9t.onnx",
    10: BASE / "data" / "yolov10" / "yolov10n.onnx",
    11: BASE / "data" / "yolov11" / "yolov11n.onnx",
    12: BASE / "data" / "yolov12" / "yolov12n.onnx",
    0: BASE / "data" / "yolox" / "yoloxn.onnx",
}
HORSE_IMAGE_PATH: str = str(BASE / "data" / "horse.jpg")
PEOPLE_IMAGE_PATH: str = str(BASE / "data" / "people.jpeg")
IMAGE_PATHS: list[str] = [
    HORSE_IMAGE_PATH,
    PEOPLE_IMAGE_PATH,
]
