# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path


BASE = Path(__file__).parent.parent.parent.parent
ENGINE_PATHS: dict[int, Path] = {
    7: BASE / "data" / "engines" / "trt_yolov7t.engine",
    8: BASE / "data" / "engines" / "trt_yolov8n.engine",
    9: BASE / "data" / "engines" / "trt_yolov9t.engine",
    10: BASE / "data" / "engines" / "trt_yolov10n.engine",
    0: BASE / "data" / "engines" / "trt_yoloxn.engine",
}
ONNX_PATHS: dict[int, Path] = {
    7: BASE / "data" / "trt_yolov7t.onnx",
    8: BASE / "data" / "trt_yolov8n.onnx",
    9: BASE / "data" / "trt_yolov9t.onnx",
    10: BASE / "data" / "trt_yolov10n.onnx",
    0: BASE / "data" / "trt_yoloxn.onnx",
}
