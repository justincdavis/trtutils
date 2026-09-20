# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
File showcasing the YOLOv11OBB oriented bounding box detector.

The pretrained OBB weights are trained on DOTA, an aerial imagery dataset,
so this example runs on an overhead photo. data/aerial.jpg is a public
domain photograph by Carol M. Highsmith (Library of Congress).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from trtutils import set_log_level
from trtutils.models import YOLOv11OBB

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def main() -> None:
    onnx_path = DATA_DIR / "yolov11n-obb" / "yolov11n-obb.onnx"
    engine_path = DATA_DIR / "yolov11n-obb" / "yolov11n-obb.engine"

    if not onnx_path.exists():
        print("Downloading YOLOv11-OBB ONNX model...")
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        YOLOv11OBB.download("yolov11n-obb", onnx_path)

    if not engine_path.exists():
        YOLOv11OBB.build(onnx_path, engine_path)

    image_path = DATA_DIR / "aerial.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        msg = f"Could not read image: {image_path}"
        raise FileNotFoundError(msg)

    model = YOLOv11OBB(engine_path, warmup=True)

    t0 = time.perf_counter()
    detections = model.end2end(image)
    t1 = time.perf_counter()
    print(f"Inference time: {round((t1 - t0) * 1000.0, 2)} ms")
    print(f"Found {len(detections)} detection(s)")

    for rbox, _score, _class_id in detections:
        cx, cy, w, h, angle = rbox
        rect = ((cx, cy), (w, h), np.degrees(angle))
        points = cv2.boxPoints(rect).astype(np.int32)
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    output_path = DATA_DIR / "aerial_obb.jpg"
    cv2.imwrite(str(output_path), image)
    print(f"Saved annotated image to {output_path}")


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
