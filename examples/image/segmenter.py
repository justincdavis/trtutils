# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""File showcasing the YOLOv11Seg segmenter."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from trtutils import set_log_level
from trtutils.models import YOLOv11Seg

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def main() -> None:
    onnx_path = DATA_DIR / "yolov11n-seg" / "yolov11n-seg.onnx"
    engine_path = DATA_DIR / "yolov11n-seg" / "yolov11n-seg.engine"

    if not onnx_path.exists():
        print("Downloading YOLOv11-Seg ONNX model...")
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        YOLOv11Seg.download("yolov11n-seg", onnx_path)

    if not engine_path.exists():
        YOLOv11Seg.build(onnx_path, engine_path)

    image_path = DATA_DIR / "horse.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        msg = f"Could not read image: {image_path}"
        raise FileNotFoundError(msg)

    model = YOLOv11Seg(engine_path, warmup=True)

    t0 = time.perf_counter()
    segmentations = model.end2end(image)
    t1 = time.perf_counter()
    print(f"Inference time: {round((t1 - t0) * 1000.0, 2)} ms")
    print(f"Found {len(segmentations)} segmentation(s)")

    overlay = image.copy()
    rng = np.random.default_rng(0)
    for bbox, _score, class_id, mask in segmentations:
        x1, y1, x2, y2 = bbox
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        overlay[mask.astype(bool)] = color
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            str(class_id),
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0.0)

    output_path = DATA_DIR / "horse_segmentations.jpg"
    cv2.imwrite(str(output_path), blended)
    print(f"Saved annotated image to {output_path}")


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
