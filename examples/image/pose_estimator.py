# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""File showcasing the YOLOv11Pose pose estimator."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from trtutils import set_log_level
from trtutils.models import YOLOv11Pose

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# COCO-17 skeleton edges, only used for drawing when the model predicts 17 keypoints
_SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def main() -> None:
    onnx_path = DATA_DIR / "yolov11n-pose" / "yolov11n-pose.onnx"
    engine_path = DATA_DIR / "yolov11n-pose" / "yolov11n-pose.engine"

    if not onnx_path.exists():
        print("Downloading YOLOv11-Pose ONNX model...")
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        YOLOv11Pose.download("yolov11n-pose", onnx_path)

    if not engine_path.exists():
        YOLOv11Pose.build(onnx_path, engine_path)

    image_path = DATA_DIR / "people.jpeg"
    image = cv2.imread(str(image_path))
    if image is None:
        msg = f"Could not read image: {image_path}"
        raise FileNotFoundError(msg)

    model = YOLOv11Pose(engine_path, warmup=True)

    t0 = time.perf_counter()
    poses = model.end2end(image)
    t1 = time.perf_counter()
    print(f"Inference time: {round((t1 - t0) * 1000.0, 2)} ms")
    print(f"Found {len(poses)} pose(s)")

    for bbox, _score, keypoints in poses:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        points = [(int(x), int(y), vis) for x, y, vis in keypoints]
        for x, y, vis in points:
            if vis > 0.5:  # noqa: PLR2004
                cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

        if len(points) == len(_SKELETON) + 1 or len(points) >= 17:  # noqa: PLR2004
            for i, j in _SKELETON:
                if points[i][2] > 0.5 and points[j][2] > 0.5:  # noqa: PLR2004
                    cv2.line(image, points[i][:2], points[j][:2], (255, 0, 0), 2)

    output_path = DATA_DIR / "people_poses.jpg"
    cv2.imwrite(str(output_path), image)
    print(f"Saved annotated image to {output_path}")


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
