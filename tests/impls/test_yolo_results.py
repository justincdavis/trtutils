# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

import cv2
import trtutils
from ultralytics import YOLO


ENGINE_PATH = engine_path = (
    Path(__file__).parent.parent.parent / "data" / "engines" / "trt_yolov8n.engine"
)


def build_yolo() -> Path:
    onnx_path = Path(__file__).parent.parent.parent / "data" / "trt_yolov8n.onnx"

    if ENGINE_PATH.exists():
        return ENGINE_PATH

    trtutils.trtexec.build_from_onnx(
        onnx_path,
        ENGINE_PATH,
    )

    return ENGINE_PATH


def test_yolo_results() -> None:
    def get_bboxes(detections) -> list[tuple[int, int, int, int]]:
        try:
            detections = detections.cpu().numpy()
        except AttributeError:
            detections = detections[0].cpu().numpy()

        # convert the PredBBox class to a tuple[int, int, int, int]
        bboxes = []
        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bboxes.append((x1, y1, x2, y2))

        return bboxes

    def bboxes_close(bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int], tolerance = 2) -> bool:
        for c1, c2 in zip(bbox1, bbox2):
            if abs(c1 - c2) > tolerance:
                return False
        return True

    engine_path = build_yolo()

    ultralytics_path = Path(__file__).parent.parent.parent / "data" / "ultralytics_yolov8n.pt"

    yolo_model = YOLO(ultralytics_path, task="detect", verbose=False)

    trt_model = trtutils.impls.yolo.YOLO(engine_path, version=7, warmup=False)

    image = cv2.imread(str(Path(__file__).parent.parent.parent / "data" / "horse.jpg"))

    yolo_bboxes = get_bboxes(yolo_model(image, verbose=False))

    trt_model.run([image])
    trt_bboxes = [bbox for (bbox, _, _) in trt_model.get_detections()]

    assert len(yolo_bboxes) == 1
    assert len(trt_bboxes) == 1
    assert bboxes_close(yolo_bboxes[0], trt_bboxes[0], tolerance=5)


if __name__ == "__main__":
    test_yolo_results()
