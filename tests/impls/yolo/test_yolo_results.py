# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

import cv2
import trtutils
from ultralytics import YOLO


_BASE = Path(__file__).parent.parent.parent.parent
ENGINE_PATHS: dict[int, Path] = {
    7: _BASE / "data" / "engines" / "trt_yolov7t.engine",
    8: _BASE / "data" / "engines" / "trt_yolov8n.engine",
    9: _BASE / "data" / "engines" / "trt_yolov9t.engine",
    10: _BASE / "data" / "engines" / "trt_yolov10n.engine",
}
ONNX_PATHS: dict[int, Path] = {
    7: _BASE / "data" / "trt_yolov7t.onnx",
    8: _BASE / "data" / "trt_yolov8n.onnx",
    9: _BASE / "data" / "trt_yolov9t.onnx",
    10: _BASE / "data" / "trt_yolov10n.onnx"
}

def build_yolo(version: int) -> Path:
    onnx_path = ONNX_PATHS[version]
    engine_path = ENGINE_PATHS[version]
    if engine_path.exists():
        return engine_path

    if version != 9:
        trtutils.trtexec.build_engine(
            onnx_path,
            engine_path,
        )
    else:
        trtutils.trtexec.build_engine(
            onnx_path,
            engine_path,
            shapes=[("images", (1, 3, 640, 640))],
        )

    return engine_path


def yolo_results(version: int) -> None:
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

    def bboxes_close(
        bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int], tolerance=2
    ) -> bool:
        for c1, c2 in zip(bbox1, bbox2):
            if abs(c1 - c2) > tolerance:
                return False
        return True

    engine_path = build_yolo(version)

    trt_model = trtutils.impls.yolo.YOLO(engine_path, version=version, warmup=False)

    for gt, ipath in zip(
        [1, 3],
        [
            str(_BASE / "data" / "horse.jpg"),
            str(_BASE / "data" / "people.jpeg"),
        ]
    ):
        image = cv2.imread(ipath)

        outputs = trt_model.run(image)
        trt_bboxes = [bbox for (bbox, _, _) in trt_model.get_detections(outputs)]

        assert len(trt_bboxes) > max(0, gt - 2)
        assert len(trt_bboxes) < gt + 2


def test_yolo_7_results():
    yolo_results(7)


def test_yolo_8_results():
    yolo_results(8)


def test_yolo_9_results():
    yolo_results(9)


def test_yolo_10_results():
    yolo_results(10)


if __name__ == "__main__":
    yolo_results(7)