# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

import cv2
import trtutils

try:
    from paths import BASE, ENGINE_PATHS, ONNX_PATHS
except ModuleNotFoundError:
    from .paths import BASE, ENGINE_PATHS, ONNX_PATHS


def build_yolo_dla(version: int) -> Path:
    onnx_path = ONNX_PATHS[version]
    engine_path = ENGINE_PATHS[version]
    if engine_path.exists():
        return engine_path

    if version != 9:
        trtutils.trtexec.build_engine(
            onnx_path,
            engine_path,
            use_dla_core=0,
            allow_gpu_fallback=True,
        )
    else:
        trtutils.trtexec.build_engine(
            onnx_path,
            engine_path,
            use_dla_core=0,
            allow_gpu_fallback=True,
            shapes=[("images", (1, 3, 640, 640))],
        )

    return engine_path


def yolo_dla_results(version: int) -> None:
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

    engine_path = build_yolo_dla(version)

    scale = (0, 1) if version != 0 else (0, 255)
    trt_model = trtutils.impls.yolo.YOLO(engine_path, conf_thres=0.25, warmup=False, input_range=scale, preprocessor="cpu")

    for gt, ipath in zip(
        [1, 4],
        [
            str(BASE / "data" / "horse.jpg"),
            str(BASE / "data" / "people.jpeg"),
        ]
    ):
        image = cv2.imread(ipath)

        outputs = trt_model.run(image)
        trt_bboxes = [bbox for (bbox, _, _) in trt_model.get_detections(outputs)]

        assert len(trt_bboxes) > max(0, gt - 2)
        assert len(trt_bboxes) < gt + 2
        assert len(trt_bboxes) >= 1

    del trt_model


def test_yolo_dla_7_results():
    yolo_dla_results(7)


def test_yolo_dla_8_results():
    yolo_dla_results(8)


def test_yolo_dla_9_results():
    yolo_dla_results(9)


def test_yolo_dla_10_results():
    yolo_dla_results(10)


# cannot build yolox engine with trtexec
# def test_yolo_dla_x_results():
#     yolo_dla_results(0)


if __name__ == "__main__":
    yolo_dla_results(7)
