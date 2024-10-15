# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path
from threading import Thread

import cv2
import trtutils
import numpy as np
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


def test_yolo_run() -> None:
    engine_path = build_yolo()

    engine = trtutils.impls.yolo.YOLO(
        engine_path,
        version=7,
        warmup=False,
    )

    outputs = engine.mock_run()

    assert outputs is not None


def test_multiple_yolos_run() -> None:
    engine_path = build_yolo()

    engines = [
        trtutils.impls.yolo.YOLO(engine_path, version=7, warmup=False) for _ in range(4)
    ]

    outputs = [engine.mock_run() for engine in engines]

    for o in outputs:
        assert o is not None


def test_yolo_run_in_thread() -> None:
    result = [False]

    def run(result: list[bool]) -> None:
        engine_path = build_yolo()

        engine = trtutils.impls.yolo.YOLO(
            engine_path,
            version=7,
            warmup=False,
        )

        outputs = engine.mock_run()

        assert outputs is not None

        result[0] = True

    thread = Thread(target=run, args=(result,), daemon=True)
    thread.start()

    thread.join()

    assert result[0]


def test_multiple_yolos_run_in_threads() -> None:
    num_engines = 4
    result = [0] * num_engines
    num_iters = 1_000

    def run(threadid: int, result: list[int], iters: int) -> None:
        engine_path = build_yolo()

        engine = trtutils.impls.yolo.YOLO(
            engine_path,
            version=7,
            warmup=False,
        )

        outputs = None
        succeses = 0
        for _ in range(iters):
            outputs = engine.mock_run()
            if outputs is not None:
                succeses += 1

        assert outputs is not None

        result[threadid] = succeses

    threads = [
        Thread(target=run, args=(threadid, result, num_iters), daemon=True)
        for threadid in range(num_engines)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for r in result:
        assert r == num_iters


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
    image = cv2.resize(image, (160, 160)).astype(np.uint8)

    yolo_bboxes = get_bboxes(yolo_model(image, verbose=False))

    trt_model.run([image])
    trt_bboxes = [bbox for (bbox, _, _) in trt_model.get_detections()[0]]

    assert len(yolo_bboxes) == 1
    assert len(trt_bboxes) == 1
    assert bboxes_close(yolo_bboxes[0], trt_bboxes[0], tolerance=5)
