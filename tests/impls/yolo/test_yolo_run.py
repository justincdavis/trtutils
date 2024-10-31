# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path
from threading import Thread, Lock

import trtutils


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

ENGINE_LOCK = Lock()


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


def yolo_run(version: int) -> None:
    engine_path = build_yolo(version)

    engine = trtutils.impls.yolo.YOLO(
        engine_path,
        warmup=False,
    )

    outputs = engine.mock_run()

    assert outputs is not None

    del engine


def multiple_yolos_run(version: int) -> None:
    engine_path = build_yolo(version)

    engines = [
        trtutils.impls.yolo.YOLO(engine_path, warmup=False) for _ in range(4)
    ]

    outputs = [engine.mock_run() for engine in engines]

    for o in outputs:
        assert o is not None

    for engine in engines:
        del engine


def yolo_run_in_thread(version: int) -> None:
    result = [False]

    def run(result: list[bool]) -> None:
        engine_path = build_yolo(version)

        engine = trtutils.impls.yolo.YOLO(
            engine_path,
            warmup=False,
        )

        outputs = engine.mock_run()

        assert outputs is not None

        result[0] = True

        del engine

    thread = Thread(target=run, args=(result,), daemon=True)
    thread.start()

    thread.join()

    assert result[0]


def multiple_yolos_run_in_threads(version: int) -> None:
    num_engines = 2
    result = [0] * num_engines
    num_iters = 50

    def run(threadid: int, result: list[int], iters: int) -> None:
        engine_path = build_yolo(version)

        engine = trtutils.impls.yolo.YOLO(
            engine_path,
            warmup=False,
        )

        outputs = None
        succeses = 0
        for _ in range(iters):
            outputs = engine.mock_run()
            if outputs is not None and len(outputs) != 0:
                succeses += 1

        assert outputs is not None

        result[threadid] = succeses

        del engine

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


# YOLO V7
def test_yolo_7_run():
    yolo_run(7)

def test_yolo_7_run_thread():
    yolo_run_in_thread(7)

def test_yolo_7_multiple():
    multiple_yolos_run(7)

def test_yolo_7_multiple_threads():
    multiple_yolos_run_in_threads(7)


# YOLO V8
def test_yolo_8_run():
    yolo_run(8)

def test_yolo_8_run_thread():
    yolo_run_in_thread(8)

def test_yolo_8_multiple():
    multiple_yolos_run(8)

def test_yolo_8_multiple_threads():
    multiple_yolos_run_in_threads(8)


# YOLO V9
def test_yolo_9_run():
    yolo_run(9)

def test_yolo_9_run_thread():
    yolo_run_in_thread(9)

def test_yolo_9_multiple():
    multiple_yolos_run(9)

def test_yolo_9_multiple_threads():
    multiple_yolos_run_in_threads(9)


# YOLO V10
def test_yolo_10_run():
    yolo_run(10)

def test_yolo_10_run_thread():
    yolo_run_in_thread(10)

def test_yolo_10_multiple():
    multiple_yolos_run(10)

def test_yolo_10_multiple_threads():
    multiple_yolos_run_in_threads(10)
