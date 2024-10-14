# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path
from threading import Thread

import trtutils


ENGINE_PATH = engine_path = (
    Path(__file__).parent.parent.parent / "data" / "engines" / "yolov7t.engine"
)


def build_yolo() -> Path:
    simple_path = Path(__file__).parent.parent.parent / "data" / "yolov7t.onnx"

    if ENGINE_PATH.exists():
        return ENGINE_PATH

    trtutils.trtexec.build_from_onnx(
        simple_path,
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
