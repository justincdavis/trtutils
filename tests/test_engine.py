# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path
from threading import Thread

import trtutils


ENGINE_PATH = engine_path = (
    Path(__file__).parent.parent / "data" / "engines" / "simple.engine"
)


def build_engine() -> Path:
    simple_path = Path(__file__).parent.parent / "data" / "simple.onnx"

    if ENGINE_PATH.exists():
        return ENGINE_PATH

    trtutils.trtexec.build_from_onnx(
        simple_path,
        ENGINE_PATH,
    )

    return ENGINE_PATH


def test_engine_run() -> None:
    engine_path = build_engine()

    engine = trtutils.TRTEngine(
        engine_path,
        warmup=False,
    )

    outputs = engine.mock_execute()

    assert outputs is not None


def test_multiple_engines_run() -> None:
    engine_path = build_engine()

    engines = [trtutils.TRTEngine(engine_path, warmup=False) for _ in range(4)]

    outputs = [engine.mock_execute() for engine in engines]

    for o in outputs:
        assert o is not None


def test_engine_run_in_thread() -> None:
    result = [False]

    def run(result: list[bool]) -> None:
        engine_path = build_engine()

        engine = trtutils.TRTEngine(
            engine_path,
            warmup=False,
        )

        outputs = engine.mock_execute()

        assert outputs is not None

        result[0] = True

    thread = Thread(target=run, args=(result,), daemon=True)
    thread.start()

    thread.join()

    assert result[0]


def test_multiple_engines_run_in_threads() -> None:
    num_engines = 4
    result = [0] * num_engines
    num_iters = 1_000

    def run(threadid: int, result: list[int], iters: int) -> None:
        engine_path = build_engine()

        engine = trtutils.TRTEngine(
            engine_path,
            warmup=False,
        )

        outputs = None
        succeses = 0
        for _ in range(iters):
            outputs = engine.mock_execute()
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
