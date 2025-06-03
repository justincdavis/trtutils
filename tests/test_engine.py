# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from threading import Thread

import trtutils

from .common import build_engine

NUM_ENGINES = 4
NUM_ITERS = 1_000


def test_engine_run() -> None:
    """Test basic engine execution with mock data."""
    engine_path = build_engine()

    engine = trtutils.TRTEngine(
        engine_path,
        warmup=False,
    )

    outputs = engine.mock_execute()

    assert outputs is not None


def test_multiple_engines_run() -> None:
    """Test running multiple engines simultaneously."""
    engine_path = build_engine()

    engines = [
        trtutils.TRTEngine(engine_path, warmup=False) for _ in range(NUM_ENGINES)
    ]

    outputs = [engine.mock_execute() for engine in engines]

    for o in outputs:
        assert o is not None


def test_engine_run_in_thread() -> None:
    """Test engine execution in a separate thread."""
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
    """Test running multiple engines in separate threads with multiple iterations."""
    result = [0] * NUM_ENGINES

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
        del engine

    threads = [
        Thread(target=run, args=(threadid, result, NUM_ITERS), daemon=True)
        for threadid in range(NUM_ENGINES)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for r in result:
        assert r == NUM_ITERS
