# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from threading import Thread

import numpy as np

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


def test_engine_run_no_pagelocked() -> None:
    """Test the engine runs when pagelocked memory is disabled."""
    engine_path = build_engine()

    engine = trtutils.TRTEngine(
        engine_path,
        warmup=False,
        pagelocked_mem=False,
    )

    outputs = engine.mock_execute()

    assert outputs is not None


def test_engine_parity_non_pagelocked() -> None:
    """Test that the same engine gets same results with/without pagelocked memory."""
    engine_path = build_engine()

    engine = trtutils.TRTEngine(
        engine_path,
        warmup=False,
        pagelocked_mem=True,
    )
    engine_no_pagelocked = trtutils.TRTEngine(
        engine_path,
        warmup=False,
        pagelocked_mem=False,
    )

    rand_input = engine.get_random_input()

    outputs = engine.execute(rand_input)
    outputs_no_pagelocked = engine_no_pagelocked.execute(rand_input)

    for out, out_no_page in zip(outputs, outputs_no_pagelocked):
        assert np.allclose(out, out_no_page)


def test_engine_pagelocked_performance() -> None:
    """Test that the engine runs faster with pagelocked memory."""
    engine_path = build_engine()

    engine = trtutils.TRTEngine(
        engine_path,
        warmup=True,
        warmup_iterations=10,
        pagelocked_mem=True,
    )

    engine_no_pagelocked = trtutils.TRTEngine(
        engine_path,
        warmup=True,
        warmup_iterations=10,
        pagelocked_mem=False,
    )

    rand_input = engine.get_random_input()

    pagelocked_times = []
    non_pagelocked_times = []

    for _ in range(NUM_ITERS * 5):
        t0 = time.time()
        engine.execute(rand_input)
        t1 = time.time()
        pagelocked_times.append(t1 - t0)

        t00 = time.time()
        engine_no_pagelocked.execute(rand_input)
        t11 = time.time()
        non_pagelocked_times.append(t11 - t00)

    pagelock_mean = np.mean(pagelocked_times)
    non_pagelock_mean = np.mean(non_pagelocked_times)
    speedup = non_pagelock_mean / pagelock_mean
    assert speedup > 1.0

    print(
        f"Pagelocked mean: {pagelock_mean}, Non-pagelocked mean: {non_pagelock_mean}, Speedup: {speedup}"
    )
