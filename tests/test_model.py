# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from threading import Thread
from typing import TYPE_CHECKING

import trtutils

from .common import build_engine

if TYPE_CHECKING:
    import numpy as np

NUM_MODELS = 4
NUM_ITERS = 1_000


def test_model_load_no_args() -> None:
    """Test TRTModel initializes with only engine path arg."""
    engine_path = build_engine()

    model = trtutils.TRTModel(engine_path)

    assert model is not None


def test_model_load_args() -> None:
    """Test TRTModel initializes with pre/post process funcs."""

    def _identity(data: list[np.ndarray]) -> list[np.ndarray]:
        return data

    engine_path = build_engine()

    model = trtutils.TRTModel(engine_path, _identity, _identity)

    assert model is not None


def test_model_run() -> None:
    """Test basic model execution with mock data."""
    engine_path = build_engine()

    model = trtutils.TRTModel(
        engine_path,
        warmup=False,
    )

    outputs = model.mock_run()

    assert outputs is not None


def test_multiple_models_run() -> None:
    """Test running multiple models simultaneously."""
    engine_path = build_engine()

    models = [trtutils.TRTModel(engine_path, warmup=False) for _ in range(NUM_MODELS)]

    outputs = [model.mock_run() for model in models]

    for o in outputs:
        assert o is not None


def test_model_run_in_thread() -> None:
    """Test model execution in a separate thread."""
    result = [False]

    def run(result: list[bool]) -> None:
        engine_path = build_engine()

        model = trtutils.TRTModel(
            engine_path,
            warmup=False,
        )

        outputs = model.mock_run()

        assert outputs is not None

        result[0] = True

    thread = Thread(target=run, args=(result,), daemon=True)
    thread.start()

    thread.join()

    assert result[0]


def test_multiple_models_run_in_threads() -> None:
    """Test running multiple models in separate threads with multiple iterations."""
    result = [0] * NUM_MODELS

    def run(threadid: int, result: list[int], iters: int) -> None:
        engine_path = build_engine()

        model = trtutils.TRTModel(
            engine_path,
            warmup=False,
        )

        outputs = None
        succeses = 0
        for _ in range(iters):
            outputs = model.mock_run()
            if outputs is not None:
                succeses += 1
        assert outputs is not None
        result[threadid] = succeses
        del model

    threads = [
        Thread(target=run, args=(threadid, result, NUM_ITERS), daemon=True)
        for threadid in range(NUM_MODELS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for r in result:
        assert r == NUM_ITERS
