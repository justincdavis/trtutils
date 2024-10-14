# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path
from threading import Thread

import trtutils


ENGINE_PATH = engine_path = Path(__file__).parent / "simple.engine"


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
