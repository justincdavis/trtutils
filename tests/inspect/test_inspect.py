# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import trtutils
from tests.common import build_engine
from tests.impls.yolo.common import build_yolo


def test_inspect_simple() -> None:
    """Test the inspect engine function on a simple engine."""
    engine_path = build_engine()

    size, batch_size, inputs, outputs = trtutils.inspect_engine(engine_path)

    # size and batch_size are weird for simple.engine
    assert size == 0
    assert batch_size == 1
    assert len(inputs) > 0
    for itensor in inputs:
        assert itensor[1] == (160, 160)
    assert len(outputs) > 0


def inspect_yolo(version: int) -> None:
    """Test the inspect engine function on YOLO."""
    engine_path = build_yolo(version)

    size, batch_size, inputs, outputs = trtutils.inspect_engine(engine_path)

    assert size >= 1
    assert batch_size == 1
    assert len(inputs) > 0
    for itensor in inputs:
        assert itensor[1] == (1, 3, 640, 640)
    assert len(outputs) > 0


def test_inspect_yolov10() -> None:
    """Test the inspect engine functio on YOLOv10."""
    inspect_yolo(10)


def test_inspect_yolov9() -> None:
    """Test the inspect engine functio on YOLOv9."""
    inspect_yolo(9)


def test_inspect_yolov8() -> None:
    """Test the inspect engine functio on YOLOv8."""
    inspect_yolo(8)


def test_inspect_yolov7() -> None:
    """Test the inspect engine functio on YOLOv7."""
    inspect_yolo(7)
