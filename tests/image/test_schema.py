# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/_schema.py -- detector schema detection from engine names."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trtutils.image._schema import InputSchema, OutputSchema, get_detector_io_schema


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("inputs", "outputs", "expected"),
    [
        pytest.param(
            InputSchema.RT_DETR.value,
            ("scores", "boxes", "labels"),
            (OutputSchema.DETR, OutputSchema.DETR_LBS),
            id="detr-shuffled",
        ),
        pytest.param(
            InputSchema.RT_DETR.value,
            OutputSchema.DETR_LBS.value,
            (OutputSchema.DETR_LBS,),
            id="detr-lbs-exact",
        ),
        pytest.param(
            InputSchema.RT_DETR_V3.value,
            OutputSchema.RT_DETR_V3.value[::-1],
            (OutputSchema.RT_DETR_V3,),
            id="rtdetrv3-swapped",
        ),
        pytest.param(
            InputSchema.YOLO.value,
            OutputSchema.EFFICIENT_NMS.value[::-1],
            (OutputSchema.EFFICIENT_NMS,),
            id="efficient-nms-reversed",
        ),
    ],
)
def test_output_schema_ignores_output_order(inputs, outputs, expected) -> None:
    """TensorRT 8.6 reorders engine outputs; detection matches on names, not order."""
    engine = SimpleNamespace(input_names=list(inputs), output_names=list(outputs))
    _, output_schema = get_detector_io_schema(engine)
    assert output_schema in expected
