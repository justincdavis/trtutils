# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/_schema.py -- detector schema detection from engine names and the task output schemas."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trtutils.image import ImageModel
from trtutils.image._schema import (
    HandInteractionOutputSchema,
    InputSchema,
    OutputSchema,
    get_detector_io_schema,
    resolve_output_schema,
)


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


def _engine(*output_names: str) -> SimpleNamespace:
    return SimpleNamespace(output_names=list(output_names))


_HOI = HandInteractionOutputSchema


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("schema_cls", "outputs", "override", "expected"),
    [
        pytest.param(_HOI, _HOI.HOI_DETR.value, None, "HOI_DETR", id="hoi-detr-auto"),
        pytest.param(_HOI, _HOI.HANDS23.value, None, "HANDS23", id="hands23-auto"),
        pytest.param(_HOI, _HOI.HANDS23.value[::-1], None, "HANDS23", id="hands23-order-reversed"),
        pytest.param(
            _HOI, _HOI.HOI_DETR.value, _HOI.HOI_DETR, "HOI_DETR", id="hoi-detr-enum-override"
        ),
        pytest.param(_HOI, _HOI.HANDS23.value, "HANDS23", "HANDS23", id="hands23-str-override"),
    ],
)
def test_resolve_output_schema(schema_cls, outputs, override, expected) -> None:
    """Auto-detection and overrides resolve to the schema member, whatever the output order."""
    assert resolve_output_schema(_engine(*outputs), schema_cls, override) is schema_cls[expected]


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("schema_cls", "outputs", "override", "match"),
    [
        pytest.param(_HOI, ("output0",), None, "match no", id="hoi-on-detector"),
        pytest.param(_HOI, ("boxes", "scores", "labels"), None, "match no", id="missing-pairs"),
        pytest.param(_HOI, _HOI.HOI_DETR.value, _HOI.HANDS23, "match no", id="enum"),
        pytest.param(_HOI, _HOI.HOI_DETR.value, "NOPE", "NOPE", id="bad-str"),
    ],
)
def test_resolve_output_schema_rejects(schema_cls, outputs, override, match) -> None:
    """A wrong engine or schema name raises, listing the valid options."""
    with pytest.raises(ValueError, match=match) as exc:
        resolve_output_schema(_engine(*outputs), schema_cls, override)
    assert "HANDS23" in str(exc.value)


@pytest.mark.cpu
def test_output_order_maps_by_name() -> None:
    """Swapped engine outputs are put back in schema order."""
    model = SimpleNamespace(_engine=_engine("side", "pair_probs", "labels", "scores", "boxes"))
    ImageModel._set_output_order(model, _HOI.HANDS23.value)  # ty: ignore[invalid-argument-type]
    reordered = ImageModel._order_outputs(model, ["s", "p", "l", "c", "b"])  # ty: ignore[invalid-argument-type]
    assert reordered == ["b", "c", "l", "p", "s"]
