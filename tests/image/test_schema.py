# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/_schema.py -- detector schema detection from engine names and the task output schemas."""

from __future__ import annotations

from types import SimpleNamespace

import onnx
import pytest

from trtutils.image import Classifier, DepthEstimator, ImageModel, TextDetector, TextRecognizer
from trtutils.image._schema import (
    ClassifierOutputSchema,
    DepthEstimatorOutputSchema,
    HandInteractionOutputSchema,
    InputSchema,
    OBBDetectorOutputSchema,
    OutputSchema,
    PoseEstimatorOutputSchema,
    SegmenterOutputSchema,
    TextDetectorOutputSchema,
    TextRecognizerOutputSchema,
    resolve_schema,
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
            (OutputSchema.DETR,),
            id="detr-lbs-names-resolve-to-detr",
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
    """TensorRT 8.6 reorders engine outputs; resolution matches on names, not order."""
    engine = SimpleNamespace(input_names=list(inputs), output_names=list(outputs))
    assert resolve_schema(engine, OutputSchema) in expected


def _engine(*output_names: str) -> SimpleNamespace:
    return SimpleNamespace(output_names=list(output_names))


_HOI = HandInteractionOutputSchema
_CLS = ClassifierOutputSchema
_DEPTH = DepthEstimatorOutputSchema
_TDET = TextDetectorOutputSchema
_TREC = TextRecognizerOutputSchema


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
        pytest.param(_CLS, ("output",), None, "TORCHVISION", id="torchvision-auto"),
        pytest.param(_CLS, ("logits",), None, "GENERIC", id="cls-generic-fallback"),
        pytest.param(_CLS, ("output0",), "GENERIC", "GENERIC", id="cls-generic-str"),
        pytest.param(_CLS, ("output",), _CLS.GENERIC, "GENERIC", id="cls-generic-forced"),
        pytest.param(_CLS, ("output", "aux"), None, "GENERIC", id="cls-generic-two-outputs"),
        pytest.param(_CLS, _HOI.HOI_DETR.value, None, "GENERIC", id="cls-generic-detector"),
        pytest.param(_DEPTH, ("output",), None, "DEPTH_ANYTHING", id="depth-anything-auto"),
        pytest.param(_DEPTH, ("depth",), None, "GENERIC", id="depth-generic-fallback"),
        pytest.param(_DEPTH, ("depth",), _DEPTH.GENERIC, "GENERIC", id="depth-generic-enum"),
        pytest.param(_DEPTH, ("depth", "conf"), None, "GENERIC", id="depth-generic-two-outputs"),
        pytest.param(_TDET, ("sigmoid_0.tmp_0",), None, "PADDLE", id="text-det-paddle-auto"),
        pytest.param(_TDET, ("maps",), None, "GENERIC", id="text-det-generic-fallback"),
        pytest.param(_TDET, ("sigmoid_0.tmp_0",), "GENERIC", "GENERIC", id="text-det-generic-str"),
        pytest.param(_TREC, ("softmax_11.tmp_0",), None, "PADDLE", id="text-rec-paddle-auto"),
        pytest.param(_TREC, ("logits",), None, "GENERIC", id="text-rec-generic-fallback"),
        pytest.param(_TREC, ("softmax_11.tmp_0",), _TREC.PADDLE, "PADDLE", id="text-rec-enum"),
        pytest.param(SegmenterOutputSchema, ("output0", "output1"), None, "YOLO", id="seg-auto"),
        pytest.param(
            SegmenterOutputSchema, ("output1", "output0"), None, "YOLO", id="seg-order-swapped"
        ),
        pytest.param(PoseEstimatorOutputSchema, ("output0",), None, "YOLO", id="pose-auto"),
        pytest.param(OBBDetectorOutputSchema, ("output0",), None, "YOLO", id="obb-auto"),
        pytest.param(
            PoseEstimatorOutputSchema,
            ("output0",),
            PoseEstimatorOutputSchema.YOLO,
            "YOLO",
            id="pose-enum-override",
        ),
        pytest.param(OBBDetectorOutputSchema, ("output0",), "YOLO", "YOLO", id="obb-str-override"),
    ],
)
def test_resolve_schema(schema_cls, outputs, override, expected) -> None:
    """Auto-detection and overrides resolve to the schema member, whatever the output order."""
    assert resolve_schema(_engine(*outputs), schema_cls, override) is schema_cls[expected]


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("schema_cls", "outputs", "override", "match"),
    [
        pytest.param(_HOI, ("output0",), None, "match no", id="hoi-on-detector"),
        pytest.param(_HOI, ("boxes", "scores", "labels"), None, "match no", id="missing-pairs"),
        pytest.param(_HOI, _HOI.HOI_DETR.value, _HOI.HANDS23, "match no", id="enum"),
        pytest.param(_HOI, _HOI.HOI_DETR.value, "NOPE", "NOPE", id="bad-str"),
        pytest.param(_CLS, ("logits",), _CLS.TORCHVISION, "match no", id="cls-named-override"),
        pytest.param(_DEPTH, ("depth",), "DEPTH_ANYTHING", "match no", id="depth-named-override"),
        pytest.param(_TDET, ("maps",), "PADDLE", "match no", id="text-det-named-override"),
        pytest.param(_TREC, ("logits",), _TREC.PADDLE, "match no", id="text-rec-named-override"),
        pytest.param(
            PoseEstimatorOutputSchema, ("output0", "output1"), None, "match no", id="pose-on-seg"
        ),
        pytest.param(SegmenterOutputSchema, ("output0",), None, "match no", id="seg-on-pose"),
        pytest.param(
            SegmenterOutputSchema, ("output0",), SegmenterOutputSchema.YOLO, "match no", id="enum"
        ),
        pytest.param(SegmenterOutputSchema, ("output0", "output1"), "NOPE", "NOPE", id="bad-str"),
    ],
)
def test_resolve_schema_rejects(schema_cls, outputs, override, match) -> None:
    """A wrong engine or schema name raises, listing the valid options."""
    with pytest.raises(ValueError, match=match) as exc:
        resolve_schema(_engine(*outputs), schema_cls, override)
    assert any(name in str(exc.value) for name in schema_cls.__members__)


@pytest.mark.cpu
def test_output_order_maps_by_name() -> None:
    """Swapped engine outputs are put back in schema order."""
    model = SimpleNamespace(_engine=_engine("side", "pair_probs", "labels", "scores", "boxes"))
    ImageModel._set_output_order(model, _HOI.HANDS23.value)  # ty: ignore[invalid-argument-type]
    reordered = ImageModel._order_outputs(model, ["s", "p", "l", "c", "b"])  # ty: ignore[invalid-argument-type]
    assert reordered == ["b", "c", "l", "p", "s"]


@pytest.fixture(scope="session")
def renamed_simple_engine(build_test_engine, simple_onnx_path, tmp_path_factory):
    """Build simple.onnx with its single output renamed to a name no schema lists."""
    model = onnx.load(str(simple_onnx_path))
    model.graph.node[-1].output[0] = "custom_head"
    model.graph.output[0].name = "custom_head"
    onnx_path = tmp_path_factory.mktemp("renamed") / "simple_custom_head.onnx"
    onnx.save(model, str(onnx_path))
    return build_test_engine(onnx_path)


# simple.onnx outputs (1, 3, 160, 160): the recognizer reads 160 classes (blank + 158 + space)
_REC_KWARGS = {"character_dict": [str(i) for i in range(158)]}


@pytest.mark.parametrize(
    ("model_cls", "kwargs", "expected"),
    [
        pytest.param(Classifier, {}, _CLS.GENERIC, id="classifier"),
        pytest.param(DepthEstimator, {}, _DEPTH.GENERIC, id="depth-estimator"),
        pytest.param(TextDetector, {}, _TDET.GENERIC, id="text-detector"),
        pytest.param(TextRecognizer, _REC_KWARGS, _TREC.GENERIC, id="text-recognizer"),
    ],
)
def test_generic_fallback_engine(renamed_simple_engine, model_cls, kwargs, expected) -> None:
    """An engine with an arbitrary output name loads under GENERIC and rejects a named override."""
    model = model_cls(renamed_simple_engine, preprocessor="cpu", warmup=False, **kwargs)
    assert model.output_schema is expected
    named = next(m for m in type(expected) if m is not expected)
    with pytest.raises(ValueError, match="match no"):
        model_cls(
            renamed_simple_engine, preprocessor="cpu", warmup=False, output_schema=named, **kwargs
        )


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        pytest.param(("images",), InputSchema.YOLO, id="yolo"),
        pytest.param(("input",), InputSchema.RF_DETR, id="rf-detr"),
        pytest.param(("orig_target_sizes", "images"), InputSchema.RT_DETR, id="rt-detr-reordered"),
        pytest.param(
            ("scale_factor", "image", "im_shape"), InputSchema.RT_DETR_V3, id="rtdetrv3-reordered"
        ),
    ],
)
def test_resolve_input_schema(inputs, expected) -> None:
    """Input names are matched ignoring order, like the outputs."""
    engine = SimpleNamespace(input_names=list(inputs))
    assert resolve_schema(engine, InputSchema, inputs=True) is expected


@pytest.mark.cpu
def test_resolve_input_schema_rejects_unknown_names() -> None:
    """Unknown input names raise instead of being guessed from the input count."""
    engine = SimpleNamespace(input_names=["x"])
    with pytest.raises(ValueError, match="match no InputSchema"):
        resolve_schema(engine, InputSchema, inputs=True)
    with pytest.raises(ValueError, match="Invalid input_schema"):
        resolve_schema(engine, InputSchema, "NOPE", inputs=True)
