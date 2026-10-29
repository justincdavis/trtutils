# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/builder/onnx/_hands23.py -- NonZero removal for Hands23 ONNX files."""

from __future__ import annotations

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import helper

from tests.conftest import DATA_DIR
from trtutils.builder.onnx import fix_hands23_onnx, remove_hands23_nonzero

HANDS23_ONNX = DATA_DIR / "hands23" / "hands23_x101_800.onnx"
_IMGSZ = 800
_needs_onnx = pytest.mark.skipif(not HANDS23_ONNX.exists(), reason=f"missing {HANDS23_ONNX}")


def _count(model: onnx.ModelProto, op_type: str) -> int:
    return sum(n.op_type == op_type for n in model.graph.node)


def _io(model: onnx.ModelProto) -> list[tuple[str, list[int]]]:
    return [
        (v.name, [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim])
        for v in (*model.graph.input, *model.graph.output)
    ]


def _ort_runnable(model: onnx.ModelProto) -> bytes:
    """Swap ROIAlign_TRT for the ONNX RoiAlign (each node keeps its own spatial_scale)."""
    runnable = onnx.ModelProto()
    runnable.CopyFrom(model)
    nodes = list(runnable.graph.node)
    del runnable.graph.node[:]
    for node in nodes:
        if node.op_type == "ROIAlign_TRT":
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            cast = f"{node.name}_i64"
            runnable.graph.node.append(
                helper.make_node("Cast", [node.input[2]], [cast], to=7, name=cast)
            )
            node.input[2] = cast
            node.op_type = "RoiAlign"
            node.domain = ""
            del node.attribute[:]
            node.attribute.extend(
                [
                    helper.make_attribute("mode", "avg" if attrs["mode"] == 1 else "max"),
                    helper.make_attribute("output_height", attrs["output_height"]),
                    helper.make_attribute("output_width", attrs["output_width"]),
                    helper.make_attribute("sampling_ratio", attrs["sampling_ratio"]),
                    helper.make_attribute("spatial_scale", attrs["spatial_scale"]),
                    helper.make_attribute(
                        "coordinate_transformation_mode",
                        "half_pixel"
                        if attrs["coordinate_transformation_mode"] == 1
                        else "output_half_pixel",
                    ),
                ]
            )
        runnable.graph.node.append(node)
    runnable.opset_import[0].version = max(runnable.opset_import[0].version, 16)
    return runnable.SerializeToString()


def _unrelated_nonzero_model() -> onnx.ModelProto:
    """A graph with a NonZero that is not the Hands23 pattern."""
    graph = helper.make_graph(
        [helper.make_node("NonZero", ["x"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.INT64, [1, None])],
    )
    return helper.make_model(graph)


@pytest.fixture(scope="module")
def original() -> onnx.ModelProto:
    """The exported Hands23 model."""
    if not HANDS23_ONNX.exists():
        pytest.skip(f"missing {HANDS23_ONNX}")
    return onnx.load(str(HANDS23_ONNX))


@pytest.fixture(scope="module")
def passed(original) -> onnx.ModelProto:
    """The Hands23 model after the NonZero pass."""
    model = onnx.ModelProto()
    model.CopyFrom(original)
    assert remove_hands23_nonzero(model)
    return model


@pytest.mark.cpu
@_needs_onnx
def test_removes_nonzero_keeps_io(original, passed) -> None:
    """The pass removes every NonZero, keeps inputs/outputs and the remaining NMS/ROIAlign ops."""
    assert _count(original, "NonZero") == 9
    assert _count(passed, "NonZero") == 0
    assert _io(passed) == _io(original)
    for op_type in ("NonMaxSuppression", "ROIAlign_TRT"):
        assert _count(passed, op_type) == _count(original, op_type)
    assert _count(passed, "ScatterND") == 0
    assert _count(passed, "GatherND") == 0
    onnx.checker.check_model(passed, full_check=False)


@pytest.mark.cpu
@_needs_onnx
def test_idempotent(passed) -> None:
    """A second pass reports no change and leaves the graph untouched."""
    again = onnx.ModelProto()
    again.CopyFrom(passed)
    assert not remove_hands23_nonzero(again)
    assert again.SerializeToString() == passed.SerializeToString()


@pytest.mark.cpu
@pytest.mark.parametrize(
    "make_model",
    [
        pytest.param(lambda: onnx.load(str(DATA_DIR / "simple.onnx")), id="no-nonzero"),
        pytest.param(_unrelated_nonzero_model, id="foreign-nonzero"),
    ],
)
def test_non_hands23_untouched(make_model, tmp_path) -> None:
    """A model that is not the Hands23 pattern is left as is, in memory and on disk."""
    model = make_model()
    before = model.SerializeToString()
    assert not remove_hands23_nonzero(model)
    assert model.SerializeToString() == before

    path = tmp_path / "in.onnx"
    onnx.save(model, str(path))
    assert not fix_hands23_onnx(path, tmp_path / "out.onnx")
    assert not (tmp_path / "out.onnx").exists()


@pytest.mark.cpu
@_needs_onnx
def test_fix_file(passed, tmp_path) -> None:
    """fix_hands23_onnx writes the rewritten model to output, or over the input by default."""
    out = tmp_path / "out.onnx"
    assert fix_hands23_onnx(HANDS23_ONNX, out)
    assert _count(onnx.load(str(out)), "NonZero") == 0
    assert not fix_hands23_onnx(out)


@_needs_onnx
@pytest.mark.parametrize("image", ["people", "horse"])
def test_ort_outputs_identical(original, passed, images, image) -> None:
    """ORT outputs of the original and rewritten graphs are bit-identical."""
    frame = cv2.cvtColor(cv2.resize(images[image].array, (_IMGSZ, _IMGSZ)), cv2.COLOR_BGR2RGB)
    tensor = frame.transpose(2, 0, 1)[None].astype(np.float32)
    outs = []
    for model in (original, passed):
        session = ort.InferenceSession(_ort_runnable(model), providers=["CPUExecutionProvider"])
        outs.append(session.run(None, {"input": tensor}))
    for ref, new in zip(*outs):
        np.testing.assert_array_equal(ref, new)
