# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/download/_models/_ppocr_pass.py -- FP16 neck rescale for PP-OCRv5 server det."""

from __future__ import annotations

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from tests.conftest import DATA_DIR
from trtutils.download._models._ppocr_pass import scale_ppocrv5_server_det_neck

SERVER_DET_ONNX = DATA_DIR / "ppocr" / "ppocrv5_server_det_736.onnx"
_FP16_MAX = 65504.0


def _neck(*, closed: bool = True) -> onnx.ModelProto:
    """A miniature of the server neck: four big lateral convs, a residual branch, resize, concat."""
    rng = np.random.default_rng(0)
    inits = []

    def const(name, shape, scale=1.0):
        inits.append(
            numpy_helper.from_array((rng.normal(size=shape) * scale).astype(np.float32), name)
        )
        return name

    nodes = []
    lateral = []
    for i in range(4):
        out = f"lat{i}"
        nodes.append(
            helper.make_node(
                "Conv",
                ["x", const(f"w{i}", (4, 3, 3, 3), 300.0), const(f"b{i}", (4,), 50.0)],
                [out],
                name=f"Conv.{91 + i}",
                pads=[1, 1, 1, 1],
            )
        )
        lateral.append(out)
    # residual branch: conv, bias add, batch norm, relu, add back
    nodes += [
        helper.make_node("Conv", [lateral[0], const("wb", (4, 4, 1, 1))], ["br0"], name="Conv.95"),
        helper.make_node("Add", ["br0", const("bb", (1, 4, 1, 1), 10.0)], ["br1"], name="Add.16"),
        helper.make_node(
            "BatchNormalization",
            ["br1", const("g0", (4,)), const("be0", (4,)), const("m0", (4,), 5.0), "v0"],
            ["br2"],
            name="BatchNormalization.80",
        ),
        helper.make_node("Relu", ["br2"], ["br3"], name="Relu.53"),
        helper.make_node("Add", [lateral[0], "br3"], ["res"], name="Add.50"),
        helper.make_node(
            "Resize", [lateral[1], "", "scales"], ["up"], name="Resize.3", mode="nearest"
        ),
        helper.make_node(
            "MaxPool", ["up"], ["down"], name="MaxPool.1", kernel_shape=[2, 2], strides=[2, 2]
        ),
        helper.make_node(
            "Concat", ["res", "down", lateral[2], lateral[3]], ["cat"], name="Concat.14", axis=1
        ),
        helper.make_node(
            "Conv", ["cat", const("wf", (2, 16, 3, 3))], ["fuse"], name="Conv.139", pads=[1, 1, 1, 1]
        ),
    ]
    inits += [
        numpy_helper.from_array(np.abs(rng.normal(size=(4,))).astype(np.float32) + 1.0, "v0"),
        numpy_helper.from_array(np.array([1, 1, 2, 2], np.float32), "scales"),
    ]
    out = "fuse"
    if closed:
        nodes += [
            helper.make_node(
                "BatchNormalization",
                ["fuse", const("g1", (2,)), const("be1", (2,)), const("m1", (2,), 1e4), "v1"],
                ["bn"],
                name="BatchNormalization.84",
            ),
            helper.make_node("Sigmoid", ["bn"], ["y"], name="Sigmoid.0"),
        ]
        inits.append(numpy_helper.from_array(np.full((2,), 4e6, np.float32), "v1"))
        out = "y"
    graph = helper.make_graph(
        nodes,
        "neck",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info(out, TensorProto.FLOAT, [1, 2, 8, 8])],
        inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])


def _run(path, x: np.ndarray, outputs: list[str] | None = None) -> list[np.ndarray]:
    model = onnx.load(str(path))
    for name in outputs or []:
        model.graph.output.append(onnx.ValueInfoProto(name=name))
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, {model.graph.input[0].name: x})


@pytest.mark.cpu
def test_scale_neck_keeps_output(tmp_path) -> None:
    """The rescaled neck gives the original output while its activations shrink by the scale."""
    path = tmp_path / "neck.onnx"
    onnx.save(_neck(), str(path))
    x = np.random.default_rng(1).normal(size=(1, 3, 8, 8)).astype(np.float32)
    ref = _run(path, x, ["fuse"])
    scale_ppocrv5_server_det_neck(path, scale=256.0)
    onnx.checker.check_model(str(path))
    got = _run(path, x, ["fuse"])
    np.testing.assert_allclose(got[0], ref[0], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(got[1] * 256.0, ref[1], rtol=1e-4, atol=1e-2)


@pytest.mark.cpu
def test_scale_neck_rejects_other_graphs(tmp_path) -> None:
    """A graph without the closing batch norm is left alone and raises."""
    path = tmp_path / "neck.onnx"
    onnx.save(_neck(closed=False), str(path))
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="not the PP-OCRv5 server detector"):
        scale_ppocrv5_server_det_neck(path)
    assert path.read_bytes() == before


@pytest.mark.skipif(not SERVER_DET_ONNX.exists(), reason=f"missing {SERVER_DET_ONNX}")
def test_downloaded_server_det_fits_fp16() -> None:
    """Every activation of the downloaded server detector stays inside the FP16 range on a page."""
    page = np.full((1080, 1920, 3), 255, np.uint8)
    for i in range(14):
        cv2.putText(page, f"Line {i}: the quick brown fox", (60, 70 + 72 * i), 0, 1.6, (0, 0, 0), 3)
    img = cv2.resize(page, (960, 544)).astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    x = np.ascontiguousarray(img.transpose(2, 0, 1)[None], np.float32)
    model = onnx.load(str(SERVER_DET_ONNX))
    final = {o.name for o in model.graph.output}
    names = [
        o
        for node in model.graph.node
        for o in node.output
        if node.op_type != "Shape" and o not in final
    ]
    peaks = [float(np.abs(r).max()) for r in _run(SERVER_DET_ONNX, x, names)[1:] if r.size]
    assert max(peaks) < _FP16_MAX / 4
