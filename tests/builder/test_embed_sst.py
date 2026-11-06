# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/builder/onnx/_sst.py -- normalize and transpose moved into the graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from trtutils.builder.onnx import embed_sst
from trtutils.image._image_model import _image_binding
from trtutils.image.preprocessors._process import preprocess
from trtutils.models import (
    DEIM,
    HOIDETR,
    YOLOX,
    DepthAnythingV2,
    Hands23,
    PPOCRDetector,
    PPOCRRecognizer,
    ResNet,
    YOLOv8Seg,
    YOLOv10,
)
from trtutils.models._model import _default_norm

if TYPE_CHECKING:
    from pathlib import Path

_SIZE = 8
_IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def _save_identity(path: Path, inputs: list, shape: list) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["out"])],
        "g",
        inputs,
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, shape)],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


@pytest.mark.parametrize(
    ("input_range", "mean", "std"),
    [
        pytest.param((0.0, 1.0), None, None, id="yolo"),
        pytest.param((0.0, 255.0), None, None, id="yolox"),
        pytest.param((-1.0, 1.0), None, None, id="offset"),
        pytest.param((0.0, 1.0), *_IMAGENET, id="imagenet"),
    ],
)
def test_embed_sst_matches_the_cpu_preprocessor(tmp_path: Path, input_range, mean, std) -> None:
    """The rewritten graph turns a uint8 BGR HWC image into the CPU preprocessor's tensor."""
    source = tmp_path / "in.onnx"
    dest = tmp_path / "out.onnx"
    shape = ["batch", 3, "height", "width"]
    # a size input in front of the image, like RT-DETRv3
    _save_identity(
        source,
        [
            helper.make_tensor_value_info("sizes", TensorProto.INT64, ["batch", 2]),
            helper.make_tensor_value_info("images", TensorProto.FLOAT, shape),
        ],
        shape,
    )
    embed_sst(
        source,
        dest,
        "images",
        height=_SIZE,
        width=_SIZE,
        input_range=input_range,
        mean=mean,
        std=std,
    )

    model = onnx.load(dest)
    assert [item.name for item in model.graph.input] == ["sizes", "images"]
    image = model.graph.input[1]
    dims = [(dim.dim_param or dim.dim_value) for dim in image.type.tensor_type.shape.dim]
    assert image.type.tensor_type.elem_type == TensorProto.UINT8
    assert dims == ["batch", _SIZE, _SIZE, 3]

    bgr = np.random.default_rng(0).integers(0, 256, (_SIZE, _SIZE, 3), dtype=np.uint8)
    expected, _, _ = preprocess(
        [bgr], (_SIZE, _SIZE), np.dtype(np.float32), input_range, mean=mean, std=std
    )
    session = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
    (got,) = session.run(None, {"images": bgr[None], "sizes": np.zeros((1, 2), np.int64)})
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("name", "shape", "match"),
    [
        pytest.param("missing", ["batch", 3, 8, 8], "no input", id="missing"),
        pytest.param("images", ["batch", 8, 8, 3], "must be", id="already-nhwc"),
        pytest.param("images", ["batch", 3, 8], "must be", id="rank-3"),
        pytest.param("images", ["batch", 3, 16, 16], "must be", id="static-size-mismatch"),
    ],
)
def test_embed_sst_rejects_bad_inputs(tmp_path: Path, name, shape, match) -> None:
    """Missing inputs, non-NCHW inputs (e.g. already NHWC), and other static sizes are refused."""
    source = tmp_path / "in.onnx"
    _save_identity(
        source, [helper.make_tensor_value_info("images", TensorProto.FLOAT, shape)], shape
    )
    with pytest.raises(ValueError, match=match):
        embed_sst(source, tmp_path / "out.onnx", name, height=8, width=8)


def test_embed_sst_rejects_mean_without_std(tmp_path: Path) -> None:
    """Mean and std come as a pair."""
    with pytest.raises(ValueError, match="both"):
        embed_sst(
            tmp_path / "in.onnx",
            tmp_path / "out.onnx",
            "images",
            height=8,
            width=8,
            mean=_IMAGENET[0],
        )


def test_uint8_hwc_is_the_in_graph_layout() -> None:
    """Float NCHW still normalizes outside; uint8 NHWC means the graph already did."""
    assert _image_binding((1, 3, 640, 480), np.dtype(np.float32)) == ((480, 640), False)
    assert _image_binding((8, 640, 480, 3), np.dtype(np.uint8)) == ((480, 640), True)
    assert _image_binding((8, 640, 480, 3), np.dtype(np.float32)) is None
    assert _image_binding((1, 1000), np.dtype(np.float32)) is None


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        pytest.param(YOLOv10, ((0, 1), None, None), id="yolov10"),
        pytest.param(YOLOv8Seg, ((0.0, 1.0), None, None), id="yolov8-seg"),
        pytest.param(YOLOX, ((0, 255), None, None), id="yolox"),
        pytest.param(Hands23, ((0.0, 255.0), None, None), id="hands23"),
        pytest.param(ResNet, ((0, 1), *_IMAGENET), id="resnet"),
        pytest.param(DepthAnythingV2, ((0, 1), *_IMAGENET), id="depth"),
        pytest.param(DEIM, ((0, 1), *_IMAGENET), id="deim"),
        pytest.param(HOIDETR, ((0.0, 1.0), *_IMAGENET), id="hoi-detr"),
    ],
)
def test_default_norm_follows_each_wrapper(cls: type, expected: tuple) -> None:
    """The norm baked by embed_sst is the one each wrapper's constructor defaults to."""
    assert _default_norm(cls) == expected


@pytest.mark.parametrize("cls", [PPOCRDetector, PPOCRRecognizer], ids=["det", "rec"])
def test_ocr_refuses_embed_sst(tmp_path: Path, cls: type) -> None:
    """OCR builds a per-image input shape and has its own build(), which takes no embed_sst."""
    with pytest.raises(TypeError, match="embed_sst"):
        cls.build(tmp_path / "missing.onnx", tmp_path / "out.engine", embed_sst=True)
