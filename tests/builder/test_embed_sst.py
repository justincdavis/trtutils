# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""embed_sst rewrites a detection ONNX input to uint8 HWC plus scale/swap/transpose."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from trtutils.builder.onnx import embed_sst
from trtutils.image._image_model import _image_binding
from trtutils.models import YOLOX, ResNet, YOLOv10
from trtutils.models._model import _embed_sst_scale

if TYPE_CHECKING:
    from pathlib import Path


def _identity_model(path: Path) -> None:
    images = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, ["batch", 3, "height", "width"]
    )
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, ["batch", 3, "height", "width"])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["out"])], "g", [images], [out]
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


def test_embed_sst_prepends_scale_swap_transpose(tmp_path: Path) -> None:
    """The image input becomes uint8 HWC and the original name is the transpose output."""
    source = tmp_path / "in.onnx"
    dest = tmp_path / "out.onnx"
    _identity_model(source)
    embed_sst(source, dest, "images", scale=1.0 / 255.0, height=8, width=8)

    model = onnx.load(dest)
    image = model.graph.input[0]
    dims = [(dim.dim_param or dim.dim_value) for dim in image.type.tensor_type.shape.dim]
    assert image.name == "images"
    assert image.type.tensor_type.elem_type == TensorProto.UINT8
    assert dims == ["batch", 8, 8, 3]
    assert [node.op_type for node in model.graph.node[:4]] == ["Cast", "Mul", "Gather", "Transpose"]
    assert model.graph.node[3].output[0] == "images_nchw"
    assert model.graph.node[4].op_type == "Identity"
    assert model.graph.node[4].input[0] == "images_nchw"


def test_embed_sst_rejects_a_missing_input(tmp_path: Path) -> None:
    """A name that is not an input is an error."""
    source = tmp_path / "in.onnx"
    _identity_model(source)
    with pytest.raises(ValueError, match="missing"):
        embed_sst(source, tmp_path / "out.onnx", "missing", scale=1.0, height=8, width=8)


def test_uint8_hwc_is_the_in_graph_layout() -> None:
    """Float NCHW still normalizes outside; uint8 NHWC means the graph already did."""
    assert _image_binding((1, 3, 640, 640), np.dtype(np.float32)) == ((640, 640), False)
    assert _image_binding((8, 640, 640, 3), np.dtype(np.uint8)) == ((640, 640), True)
    assert _image_binding((1, 1000), np.dtype(np.float32)) is None


def test_detection_scales_match_the_wrapper_defaults() -> None:
    """YOLO divides by 255; YOLOX keeps 0-255. Mean/std models are not this path."""
    assert _embed_sst_scale(YOLOv10) == pytest.approx(1.0 / 255.0)
    assert _embed_sst_scale(YOLOX) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="mean"):
        _embed_sst_scale(ResNet)
