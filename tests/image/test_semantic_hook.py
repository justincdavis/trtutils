# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/hooks/_semantic.py and models/_hooks.py::argmax_build_hook -- argmax in the engine."""

from __future__ import annotations

import cv2
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from trtutils import TRTEngine
from trtutils.builder import build_engine
from trtutils.builder._onnx import read_onnx
from trtutils.compat._libs import trt
from trtutils.core import Buffer
from trtutils.image.hooks import semantic_argmax_hook
from trtutils.models._hooks import argmax_build_hook

_SIZE = 32
_CLASSES = 5


@pytest.fixture(scope="module")
def logits_onnx(tmp_path_factory):
    """Factory for a (1, 5, 32, 32) -> logits ONNX, pooled by ``stride`` (1 keeps the size)."""

    def _make(stride: int) -> str:
        node = (
            helper.make_node(
                "AveragePool", ["input"], ["logits"], kernel_shape=[stride] * 2, strides=[stride] * 2
            )
            if stride > 1
            else helper.make_node("Relu", ["input"], ["logits"])
        )
        side = _SIZE // stride
        graph = helper.make_graph(
            [node],
            "logits",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, _CLASSES, _SIZE, _SIZE])],
            [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, _CLASSES, side, side])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        path = tmp_path_factory.mktemp("semantic_hook") / f"logits_s{stride}.onnx"
        onnx.save(model, str(path))
        return str(path)

    return _make


@pytest.mark.parametrize(
    ("stride", "resized"),
    [
        pytest.param(4, True, id="quarter-res-resized"),
        pytest.param(1, False, id="input-res-no-resize"),
    ],
)
def test_hook_replaces_logits_with_class_map(logits_onnx, stride, resized) -> None:
    """The hook unmarks the logits, outputs an int32 class_map, and resizes only smaller logits."""
    network, _builder, _config, _parser = read_onnx(logits_onnx(stride))
    network = semantic_argmax_hook(_SIZE, _SIZE)(network)

    assert network.num_outputs == 1
    output = network.get_output(0)
    assert output.name == "class_map"
    assert output.dtype == trt.DataType.INT32
    assert tuple(output.shape) == (1, 1, _SIZE, _SIZE)
    types = {network.get_layer(i).type for i in range(network.num_layers)}
    assert trt.LayerType.TOPK in types
    assert (trt.LayerType.RESIZE in types) is resized


def test_hook_engine_matches_bilinear_argmax(logits_onnx, tmp_path) -> None:
    """The class map equals a half-pixel bilinear upsample of the logits, then argmax."""
    onnx_path = logits_onnx(4)
    engine_path = tmp_path / "class_map.engine"
    build_engine(onnx_path, engine_path, hooks=[semantic_argmax_hook(_SIZE, _SIZE)])
    engine = TRTEngine(engine_path, warmup=False)
    data = np.random.default_rng(0).standard_normal((1, _CLASSES, _SIZE, _SIZE)).astype(np.float32)

    (class_map,) = engine.execute([Buffer.wrap(data)])

    pooled = data[0].reshape(_CLASSES, 8, 4, 8, 4).mean(axis=(2, 4))
    upsampled = np.stack(
        [cv2.resize(c, (_SIZE, _SIZE), interpolation=cv2.INTER_LINEAR) for c in pooled]
    )
    expected = upsampled.argmax(axis=0)
    assert engine.output_names == ["class_map"]
    assert (np.asarray(class_map).reshape(_SIZE, _SIZE) == expected).mean() >= 0.99
    del engine


@pytest.mark.parametrize(
    ("argmax", "hooked"),
    [pytest.param(True, True, id="argmax"), pytest.param(False, False, id="no-argmax")],
)
def test_argmax_build_hook_overrides(argmax, hooked) -> None:
    """argmax_build_hook adds one network hook, or nothing with argmax=False."""
    overrides = argmax_build_hook(imgsz=_SIZE, argmax=argmax, onnx="unused.onnx")
    assert ("hooks" in overrides) is hooked
    assert len(overrides.get("hooks", [])) == int(hooked)
