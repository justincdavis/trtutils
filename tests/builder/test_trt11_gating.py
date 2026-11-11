# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for the TensorRT 11 gating of implicit INT8 calibration."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

import pytest

from trtutils._flags import FLAGS
from trtutils.builder import build_dla_engine, build_engine
from trtutils.models import YOLOv8
from trtutils.models._hooks import depthwise_fp32_build_hook, hands23_fp32_build_hook

pytestmark = pytest.mark.cpu

_MATCH = "removed in TensorRT 11"
_MATCH_WEAK = "Weakly-typed networks were removed in TensorRT 11"

# imports trtutils on a tensorrt module that lost the implicit calibrators
_NO_CALIBRATORS = """
import tensorrt as trt
for name in [n for n in dir(trt) if n.startswith("IInt8")]:
    delattr(trt, name)
import trtutils
from trtutils import builder
assert trtutils.FLAGS.INT8_CALIBRATION is False
assert not hasattr(builder, "EngineCalibrator")
assert "EngineCalibrator" not in builder.__all__
assert "EngineCalibrator" not in dir(builder)
assert builder.ImageBatcher and builder.SyntheticBatcher and builder.build_engine
"""


def test_import_without_calibrators() -> None:
    """The package imports and hides EngineCalibrator when trt lacks it."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _NO_CALIBRATORS], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_flag_matches_trt() -> None:
    """INT8_CALIBRATION tracks the presence of the entropy calibrator."""
    import tensorrt as trt

    assert hasattr(trt, "IInt8EntropyCalibrator2") == FLAGS.INT8_CALIBRATION


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"calibration_cache": "x.cache"}, id="calibration-cache"),
        pytest.param({"data_batcher": object()}, id="data-batcher"),
        pytest.param({"int8": True}, id="weakly-typed-int8"),
    ],
)
def test_build_engine_raises_without_calibration(monkeypatch, tmp_path, kwargs) -> None:
    """build_engine names the TensorRT 11 removal instead of failing later."""
    monkeypatch.setattr(FLAGS, "INT8_CALIBRATION", False)
    with pytest.raises(RuntimeError, match=_MATCH):
        build_engine("missing.onnx", tmp_path / "out.engine", **kwargs)


def test_build_engine_strongly_typed_not_gated(monkeypatch, tmp_path) -> None:
    """Explicit quantization (strongly typed) still reaches the normal checks."""
    monkeypatch.setattr(FLAGS, "INT8_CALIBRATION", False)
    with pytest.raises(FileNotFoundError):
        build_engine("missing.onnx", tmp_path / "out.engine", strongly_typed=True)


def test_build_dla_engine_raises_without_calibration(monkeypatch, tmp_path) -> None:
    """Weakly-typed DLA builds need calibration, so they raise the TRT 11 error."""
    monkeypatch.setattr(FLAGS, "INT8_CALIBRATION", False)
    with pytest.raises(RuntimeError, match=_MATCH):
        build_dla_engine("missing.onnx", tmp_path / "out.engine")


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"fp16": True}, id="fp16"),
        pytest.param({"fp8": True}, id="fp8"),
        pytest.param({"prefer_precision_constraints": True}, id="prefer-precision"),
        pytest.param({"reject_empty_algorithms": True}, id="reject-empty"),
        pytest.param({"layer_precision": [(0, None)]}, id="layer-precision"),
    ],
)
def test_build_engine_raises_without_weak_typing(monkeypatch, tmp_path, kwargs) -> None:
    """Precision options removed with weak typing name TensorRT 11."""
    monkeypatch.setattr(FLAGS, "WEAK_TYPING", False)
    with pytest.raises(RuntimeError, match=_MATCH_WEAK):
        build_engine("missing.onnx", tmp_path / "out.engine", **kwargs)


def test_flag_weak_typing_matches_trt() -> None:
    """WEAK_TYPING tracks the presence of BuilderFlag.FP16."""
    import tensorrt as trt

    assert hasattr(trt.BuilderFlag, "FP16") == FLAGS.WEAK_TYPING


@pytest.mark.parametrize(
    "hook",
    [
        pytest.param(depthwise_fp32_build_hook, id="depthwise"),
        pytest.param(hands23_fp32_build_hook, id="hands23"),
    ],
)
def test_precision_hooks_noop_without_weak_typing(monkeypatch, hook) -> None:
    """FP32 pinning hooks return no overrides where per-layer precision is gone."""
    monkeypatch.setattr(FLAGS, "WEAK_TYPING", False)
    assert hook(onnx="missing.onnx") == {}


def test_model_build_default_fp16_off_without_weak_typing(monkeypatch, tmp_path) -> None:
    """Model.build stops defaulting to fp16 where the flag no longer exists."""
    monkeypatch.setattr(FLAGS, "WEAK_TYPING", False)
    with patch("trtutils.models._model.build_engine") as mock_build:
        YOLOv8.build(tmp_path / "fake.onnx", tmp_path / "out.engine")
    assert mock_build.call_args.kwargs["fp16"] is False
