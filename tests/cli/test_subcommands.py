# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for CLI handler functions with mocked dependencies."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from trtutils.__main__ import (
    _benchmark,
    _build,
    _classify,
    _clear_cache,
    _detect,
    _download,
    _profile,
)

pytestmark = pytest.mark.cpu

_FAKE_ENGINE = "/fake/test.engine"
_FAKE_ONNX = "/fake/model.onnx"
_FAKE_OUT_ENGINE = "/fake/model.engine"
_FAKE_IMG = "/fake/img.jpg"
_FAKE_JSON = "/fake/profile.json"

_DEFAULTS: dict[str, dict[str, object]] = {
    "benchmark": {
        "engine": _FAKE_ENGINE,
        "iterations": 100,
        "warmup_iterations": 10,
        "jetson": False,
        "tegra_interval": 5,
        "dla_core": None,
        "warmup": True,
        "cuda_graph": False,
        "verbose": False,
    },
    "build": {
        "onnx": _FAKE_ONNX,
        "output": _FAKE_OUT_ENGINE,
        "int8": False,
        "fp16": False,
        "calibration_dir": None,
        "input_shape": None,
        "input_dtype": None,
        "batch_size": 8,
        "data_order": "NCHW",
        "max_images": None,
        "resize_method": "letterbox",
        "input_scale": [0.0, 1.0],
        "shape": None,
        "timing_cache": None,
        "workspace": 4.0,
        "optimization_level": 3,
        "dla_core": None,
        "calibration_cache": None,
        "gpu_fallback": False,
        "direct_io": False,
        "prefer_precision_constraints": False,
        "reject_empty_algorithms": False,
        "ignore_timing_mismatch": False,
        "cache": False,
        "verbose": False,
    },
    "detect": {
        "engine": "/fake/det.engine",
        "input": _FAKE_IMG,
        "warmup_iterations": 10,
        "input_range": [0.0, 1.0],
        "preprocessor": "trt",
        "resize_method": "letterbox",
        "conf_thres": 0.1,
        "nms_iou_thres": 0.5,
        "dla_core": None,
        "warmup": True,
        "pagelocked_mem": False,
        "unified_mem": False,
        "extra_nms": False,
        "agnostic_nms": False,
        "no_warn": False,
        "verbose": False,
        "show": False,
    },
    "classify": {
        "engine": "/fake/cls.engine",
        "input": _FAKE_IMG,
        "warmup_iterations": 10,
        "input_range": [0.0, 1.0],
        "preprocessor": "trt",
        "resize_method": "letterbox",
        "dla_core": None,
        "warmup": True,
        "pagelocked_mem": False,
        "unified_mem": False,
        "no_warn": False,
        "verbose": False,
        "show": False,
    },
    "profile": {
        "engine": _FAKE_ENGINE,
        "output": _FAKE_JSON,
        "iterations": 100,
        "warmup_iterations": 10,
        "dla_core": None,
        "warmup": True,
        "verbose": False,
        "save_raw": False,
        "jetson": False,
        "tegra_interval": 5,
    },
    "download": {
        "model": "yolov8n",
        "output": Path("/fake/yolov8n.onnx"),
        "list_models": False,
        "opset": 17,
        "imgsz": None,
        "requirements_export": None,
        "verbose": False,
        "no_cache": False,
        "simplify": None,
    },
}


def _args(subcommand: str, **overrides: object) -> SimpleNamespace:
    return SimpleNamespace(**{**_DEFAULTS[subcommand], **overrides})


@pytest.mark.parametrize(
    ("jetson", "mock_target"),
    [
        pytest.param(False, "trtutils.__main__.trtutils.benchmark_engine", id="standard"),
        pytest.param(True, "trtutils.__main__.trtutils.jetson.benchmark_engine", id="jetson"),
    ],
)
@pytest.mark.parametrize(
    "cuda_graph",
    [pytest.param(False, id="no-cuda-graph"), pytest.param(True, id="cuda-graph")],
)
def test_benchmark_dispatches_by_jetson_flag(jetson, mock_target, cuda_graph) -> None:
    """The jetson flag selects the backend; --cuda_graph is forwarded to it."""
    metric = MagicMock(mean=0.01, median=0.01, min=0.005, max=0.02)
    with patch(mock_target) as mock_bench:
        mock_bench.return_value = MagicMock(latency=metric, energy=metric, power_draw=metric)
        with patch.object(Path, "exists", return_value=True):
            _benchmark(_args("benchmark", jetson=jetson, cuda_graph=cuda_graph))
        mock_bench.assert_called_once()
        assert mock_bench.call_args.kwargs["cuda_graph"] is cuda_graph


def test_build_int8_calibration_dir_without_input_shape_raises() -> None:
    """int8 + calibration_dir but no input_shape raises ValueError."""
    args = _args("build", int8=True, calibration_dir="/fake/calib", input_shape=None)
    with pytest.raises(ValueError, match="Input shape must be provided"):
        _build(args)


@pytest.mark.parametrize(
    ("handler", "subcommand", "bad_input"),
    [
        pytest.param(_detect, "detect", "/fake/data.txt", id="detect"),
        pytest.param(_classify, "classify", "/fake/data.csv", id="classify"),
    ],
)
def test_image_handler_rejects_invalid_extension(handler, subcommand, bad_input) -> None:
    """Image handlers reject inputs with unsupported file extensions."""
    with pytest.raises(ValueError, match="Invalid input file"):
        handler(_args(subcommand, input=bad_input))


def test_profile_nonexistent_engine_raises_file_not_found() -> None:
    """A missing engine file raises FileNotFoundError."""
    args = _args("profile", engine="/fake/nonexistent_abc123.engine")
    with pytest.raises(FileNotFoundError, match="Cannot find provided engine"):
        _profile(args)


@pytest.mark.parametrize(
    ("simplify_arg", "expected"),
    [
        pytest.param(None, None, id="none"),
        pytest.param([], True, id="empty-list"),
        pytest.param(["polygraphy"], ["polygraphy"], id="with-tools"),
    ],
)
def test_download_simplify_argument_translation(simplify_arg, expected) -> None:
    """The CLI's --simplify flag translates correctly into the download() simplify kwarg."""
    with patch("trtutils.__main__.trtutils.download.download") as mock_dl:
        _download(_args("download", simplify=simplify_arg))
    mock_dl.assert_called_once()
    assert mock_dl.call_args[1]["simplify"] == expected


def test_download_list_models_skips_download() -> None:
    """--list_models prints configs without invoking the downloader."""
    with patch("trtutils.__main__.trtutils.download.load_model_configs") as mock_configs, patch(
        "trtutils.__main__.trtutils.download.download"
    ) as mock_dl:
        mock_configs.return_value = {"yolo": {"yolov8n": {}, "yolov8s": {}}}
        _download(_args("download", list_models=True))
    mock_configs.assert_called_once()
    mock_dl.assert_not_called()


def test_clear_cache_forwards_no_warn() -> None:
    """_clear_cache forwards the no_warn flag to caching_tools.clear."""
    with patch("trtutils.__main__.caching_tools.clear") as mock_clear:
        _clear_cache(SimpleNamespace(no_warn=True))
    mock_clear.assert_called_once_with(no_warn=True)
