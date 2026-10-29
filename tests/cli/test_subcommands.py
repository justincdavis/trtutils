# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for CLI handler functions with mocked dependencies."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import trtutils
from trtutils.__main__ import (
    _AUTOTUNE_TASKS,
    _autotune,
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
    "autotune": {
        "engine": _FAKE_ENGINE,
        "task": "classify",
        "input": _FAKE_IMG,
        "iterations": 3,
        "warmup": 1,
        "preprocessors": None,
        "postprocessors": None,
        "no_graph_off": False,
        "verbose": False,
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
        "batch": None,
        "dynamic": None,
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
def test_benchmark_dispatches_by_jetson_flag(jetson, mock_target) -> None:
    """The jetson flag selects between standard and jetson benchmark backends."""
    metric = MagicMock(mean=0.01, median=0.01, min=0.005, max=0.02)
    with patch(mock_target) as mock_bench:
        mock_bench.return_value = MagicMock(latency=metric, energy=metric, power_draw=metric)
        with patch.object(Path, "exists", return_value=True):
            _benchmark(_args("benchmark", jetson=jetson))
        mock_bench.assert_called_once()


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


@pytest.mark.parametrize("dynamic", [None, True])
@pytest.mark.parametrize("batch", [None, 8])
def test_download_batch_and_dynamic_forwarded(batch, dynamic) -> None:
    """The CLI's --batch and --dynamic flags reach download() as keywords, unset as None."""
    with patch("trtutils.__main__.trtutils.download.download") as mock_dl:
        _download(_args("download", batch=batch, dynamic=dynamic))
    mock_dl.assert_called_once()
    assert mock_dl.call_args.kwargs["batch"] == batch
    assert mock_dl.call_args.kwargs["dynamic"] is dynamic


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


def _trial(pre, post, *, graph, median, p90, same, error) -> trtutils.image.AutotuneTrial:
    return trtutils.image.AutotuneTrial(
        preprocessor=pre,
        postprocessor=post,
        cuda_graph=graph,
        median_ms=median,
        p90_ms=p90,
        same_as_default=same,
        error=error,
    )


def _logged(log: MagicMock) -> str:
    return "\n".join(c.args[0] for c in log.info.call_args_list)


@pytest.mark.parametrize("task", ["detect", "classify", "segment", "pose", "obb", "depth", "hand"])
@pytest.mark.parametrize("no_graph_off", [False, True])
def test_autotune_dispatches_task_and_options(task, no_graph_off, tmp_path) -> None:
    """The task picks the model class and the flags reach autotune()."""
    img = tmp_path / "a.jpg"
    img.touch()
    trial = _trial("cuda", "cpu", graph=True, median=1.0, p90=2.0, same=True, error=None)
    with patch("trtutils.__main__.cv2.imread", return_value=np.zeros((8, 8, 3), np.uint8)), patch(
        "trtutils.__main__.trtutils.image.autotune", return_value=[trial]
    ) as mock:
        _autotune(_args("autotune", task=task, input=str(img), no_graph_off=no_graph_off))
    assert mock.call_args[0][0] is _AUTOTUNE_TASKS[task]
    assert mock.call_args.kwargs["cuda_graphs"] == ((True,) if no_graph_off else None)
    assert mock.call_args.kwargs["iterations"] == 3
    assert mock.call_args.kwargs["warmup"] == 1


def test_autotune_directory_takes_sorted_images(tmp_path) -> None:
    """A directory yields its sorted jpg/jpeg/png files and ignores the rest."""
    for name in ("b.png", "a.jpg", "c.JPEG", "notes.txt"):
        (tmp_path / name).touch()
    trial = _trial("cuda", "cpu", graph=True, median=1.0, p90=2.0, same=True, error=None)
    with patch(
        "trtutils.__main__.cv2.imread", return_value=np.zeros((8, 8, 3), np.uint8)
    ) as read, patch("trtutils.__main__.trtutils.image.autotune", return_value=[trial]):
        _autotune(_args("autotune", input=str(tmp_path)))
    assert [Path(c[0][0]).name for c in read.call_args_list] == ["a.jpg", "b.png", "c.JPEG"]


def test_autotune_empty_directory_raises(tmp_path) -> None:
    """A directory without images is an error."""
    with pytest.raises(ValueError, match="No images found"):
        _autotune(_args("autotune", input=str(tmp_path)))


def test_autotune_unreadable_image_raises(tmp_path) -> None:
    """An image cv2 cannot read is an error."""
    (tmp_path / "a.jpg").touch()
    with patch("trtutils.__main__.cv2.imread", return_value=None), pytest.raises(
        ValueError, match="Could not read image"
    ):
        _autotune(_args("autotune", input=str(tmp_path)))


def test_autotune_logs_table_and_best(tmp_path) -> None:
    """Every trial gets a row and the fastest working one is reported as best."""
    (tmp_path / "a.jpg").touch()
    ok = _trial("hybrid", "cuda", graph=True, median=0.5, p90=0.6, same=True, error=None)
    diff = _trial("cpu", "cuda", graph=False, median=0.9, p90=1.0, same=False, error=None)
    bad = _trial(
        "trt", "cpu", graph=True, median=float("inf"), p90=float("inf"), same=None, error="boom"
    )
    with patch("trtutils.__main__.LOG") as log, patch(
        "trtutils.__main__.cv2.imread", return_value=np.zeros((8, 8, 3), np.uint8)
    ), patch("trtutils.__main__.trtutils.image.autotune", return_value=[ok, diff, bad]):
        _autotune(_args("autotune", input=str(tmp_path)))
    text = _logged(log)
    assert "same" in text
    assert "differs" in text
    assert "error: boom" in text
    assert (
        "Best: preprocessor='hybrid', postprocessor='cuda', cuda_graph=True (0.50 ms/image)" in text
    )


def test_autotune_all_failed_reports_none(tmp_path) -> None:
    """With every trial failed there is no best configuration."""
    (tmp_path / "a.jpg").touch()
    bad = _trial(
        "trt", "cpu", graph=True, median=float("inf"), p90=float("inf"), same=None, error="boom"
    )
    with patch("trtutils.__main__.LOG") as log, patch(
        "trtutils.__main__.cv2.imread", return_value=np.zeros((8, 8, 3), np.uint8)
    ), patch("trtutils.__main__.trtutils.image.autotune", return_value=[bad]):
        _autotune(_args("autotune", input=str(tmp_path)))
    assert "Best: none" in _logged(log)
