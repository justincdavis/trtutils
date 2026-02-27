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
    _build_dla,
    _build_yolo,
    _can_run_on_dla,
    _classify,
    _clear_cache,
    _detect,
    _download,
    _inspect,
    _profile,
)


# ---------------------------------------------------------------------------
# Arg factories
# ---------------------------------------------------------------------------
def _benchmark_args(**overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "engine": "/fake/test.engine",
        "iterations": 100,
        "warmup_iterations": 10,
        "jetson": False,
        "tegra_interval": 5,
        "dla_core": None,
        "warmup": True,
        "verbose": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _build_args(**overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "onnx": "/fake/model.onnx",
        "output": "/fake/model.engine",
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
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _build_dla_args(**overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "onnx": "/fake/model.onnx",
        "output": "/fake/model.engine",
        "calibration_dir": "/fake/calib",
        "input_shape": (224, 224, 3),
        "input_dtype": "float32",
        "batch_size": 8,
        "data_order": "NCHW",
        "max_images": None,
        "resize_method": "letterbox",
        "input_scale": [0.0, 1.0],
        "shape": None,
        "timing_cache": None,
        "workspace": 4.0,
        "optimization_level": 3,
        "dla_core": 0,
        "max_chunks": 1,
        "min_layers": 20,
        "calibration_cache": None,
        "direct_io": False,
        "prefer_precision_constraints": False,
        "reject_empty_algorithms": False,
        "ignore_timing_mismatch": False,
        "cache": False,
        "verbose": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _detect_args(**overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "engine": "/fake/det.engine",
        "input": "/fake/img.jpg",
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
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _classify_args(**overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "engine": "/fake/cls.engine",
        "input": "/fake/img.jpg",
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
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _inspect_args(**overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "engine": "/fake/test.engine",
        "verbose": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _profile_args(**overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "engine": "/fake/test.engine",
        "output": "/fake/profile.json",
        "iterations": 100,
        "warmup_iterations": 10,
        "dla_core": None,
        "warmup": True,
        "verbose": False,
        "save_raw": False,
        "jetson": False,
        "tegra_interval": 5,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _download_args(**overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "model": "yolov8n",
        "output": Path("/fake/yolov8n.onnx"),
        "list_models": False,
        "opset": 17,
        "imgsz": None,
        "requirements_export": None,
        "verbose": False,
        "no_cache": False,
        "simplify": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _clear_cache_args(**overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "no_warn": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# _benchmark handler
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestBenchmarkHandler:
    """Tests for _benchmark handler."""

    def test_empty_namespace_raises_attribute_error(self) -> None:
        """An empty SimpleNamespace should raise AttributeError (missing engine)."""
        with pytest.raises(AttributeError):
            _benchmark(SimpleNamespace())

    @patch("trtutils.__main__.trtutils.benchmark_engine")
    def test_calls_benchmark_engine(self, mock_bench: MagicMock) -> None:
        """Standard (non-Jetson) path calls trtutils.benchmark_engine."""
        metric = MagicMock(mean=0.01, median=0.01, min=0.005, max=0.02)
        mock_bench.return_value = MagicMock(latency=metric)

        args = _benchmark_args(engine="/fake/fake.engine")
        with patch.object(Path, "exists", return_value=True):
            _benchmark(args)

        mock_bench.assert_called_once()
        call_kwargs = mock_bench.call_args
        assert call_kwargs[1]["iterations"] == 100
        assert call_kwargs[1]["warmup_iterations"] == 10

    @patch("trtutils.__main__.trtutils.jetson.benchmark_engine")
    def test_jetson_path(self, mock_jbench: MagicMock) -> None:
        """Jetson path calls trtutils.jetson.benchmark_engine."""
        metric = MagicMock(mean=0.01, median=0.01, min=0.005, max=0.02)
        mock_jbench.return_value = MagicMock(
            latency=metric,
            energy=metric,
            power_draw=metric,
        )

        args = _benchmark_args(engine="/fake/fake.engine", jetson=True)
        with patch.object(Path, "exists", return_value=True):
            _benchmark(args)

        mock_jbench.assert_called_once()


# ---------------------------------------------------------------------------
# _build handler
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestBuildHandler:
    """Tests for _build handler."""

    def test_empty_namespace_raises_attribute_error(self) -> None:
        """An empty SimpleNamespace should raise AttributeError."""
        with pytest.raises(AttributeError):
            _build(SimpleNamespace())

    @patch("trtutils.__main__.trtutils.build_engine")
    def test_calls_build_engine(self, mock_build: MagicMock) -> None:
        """_build forwards args to trtutils.build_engine."""
        args = _build_args()
        _build(args)

        mock_build.assert_called_once()
        call_kwargs = mock_build.call_args[1]
        assert call_kwargs["onnx"] == Path("/fake/model.onnx")
        assert call_kwargs["output"] == Path("/fake/model.engine")
        assert call_kwargs["fp16"] is False
        assert call_kwargs["int8"] is False

    def test_int8_calibration_dir_without_input_shape_raises(self) -> None:
        """int8 + calibration_dir but no input_shape raises ValueError."""
        args = _build_args(int8=True, calibration_dir="/fake/calib", input_shape=None)
        with pytest.raises(ValueError, match="Input shape must be provided"):
            _build(args)


# ---------------------------------------------------------------------------
# _build_yolo handler
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestBuildYoloHandler:
    """Tests for _build_yolo handler."""

    def test_empty_namespace_raises_attribute_error(self) -> None:
        """An empty SimpleNamespace should raise AttributeError."""
        with pytest.raises(AttributeError):
            _build_yolo(SimpleNamespace())

    @patch("trtutils.__main__.trtutils.build_engine")
    @patch("trtutils.__main__.trtutils.builder.hooks.yolo_efficient_nms_hook")
    def test_calls_build_with_yolo_hook(self, mock_hook: MagicMock, mock_build: MagicMock) -> None:
        """_build_yolo injects the YOLO NMS hook into the build call."""
        mock_hook.return_value = MagicMock()
        args = _build_args(
            num_classes=80,
            conf_threshold=0.25,
            iou_threshold=0.5,
            top_k=100,
            box_coding="center_size",
            class_agnostic=False,
        )
        _build_yolo(args)

        mock_hook.assert_called_once_with(
            num_classes=80,
            conf_threshold=0.25,
            iou_threshold=0.5,
            top_k=100,
            class_agnostic=False,
            box_coding="center_size",
        )
        mock_build.assert_called_once()
        # The hooks list should contain the mock hook
        call_kwargs = mock_build.call_args[1]
        assert len(call_kwargs["hooks"]) == 1


# ---------------------------------------------------------------------------
# _build_dla handler
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestBuildDlaHandler:
    """Tests for _build_dla handler."""

    def test_empty_namespace_raises_attribute_error(self) -> None:
        """An empty SimpleNamespace should raise AttributeError."""
        with pytest.raises(AttributeError):
            _build_dla(SimpleNamespace())

    @patch("trtutils.__main__.trtutils.builder.build_dla_engine")
    @patch("trtutils.__main__.trtutils.builder.ImageBatcher")
    def test_calls_build_dla_engine(
        self, mock_batcher: MagicMock, mock_build_dla: MagicMock
    ) -> None:
        """_build_dla forwards to trtutils.builder.build_dla_engine."""
        args = _build_dla_args()
        _build_dla(args)

        mock_batcher.assert_called_once()
        mock_build_dla.assert_called_once()
        call_kwargs = mock_build_dla.call_args[1]
        assert call_kwargs["onnx"] == Path("/fake/model.onnx")
        assert call_kwargs["dla_core"] == 0


# ---------------------------------------------------------------------------
# _can_run_on_dla handler
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestCanRunOnDlaHandler:
    """Tests for _can_run_on_dla handler."""

    def test_empty_namespace_raises_attribute_error(self) -> None:
        """An empty SimpleNamespace should raise AttributeError."""
        with pytest.raises(AttributeError):
            _can_run_on_dla(SimpleNamespace())

    @patch("trtutils.__main__.trtutils.builder.can_run_on_dla")
    def test_calls_can_run_on_dla(self, mock_check: MagicMock) -> None:
        """_can_run_on_dla forwards to trtutils.builder.can_run_on_dla."""
        # chunks: list of (name, start, end, compatible)
        mock_check.return_value = (
            True,
            [("chunk0", 0, 9, True), ("chunk1", 10, 19, True)],
        )
        args = SimpleNamespace(
            onnx="/fake/model.onnx",
            verbose_layers=False,
            verbose_chunks=False,
        )
        _can_run_on_dla(args)

        mock_check.assert_called_once()
        call_kwargs = mock_check.call_args[1]
        assert call_kwargs["onnx"] == Path("/fake/model.onnx")


# ---------------------------------------------------------------------------
# _detect handler
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestDetectHandler:
    """Tests for _detect handler."""

    def test_empty_namespace_raises_attribute_error(self) -> None:
        """An empty SimpleNamespace should raise AttributeError."""
        with pytest.raises(AttributeError):
            _detect(SimpleNamespace())

    def test_invalid_file_extension_raises_value_error(self) -> None:
        """An unsupported file extension raises ValueError."""
        args = _detect_args(input="/fake/data.txt")
        with pytest.raises(ValueError, match="Invalid input file"):
            _detect(args)


# ---------------------------------------------------------------------------
# _classify handler
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestClassifyHandler:
    """Tests for _classify handler."""

    def test_empty_namespace_raises_attribute_error(self) -> None:
        """An empty SimpleNamespace should raise AttributeError."""
        with pytest.raises(AttributeError):
            _classify(SimpleNamespace())

    def test_invalid_file_extension_raises_value_error(self) -> None:
        """An unsupported file extension raises ValueError."""
        args = _classify_args(input="/fake/data.csv")
        with pytest.raises(ValueError, match="Invalid input file"):
            _classify(args)


# ---------------------------------------------------------------------------
# _inspect handler
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestInspectHandler:
    """Tests for _inspect handler."""

    def test_empty_namespace_raises_attribute_error(self) -> None:
        """An empty SimpleNamespace should raise AttributeError."""
        with pytest.raises(AttributeError):
            _inspect(SimpleNamespace())

    @patch("trtutils.__main__.trtutils.inspect.inspect_engine")
    def test_calls_inspect_engine(self, mock_inspect: MagicMock) -> None:
        """_inspect forwards to trtutils.inspect.inspect_engine."""
        mock_inspect.return_value = (
            1024 * 1024,  # engine_size
            1,  # max_batch
            [("input", (1, 3, 224, 224), "float32", "linear")],
            [("output", (1, 1000), "float32", "linear")],
        )
        args = _inspect_args()
        with patch.object(Path, "exists", return_value=True):
            _inspect(args)

        mock_inspect.assert_called_once()


# ---------------------------------------------------------------------------
# _profile handler
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestProfileHandler:
    """Tests for _profile handler."""

    def test_empty_namespace_raises_attribute_error(self) -> None:
        """An empty SimpleNamespace should raise AttributeError."""
        with pytest.raises(AttributeError):
            _profile(SimpleNamespace())

    def test_nonexistent_engine_raises_file_not_found(self) -> None:
        """A missing engine file raises FileNotFoundError."""
        args = _profile_args(engine="/fake/nonexistent_abc123.engine")
        with pytest.raises(FileNotFoundError, match="Cannot find provided engine"):
            _profile(args)


# ---------------------------------------------------------------------------
# _download handler
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestDownloadHandler:
    """Tests for _download handler."""

    @patch("trtutils.__main__.trtutils.download.load_model_configs")
    def test_list_models_mode(self, mock_configs: MagicMock) -> None:
        """--list_models prints configs and returns without downloading."""
        mock_configs.return_value = {"yolo": {"yolov8n": {}, "yolov8s": {}}}
        args = _download_args(list_models=True)
        _download(args)

        mock_configs.assert_called_once()

    @patch("trtutils.__main__.trtutils.download.download")
    def test_simplify_none(self, mock_dl: MagicMock) -> None:
        """simplify=None passes simplify_value=None to download."""
        args = _download_args(simplify=None)
        _download(args)

        mock_dl.assert_called_once()
        call_kwargs = mock_dl.call_args
        # positional: model, output, opset, imgsz
        assert call_kwargs[1]["simplify"] is None

    @patch("trtutils.__main__.trtutils.download.download")
    def test_simplify_empty_list(self, mock_dl: MagicMock) -> None:
        """simplify=[] (no tools specified) passes simplify_value=True."""
        args = _download_args(simplify=[])
        _download(args)

        mock_dl.assert_called_once()
        call_kwargs = mock_dl.call_args
        assert call_kwargs[1]["simplify"] is True

    @patch("trtutils.__main__.trtutils.download.download")
    def test_simplify_with_tools(self, mock_dl: MagicMock) -> None:
        """simplify=["polygraphy"] passes the tool list through."""
        args = _download_args(simplify=["polygraphy"])
        _download(args)

        mock_dl.assert_called_once()
        call_kwargs = mock_dl.call_args
        assert call_kwargs[1]["simplify"] == ["polygraphy"]


# ---------------------------------------------------------------------------
# _clear_cache handler
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestClearCacheHandler:
    """Tests for _clear_cache handler."""

    @patch("trtutils.__main__.caching_tools.clear")
    def test_calls_cache_clear(self, mock_clear: MagicMock) -> None:
        """_clear_cache calls caching_tools.clear."""
        args = _clear_cache_args()
        _clear_cache(args)

        mock_clear.assert_called_once_with(no_warn=False)
