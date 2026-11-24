# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for argparse configuration in trtutils.__main__."""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

import pytest

from trtutils.__main__ import _main

# Fake paths that do not reference /tmp to avoid S108 warnings.
_ENGINE = "/fake/test.engine"
_ONNX = "/fake/model.onnx"
_OUT_ENGINE = "/fake/model.engine"
_IMG = "/fake/img.jpg"
_JSON = "/fake/profile.json"

pytestmark = pytest.mark.cpu


def test_no_args_prints_help() -> None:
    """Calling main with no arguments prints the help text."""
    with patch("sys.argv", ["trtutils"]), patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        _main()
        assert "Utilities for TensorRT." in mock_stdout.getvalue()


def test_help_flag_exits_zero() -> None:
    """--help exits with code 0."""
    with patch("sys.argv", ["trtutils", "--help"]), pytest.raises(SystemExit) as exc_info:
        _main()
    assert exc_info.value.code == 0


@pytest.mark.parametrize(
    "subcommand",
    [
        "benchmark",
        "trtexec",
        "build",
        "build_yolo",
        "can_run_on_dla",
        "build_dla",
        "detect",
        "classify",
        "inspect",
        "profile",
        "download",
        "clear_cache",
    ],
)
def test_subcommand_help_exits_zero(subcommand) -> None:
    """Each subcommand's --help exits with code 0."""
    with patch("sys.argv", ["trtutils", subcommand, "--help"]), pytest.raises(
        SystemExit
    ) as exc_info:
        _main()
    assert exc_info.value.code == 0


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["trtutils", "benchmark"], id="benchmark-no-engine"),
        pytest.param(["trtutils", "build"], id="build-no-onnx-output"),
        pytest.param(["trtutils", "build_yolo"], id="build_yolo-no-onnx-output"),
        pytest.param(["trtutils", "can_run_on_dla"], id="can_run_on_dla-no-onnx"),
        pytest.param(["trtutils", "build_dla"], id="build_dla-no-onnx-output"),
        pytest.param(["trtutils", "detect"], id="detect-no-engine-input"),
        pytest.param(["trtutils", "classify"], id="classify-no-engine-input"),
        pytest.param(["trtutils", "inspect"], id="inspect-no-engine"),
        pytest.param(["trtutils", "profile"], id="profile-no-engine-output"),
        pytest.param(
            ["trtutils", "build", "--onnx", _ONNX, "--output", _OUT_ENGINE, "--shapes", "a:1"],
            id="build-unrecognized-arg",
        ),
    ],
)
def test_invalid_args_exit_2(argv) -> None:
    """Missing required args or unrecognized flags cause exit code 2."""
    with patch("sys.argv", argv), pytest.raises(SystemExit) as exc_info:
        _main()
    assert exc_info.value.code == 2


def test_trtexec_passthrough_args() -> None:
    """Trtexec forwards unknown args instead of erroring."""
    argv = ["trtutils", "trtexec", "--onnx=/fake/m.onnx", "--fp16"]
    with patch("sys.argv", argv), patch("trtutils.__main__.cli_trtexec") as mock_fn:
        _main()
    assert mock_fn.call_args.args[0] == ["--onnx=/fake/m.onnx", "--fp16"]


@pytest.mark.parametrize(
    ("argv", "mock_target", "expected"),
    [
        pytest.param(
            ["trtutils", "benchmark", "--engine", _ENGINE],
            "trtutils.__main__._benchmark",
            {
                "iterations": 1000,
                "warmup_iterations": 10,
                "jetson": False,
                "tegra_interval": 5,
                "dla_core": None,
                "cuda_graph": False,
                "verbose": False,
            },
            id="benchmark",
        ),
        pytest.param(
            ["trtutils", "build", "--onnx", _ONNX, "--output", _OUT_ENGINE],
            "trtutils.__main__._build",
            {
                "fp16": False,
                "int8": False,
                "workspace": 4.0,
                "optimization_level": 3,
                "calibration_dir": None,
                "input_shape": None,
                "cache": False,
            },
            id="build",
        ),
        pytest.param(
            ["trtutils", "detect", "--engine", _ENGINE, "--input", _IMG],
            "trtutils.__main__._detect",
            {
                "conf_thres": pytest.approx(0.1),
                "nms_iou_thres": pytest.approx(0.5),
                "preprocessor": "cuda",
                "cpu_resize": None,
                "resize_method": "letterbox",
                "extra_nms": False,
                "agnostic_nms": False,
                "show": False,
                "pagelocked_mem": False,
                "unified_mem": False,
            },
            id="detect",
        ),
        pytest.param(
            ["trtutils", "classify", "--engine", _ENGINE, "--input", _IMG],
            "trtutils.__main__._classify",
            {
                "preprocessor": "cuda",
                "cpu_resize": None,
                "resize_method": "letterbox",
                "crop_pct": 0.875,
                "input_range": [0.0, 1.0],
                "show": False,
                "warmup": False,
                "warmup_iterations": 10,
            },
            id="classify",
        ),
        pytest.param(
            ["trtutils", "profile", "--engine", _ENGINE, "--output", _JSON],
            "trtutils.__main__._profile",
            {
                "iterations": 100,
                "warmup_iterations": 10,
                "save_raw": False,
                "jetson": False,
                "tegra_interval": 5,
                "dla_core": None,
            },
            id="profile",
        ),
        pytest.param(
            ["trtutils", "detect", "--engine", _ENGINE, "--input", _IMG, "--warmup"],
            "trtutils.__main__._detect",
            {"warmup": True, "warmup_iterations": 10},
            id="detect-warmup-flag",
        ),
        pytest.param(
            ["trtutils", "detect", "--engine", _ENGINE, "--input", _IMG, "--cpu_resize"],
            "trtutils.__main__._detect",
            {"cpu_resize": True},
            id="detect-cpu-resize-flag",
        ),
        pytest.param(
            ["trtutils", "classify", "--engine", _ENGINE, "--input", _IMG, "--no_cpu_resize"],
            "trtutils.__main__._classify",
            {"cpu_resize": False},
            id="classify-no-cpu-resize-flag",
        ),
        pytest.param(
            ["trtutils", "download", "--model", "m", "--output", _ONNX],
            "trtutils.__main__._download",
            {"imgsz": None},
            id="download-imgsz-unset",
        ),
        pytest.param(
            ["trtutils", "download", "--model", "m", "--output", _ONNX, "--imgsz", "640"],
            "trtutils.__main__._download",
            {"imgsz": [640]},
            id="download-imgsz-square",
        ),
        pytest.param(
            ["trtutils", "download", "--model", "m", "--output", _ONNX, "--imgsz", "256", "128"],
            "trtutils.__main__._download",
            {"imgsz": [256, 128]},
            id="download-imgsz-height-width",
        ),
        pytest.param(
            ["trtutils", "download", "--model", "m", "--output", _ONNX],
            "trtutils.__main__._download",
            {"classes": None},
            id="download-classes-unset",
        ),
        pytest.param(
            [
                "trtutils",
                "download",
                "--model",
                "m",
                "--output",
                _ONNX,
                "--classes",
                "person",
                "traffic light",
            ],
            "trtutils.__main__._download",
            {"classes": ["person", "traffic light"]},
            id="download-classes",
        ),
    ],
)
def test_subcommand_defaults(argv, mock_target, expected) -> None:
    """Parsed default values match expected for each subcommand."""
    with patch("sys.argv", argv), patch(mock_target) as mock:
        _main()
    args = mock.call_args[0][0]
    for key, value in expected.items():
        assert getattr(args, key) == value, f"{key}: got {getattr(args, key)!r}, expected {value!r}"


@pytest.mark.parametrize(
    "flags",
    [
        pytest.param(["--cpu_resize", "--no_cpu_resize"], id="both-cpu-resize-flags"),
        pytest.param(["--preprocessor", "hybrid"], id="hybrid"),
    ],
)
def test_preprocessor_usage_errors(flags) -> None:
    """Both cpu_resize flags at once, and the removed hybrid preprocessor, are usage errors."""
    argv = ["trtutils", "detect", "--engine", _ENGINE, "--input", _IMG, *flags]
    with patch("sys.argv", argv), patch("sys.stderr", new_callable=StringIO):
        with pytest.raises(SystemExit) as exc_info:
            _main()
    assert exc_info.value.code == 2


def test_benchmark_namespace_keeps_warmup_iterations_only() -> None:
    """Benchmark parses --warmup_iterations but defines no warmup attribute."""
    with patch("sys.argv", ["trtutils", "benchmark", "--engine", _ENGINE]), patch(
        "trtutils.__main__._benchmark"
    ) as mock:
        _main()
    args = mock.call_args[0][0]
    assert args.warmup_iterations == 10
    assert not hasattr(args, "warmup")
