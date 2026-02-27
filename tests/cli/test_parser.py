# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
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


# ---------------------------------------------------------------------------
# No-args / help
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestMainNoArgs:
    """Tests for invoking the CLI with no arguments or --help."""

    def test_no_args_prints_help(self) -> None:
        """Calling main with no arguments prints the help text."""
        with patch("sys.argv", ["trtutils"]):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                _main()
                assert "Utilities for TensorRT." in mock_stdout.getvalue()

    def test_help_flag_exits_zero(self) -> None:
        """--help exits with code 0."""
        with patch("sys.argv", ["trtutils", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                _main()
            assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Subcommand --help
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestSubcommandHelp:
    """Each subcommand should accept --help and exit 0."""

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
    def test_subcommand_help_exits_zero(self, subcommand: str) -> None:
        """Each subcommand's --help exits with code 0."""
        with patch("sys.argv", ["trtutils", subcommand, "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                _main()
            assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Missing required args
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestSubcommandRequiredArgs:
    """Subcommands should exit 2 when required arguments are missing."""

    @pytest.mark.parametrize(
        "argv",
        [
            ["trtutils", "benchmark"],
            ["trtutils", "build"],
            ["trtutils", "build_yolo"],
            ["trtutils", "can_run_on_dla"],
            ["trtutils", "build_dla"],
            ["trtutils", "detect"],
            ["trtutils", "classify"],
            ["trtutils", "inspect"],
            ["trtutils", "profile"],
        ],
        ids=[
            "benchmark-no-engine",
            "build-no-onnx-output",
            "build_yolo-no-onnx-output",
            "can_run_on_dla-no-onnx",
            "build_dla-no-onnx-output",
            "detect-no-engine-input",
            "classify-no-engine-input",
            "inspect-no-engine",
            "profile-no-engine-output",
        ],
    )
    def test_missing_required_args_exit_2(self, argv: list) -> None:
        """Missing required positional/flag args cause exit code 2."""
        with patch("sys.argv", argv):
            with pytest.raises(SystemExit) as exc_info:
                _main()
            assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# Parser defaults
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestParserDefaults:
    """Verify parsed default values for key subcommands."""

    def test_benchmark_defaults(self) -> None:
        """Benchmark subcommand defaults are correct."""
        with patch("sys.argv", ["trtutils", "benchmark", "--engine", _ENGINE]):
            with patch("trtutils.__main__._benchmark") as mock:
                _main()
                args = mock.call_args[0][0]
                assert args.iterations == 1000
                assert args.warmup_iterations == 10
                assert args.jetson is False
                assert args.tegra_interval == 5
                assert args.dla_core is None
                assert args.verbose is False

    def test_build_defaults(self) -> None:
        """Build subcommand defaults are correct."""
        argv = ["trtutils", "build", "--onnx", _ONNX, "--output", _OUT_ENGINE]
        with patch("sys.argv", argv):
            with patch("trtutils.__main__._build") as mock:
                _main()
                args = mock.call_args[0][0]
                assert args.fp16 is False
                assert args.int8 is False
                assert args.workspace == 4.0
                assert args.optimization_level == 3
                assert args.calibration_dir is None
                assert args.input_shape is None
                assert args.cache is False
                assert args.direct_io is False

    def test_detect_defaults(self) -> None:
        """Detect subcommand defaults are correct."""
        argv = ["trtutils", "detect", "--engine", _ENGINE, "--input", _IMG]
        with patch("sys.argv", argv):
            with patch("trtutils.__main__._detect") as mock:
                _main()
                args = mock.call_args[0][0]
                assert args.conf_thres == pytest.approx(0.1)
                assert args.nms_iou_thres == pytest.approx(0.5)
                assert args.preprocessor == "trt"
                assert args.resize_method == "letterbox"
                assert args.extra_nms is False
                assert args.agnostic_nms is False
                assert args.show is False
                assert args.pagelocked_mem is False
                assert args.unified_mem is False

    def test_classify_defaults(self) -> None:
        """Classify subcommand defaults are correct."""
        argv = ["trtutils", "classify", "--engine", _ENGINE, "--input", _IMG]
        with patch("sys.argv", argv):
            with patch("trtutils.__main__._classify") as mock:
                _main()
                args = mock.call_args[0][0]
                assert args.preprocessor == "trt"
                assert args.input_range == [0.0, 1.0]
                assert args.show is False
                assert args.warmup is False
                assert args.warmup_iterations == 10

    def test_profile_defaults(self) -> None:
        """Profile subcommand defaults are correct."""
        argv = ["trtutils", "profile", "--engine", _ENGINE, "--output", _JSON]
        with patch("sys.argv", argv):
            with patch("trtutils.__main__._profile") as mock:
                _main()
                args = mock.call_args[0][0]
                assert args.iterations == 100
                assert args.warmup_iterations == 10
                assert args.save_raw is False
                assert args.jetson is False
                assert args.tegra_interval == 5
                assert args.dla_core is None
