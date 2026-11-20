# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/shift/_cli.py -- the research shift subcommands."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from trtutils.__main__ import _main
from trtutils.research.shift import ShiftConfig
from trtutils.research.shift._cli import _make_config, _parse_models, _read_labels

pytestmark = pytest.mark.cpu

_ENGINE = "/fake/model.engine"
_JSON = "/fake/traits.json"
_VIDEO = "/fake/video.mp4"
_CONFIG_FIELDS = [
    "accuracy_threshold",
    "momentum",
    "distance_threshold",
    "accuracy_weight",
    "energy_weight",
    "latency_weight",
    "similarity",
    "confidence_metric",
    "accuracy_metric",
    "num_bins",
    "outlier_percentile",
    "min_connects",
]


@pytest.mark.parametrize(
    ("argv", "code"),
    [
        pytest.param(["research", "shift", "--help"], 0, id="shift-help"),
        pytest.param(["research", "shift", "characterize", "--help"], 0, id="characterize-help"),
        pytest.param(["research", "shift", "run", "--help"], 0, id="run-help"),
        pytest.param(["research", "shift", "characterize"], 2, id="characterize-missing-args"),
        pytest.param(
            ["research", "shift", "run", "--model", "a", _ENGINE], 2, id="run-missing-args"
        ),
    ],
)
def test_shift_cli_exits(argv, code) -> None:
    """--help exits 0 and missing required arguments exit 2."""
    with patch("sys.argv", ["trtutils", *argv]), pytest.raises(SystemExit) as exc_info:
        _main()
    assert exc_info.value.code == code


def test_shift_run_arguments_reach_handler() -> None:
    """Run parses repeated --model specs, the preset and per-field overrides."""
    argv = [
        "trtutils", "research", "shift", "run",
        "--model", "big", _ENGINE,
        "--model", "small_dla", _ENGINE, "1",
        "--characterization", _JSON,
        "--input", _VIDEO,
        "--preset", "shiftdetector",
        "--momentum", "5",
        "--similarity", "min",
    ]  # fmt: skip
    with patch("sys.argv", argv), patch("trtutils.research.shift._cli._run") as mock:
        _main()
    args = mock.call_args[0][0]
    assert args.model == [["big", _ENGINE], ["small_dla", _ENGINE, "1"]]
    assert args.memory_budget is None
    config = _make_config(args)
    expected = ShiftConfig.shiftdetector()
    assert config == ShiftConfig(**{**vars(expected), "momentum": 5, "similarity": "min"})


def test_shift_characterize_arguments_reach_handler() -> None:
    """Characterize parses its paths and benchmark iteration defaults."""
    argv = [
        "trtutils", "research", "shift", "characterize",
        "--model", "big", _ENGINE,
        "--images", "/fake/images",
        "--labels", "/fake/labels",
        "--output", _JSON,
    ]  # fmt: skip
    with patch("sys.argv", argv), patch("trtutils.research.shift._cli._characterize") as mock:
        _main()
    args = mock.call_args[0][0]
    assert (args.images, args.labels, args.output) == ("/fake/images", "/fake/labels", _JSON)
    assert (args.iterations, args.warmup_iterations) == (1000, 50)


def test_make_config_defaults_to_paper() -> None:
    """Without a preset or overrides the config is the paper's."""
    args = SimpleNamespace(preset="paper", **dict.fromkeys(_CONFIG_FIELDS))
    assert _make_config(args) == ShiftConfig()


@pytest.mark.parametrize(
    ("specs", "match"),
    [
        pytest.param([["only_name"]], "NAME ENGINE", id="too-few"),
        pytest.param([["a", _ENGINE, "0", "extra"]], "NAME ENGINE", id="too-many"),
        pytest.param([["a", _ENGINE], ["a", _ENGINE]], "more than once", id="duplicate"),
    ],
)
def test_parse_models_invalid(specs, match) -> None:
    """Malformed or repeated --model specs raise ValueError."""
    with pytest.raises(ValueError, match=match):
        _parse_models(specs)


def test_parse_models_dla_disables_cuda_graph() -> None:
    """A DLA core in the spec sets dla_core and turns CUDA graphs off."""
    models = _parse_models([["gpu", _ENGINE], ["dla", _ENGINE, "1"]])
    assert models["gpu"].keywords["dla_core"] is None
    assert models["gpu"].keywords["cuda_graph"] is None
    assert models["dla"].keywords["dla_core"] == 1
    assert models["dla"].keywords["cuda_graph"] is False


def test_read_labels(tmp_path) -> None:
    """YOLO-format labels convert to pixel xyxy boxes; a missing file means no boxes."""
    label = tmp_path / "frame.txt"
    label.write_text("0 0.5 0.5 0.5 0.25\n\n3 0.1 0.2 0.2 0.2\n")
    assert _read_labels(label, 200, 100) == [(50, 38, 150, 62), (0, 10, 40, 30)]
    assert _read_labels(tmp_path / "missing.txt", 200, 100) == []
