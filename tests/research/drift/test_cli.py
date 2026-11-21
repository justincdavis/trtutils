# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/drift/_cli.py -- the research drift subcommands."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from trtutils.__main__ import _main
from trtutils.research.drift import DriftConfig
from trtutils.research.drift._cli import _make_config, _parse_models, _read_labels

pytestmark = pytest.mark.cpu

_ENGINE = "/fake/model.engine"
_JSON = "/fake/traits.json"
_VIDEO = "/fake/video.mp4"
_CONFIG_FIELDS = [f.name for f in dataclasses.fields(DriftConfig)]


@pytest.mark.parametrize(
    ("argv", "code"),
    [
        pytest.param(["research", "drift", "--help"], 0, id="drift-help"),
        pytest.param(["research", "drift", "characterize", "--help"], 0, id="characterize-help"),
        pytest.param(["research", "drift", "run", "--help"], 0, id="run-help"),
        pytest.param(["research", "drift", "characterize"], 2, id="characterize-missing-args"),
        pytest.param(
            ["research", "drift", "run", "--model", "a", _ENGINE, "--input", _VIDEO],
            2,
            id="run-missing-args",
        ),
    ],
)
def test_drift_cli_exits(argv, code) -> None:
    """--help exits 0 and missing required arguments exit 2."""
    with patch("sys.argv", ["trtutils", *argv]), pytest.raises(SystemExit) as exc_info:
        _main()
    assert exc_info.value.code == code


def test_drift_run_arguments_reach_handler() -> None:
    """Run parses repeated specs, priorities, the preset and per-field overrides of every kind."""
    argv = [
        "trtutils", "research", "drift", "run",
        "--model", "base", _ENGINE,
        "--model", "yolox_dla", _ENGINE, "0",
        "--input_range", "yolox_dla", "0", "255",
        "--characterization", _JSON,
        "--input", _VIDEO,
        "--priority", "0", "2",
        "--seed", "3",
        "--preset", "upstream",
        "--hp_padding", "32",
        "--dense_groups", "rectangles",
        "--no_hp_square",
        "--hp_trusted_models", "a", "b",
        "--merge_iou", "0.6",
    ]  # fmt: skip
    with patch("sys.argv", argv), patch("trtutils.research.drift._cli._run") as mock:
        _main()
    args = mock.call_args[0][0]
    assert args.model == [["base", _ENGINE], ["yolox_dla", _ENGINE, "0"]]
    assert args.input_range == [["yolox_dla", "0", "255"]]
    assert (args.priority, args.seed, args.baseline) == ([0, 2], 3, None)
    expected = dataclasses.replace(
        DriftConfig.upstream(),
        hp_padding=32,
        dense_groups="rectangles",
        hp_square=False,
        hp_trusted_models=("a", "b"),
        merge_iou=0.6,
    )
    assert _make_config(args) == expected


def test_drift_characterize_arguments_reach_handler() -> None:
    """Characterize parses its paths and timing defaults."""
    argv = [
        "trtutils", "research", "drift", "characterize",
        "--model", "base", _ENGINE,
        "--images", "/fake/images",
        "--labels", "/fake/labels",
        "--output", _JSON,
    ]  # fmt: skip
    with patch("sys.argv", argv), patch("trtutils.research.drift._cli._characterize") as mock:
        _main()
    args = mock.call_args[0][0]
    assert (args.images, args.labels, args.output) == ("/fake/images", "/fake/labels", _JSON)
    assert (args.iterations, args.warmup_iterations, args.iou_threshold) == (200, 20, 0.5)
    assert (args.latency_input, args.latency_frames, args.input_range) == (None, 30, None)


def test_make_config_defaults_to_paper() -> None:
    """Without a preset or overrides the config is the paper's."""
    args = SimpleNamespace(preset="paper", **dict.fromkeys(_CONFIG_FIELDS))
    assert _make_config(args) == DriftConfig()


@pytest.mark.parametrize(
    ("specs", "ranges", "match"),
    [
        pytest.param([["only_name"]], None, "NAME ENGINE", id="too-few"),
        pytest.param([["a", _ENGINE, "0", "extra"]], None, "NAME ENGINE", id="too-many"),
        pytest.param([["a", _ENGINE], ["a", _ENGINE]], None, "more than once", id="duplicate"),
        pytest.param([["a", _ENGINE]], [["b", "0", "255"]], "unknown models", id="range-name"),
    ],
)
def test_parse_models_invalid(specs, ranges, match) -> None:
    """Malformed or repeated --model specs, and ranges for unknown models, raise ValueError."""
    with pytest.raises(ValueError, match=match):
        _parse_models(specs, ranges)


def test_parse_models_dla_and_input_range() -> None:
    """A DLA core turns CUDA graphs off; an input range reaches only its model."""
    models = _parse_models([["gpu", _ENGINE], ["dla", _ENGINE, "1"]], [["dla", "0", "255"]])
    assert models["gpu"].keywords["dla_core"] is None
    assert models["gpu"].keywords["cuda_graph"] is None
    assert "input_range" not in models["gpu"].keywords
    assert models["dla"].keywords["dla_core"] == 1
    assert models["dla"].keywords["cuda_graph"] is False
    assert models["dla"].keywords["input_range"] == (0.0, 255.0)


def test_read_labels(tmp_path) -> None:
    """YOLO-format labels convert to pixel boxes with class ids; a missing file means none."""
    label = tmp_path / "frame.txt"
    label.write_text("0 0.5 0.5 0.5 0.25\n\n3 0.1 0.2 0.2 0.2\n")
    assert _read_labels(label, 200, 100) == [((50, 38, 150, 62), 0), ((0, 10, 40, 30), 3)]
    assert _read_labels(tmp_path / "missing.txt", 200, 100) == []
    label.write_text("0 0.5 0.5 0.5 0.25\n0 0.1 0.1 0.2 0.1 0.3 0.3\n")
    with pytest.raises(ValueError, match=r"frame.txt:2: expected 'class cx cy w h'"):
        _read_labels(label, 200, 100)
