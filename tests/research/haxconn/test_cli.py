# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/haxconn/_cli.py -- the research haxconn subcommands."""

from __future__ import annotations

import dataclasses
from unittest.mock import patch

import pytest

from trtutils.__main__ import _main
from trtutils.research.haxconn import HaxConnConfig, load_schedule, save_traits
from trtutils.research.haxconn._cli import _make_config

pytestmark = pytest.mark.cpu

_JSON = "/fake/traits.json"
_ONNX = "/fake/model.onnx"


@pytest.mark.parametrize(
    ("argv", "code"),
    [
        pytest.param(["research", "haxconn", "--help"], 0, id="haxconn-help"),
        pytest.param(["research", "haxconn", "characterize", "--help"], 0, id="characterize-help"),
        pytest.param(["research", "haxconn", "calibrate", "--help"], 0, id="calibrate-help"),
        pytest.param(["research", "haxconn", "solve", "--help"], 0, id="solve-help"),
        pytest.param(["research", "haxconn", "run", "--help"], 0, id="run-help"),
        pytest.param(["research", "haxconn", "characterize"], 2, id="characterize-missing"),
        pytest.param(["research", "haxconn", "solve"], 2, id="solve-missing"),
        pytest.param(["research", "haxconn", "run", "--model", "a", _ONNX], 2, id="run-short-model"),
        pytest.param(
            ["research", "haxconn", "solve", "--model", "a", _JSON, "--solver", "greedy"],
            2,
            id="bad-choice",
        ),
    ],
)
def test_haxconn_cli_exits(argv, code) -> None:
    """--help exits 0; missing or invalid arguments exit 2."""
    with patch("sys.argv", ["trtutils", *argv]), pytest.raises(SystemExit) as exc_info:
        _main()
    assert exc_info.value.code == code


def test_haxconn_run_arguments_reach_handler() -> None:
    """Run parses model triples, the preset and per-field overrides of every kind."""
    argv = [
        "trtutils", "research", "haxconn", "run",
        "--model", "a", _ONNX, _JSON,
        "--model", "b", _ONNX, _JSON,
        "--calibration", "/fake/calibration.json",
        "--dla_core", "1",
        "--dynamic",
        "--preset", "upstream",
        "--objective", "throughput",
        "--max_transitions", "none",
        "--time_budget", "5",
    ]  # fmt: skip
    with patch("sys.argv", argv), patch("trtutils.research.haxconn._cli._run") as mock:
        _main()
    args = mock.call_args[0][0]
    assert args.model == [["a", _ONNX, _JSON], ["b", _ONNX, _JSON]]
    assert (args.calibration, args.dla_core, args.dynamic) == ("/fake/calibration.json", 1, True)
    expected = dataclasses.replace(
        HaxConnConfig.upstream(), objective="throughput", max_transitions=None, time_budget=5.0
    )
    assert _make_config(args) == expected


def test_haxconn_solve_writes_a_schedule(tmp_path, paper_traits) -> None:
    """Solve reads traits, ignores contention when told to, and writes the schedule."""
    argv = ["trtutils", "research", "haxconn", "solve", "--contention", "none"]
    for name, traits in paper_traits.items():
        save_traits(traits, tmp_path / f"{name}.json")
        argv += ["--model", name, str(tmp_path / f"{name}.json")]
    argv += ["--solver", "exhaustive", "--output", str(tmp_path / "schedule.json")]
    with patch("sys.argv", argv):
        _main()
    schedule = load_schedule(tmp_path / "schedule.json")
    assert schedule.optimal is True
    assert set(schedule.assignments) == set(paper_traits)
