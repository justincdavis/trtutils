# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/haxconn/_cli.py -- the research haxconn subcommands."""

from __future__ import annotations

import dataclasses
import json
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


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param(["none"], None, id="none"),
        pytest.param(["10", "50"], (10.0, 50.0), id="times"),
    ],
)
def test_update_ms_flag(values, expected) -> None:
    """--update_ms takes milliseconds or 'none' (switch on every improvement)."""
    argv = ["trtutils", "research", "haxconn", "solve", "--model", "a", _JSON, "--update_ms"]
    with patch("sys.argv", [*argv, *values]), patch("trtutils.research.haxconn._cli._solve") as mock:
        _main()
    assert _make_config(mock.call_args[0][0]).update_ms == expected


@pytest.mark.parametrize("command", ["prebuild", "dynamic"])
def test_dynamic_commands_help(command) -> None:
    """The D-HaX-CoNN commands parse and print help."""
    with patch("sys.argv", ["trtutils", "research", "haxconn", command, "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            _main()
    assert exc_info.value.code == 0


def test_dynamic_runs_each_workload_and_writes_a_timeline(tmp_path, paper_traits, fakes) -> None:
    """Dynamic runs every workload for its period and summarizes each in the JSON output."""
    argv = ["trtutils", "research", "haxconn", "dynamic", "--contention", "none"]
    for name, traits in paper_traits.items():
        save_traits(traits, tmp_path / f"{name}.json")
        (tmp_path / f"{name}.onnx").write_bytes(b"")
        argv += ["--model", name, str(tmp_path / f"{name}.onnx"), str(tmp_path / f"{name}.json")]
    argv += [
        "--workload", "googlenet,alexnet",
        "--workload", "alexnet",
        "--period", "0.3",
        "--cycles", "2",
        "--solver", "exhaustive",
        "--update_ms", "none",
        "--prebuild",
        "--build", "prebuilt",
        "--engine_dir", str(tmp_path / "engines"),
        "--output", str(tmp_path / "timeline.json"),
    ]  # fmt: skip
    with patch("sys.argv", argv):
        _main()
    result = json.loads((tmp_path / "timeline.json").read_text())
    segments = result["segments"]
    assert [(s["cycle"], s["workload"]) for s in segments] == [
        (0, "googlenet,alexnet"),
        (0, "alexnet"),
        (1, "googlenet,alexnet"),
        (1, "alexnet"),
    ]
    assert all(s["rounds"] > 0 for s in segments)
    # the second time a set runs it starts on the schedule found the first time
    assert segments[2]["first"] == segments[0]["final"]
    assert segments[2]["switch_s"] == pytest.approx(0.0, abs=0.05)
    assert len(result["rounds"]) == sum(s["rounds"] for s in segments)
