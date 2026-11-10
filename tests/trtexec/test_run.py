# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/trtexec/_run.py -- run_trtexec."""

from __future__ import annotations

import subprocess

import pytest

from trtutils.trtexec import run_trtexec


@pytest.mark.jetson
def test_run_trtexec_help_succeeds() -> None:
    """run_trtexec('--help') exits successfully and returns string streams."""
    success, stdout, stderr = run_trtexec("--help")
    assert success is True
    assert isinstance(stdout, str)
    assert isinstance(stderr, str)


@pytest.mark.jetson
def test_run_trtexec_bad_arg_returns_failure() -> None:
    """run_trtexec returns (False, stdout, stderr) when trtexec rejects an arg."""
    success, stdout, stderr = run_trtexec("--definitely-not-a-valid-flag-xyz")
    assert success is False
    assert isinstance(stdout, str)
    assert isinstance(stderr, str)


@pytest.mark.cpu
def test_run_trtexec_keeps_spaced_args_whole(monkeypatch) -> None:
    """A list command keeps an arg containing spaces as one argv element."""
    calls = []

    def fake_run(argv: list[str], **_kwargs: object):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_trtexec(["--onnx=/a b/m.onnx"], trtexec_path="trtexec")
    assert calls[0] == ["trtexec", "--onnx=/a b/m.onnx"]


@pytest.mark.cpu
def test_run_trtexec_string_command_uses_shlex(monkeypatch) -> None:
    """A string command is shlex-split, so quotes protect spaces."""
    calls = []

    def fake_run(argv: list[str], **_kwargs: object):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_trtexec('--onnx="/a b/m.onnx" --fp16', trtexec_path="trtexec")
    assert calls[0] == ["trtexec", "--onnx=/a b/m.onnx", "--fp16"]
