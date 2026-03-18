# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for ProgressBar -- TRT build progress monitoring."""

from __future__ import annotations

import pytest

from trtutils.builder._progress import ProgressBar


@pytest.mark.cpu
def test_init() -> None:
    """ProgressBar initializes without error."""
    pb = ProgressBar()
    assert pb._progress_bars == {}
    assert pb._interrupted is False


@pytest.mark.cpu
def test_phase_start() -> None:
    """phase_start creates a progress bar entry."""
    pb = ProgressBar()
    pb.phase_start("test_phase", None, 100)
    assert "test_phase" in pb._progress_bars
    assert pb._last_steps["test_phase"] == 0
    pb.phase_finish("test_phase")


@pytest.mark.cpu
def test_phase_start_with_parent() -> None:
    """phase_start with parent sets correct indentation."""
    pb = ProgressBar()
    pb.phase_start("parent", None, 50)
    pb.phase_start("child", "parent", 25)
    assert pb._indentation_levels["parent"] == 0
    assert pb._indentation_levels["child"] == 1
    pb.phase_finish("child")
    pb.phase_finish("parent")


@pytest.mark.cpu
def test_step_complete() -> None:
    """step_complete updates progress."""
    pb = ProgressBar()
    pb.phase_start("phase", None, 10)
    result = pb.step_complete("phase", 5)
    assert result is True
    assert pb._last_steps["phase"] == 5
    pb.phase_finish("phase")


@pytest.mark.cpu
def test_step_complete_returns_true() -> None:
    """step_complete returns True when not interrupted."""
    pb = ProgressBar()
    pb.phase_start("phase", None, 10)
    assert pb.step_complete("phase", 1) is True
    pb.phase_finish("phase")


@pytest.mark.cpu
def test_step_complete_no_regression() -> None:
    """step_complete ignores steps <= last step."""
    pb = ProgressBar()
    pb.phase_start("phase", None, 10)
    pb.step_complete("phase", 5)
    pb.step_complete("phase", 3)  # Should be ignored
    assert pb._last_steps["phase"] == 5
    pb.phase_finish("phase")


@pytest.mark.cpu
def test_phase_finish_cleanup() -> None:
    """phase_finish removes the phase from tracking."""
    pb = ProgressBar()
    pb.phase_start("phase", None, 10)
    pb.phase_finish("phase")
    assert "phase" not in pb._progress_bars
    assert "phase" not in pb._phase_parents
    assert "phase" not in pb._indentation_levels
    assert "phase" not in pb._last_steps


@pytest.mark.cpu
def test_del_closes_bars() -> None:
    """__del__ closes all progress bars."""
    pb = ProgressBar()
    pb.phase_start("phase", None, 10)
    del pb  # Should not crash


@pytest.mark.cpu
def test_step_zero_delta() -> None:
    """step_diff=0 produces no update."""
    pb = ProgressBar()
    pb.phase_start("phase", None, 10)
    pb.step_complete("phase", 5)
    pb.step_complete("phase", 5)
    assert pb._last_steps["phase"] == 5
    pb.phase_finish("phase")


@pytest.mark.cpu
def test_unknown_phase_step() -> None:
    """step_complete on unknown phase silently does nothing."""
    pb = ProgressBar()
    result = pb.step_complete("nonexistent_phase", 1)
    assert result is True


@pytest.mark.cpu
def test_phase_finish_unknown_phase() -> None:
    """phase_finish on unknown phase does not raise."""
    pb = ProgressBar()
    pb.phase_finish("nonexistent")


@pytest.mark.cpu
def test_nested_phases_indentation() -> None:
    """Nested phases get increasing indentation levels."""
    pb = ProgressBar()
    pb.phase_start("root", None, 100)
    pb.phase_start("level1", "root", 50)
    pb.phase_start("level2", "level1", 25)
    assert pb._indentation_levels["root"] == 0
    assert pb._indentation_levels["level1"] == 1
    assert pb._indentation_levels["level2"] == 2
    pb.phase_finish("level2")
    pb.phase_finish("level1")
    pb.phase_finish("root")


@pytest.mark.cpu
def test_parent_not_in_levels() -> None:
    """phase_start with unknown parent defaults to indentation 0."""
    pb = ProgressBar()
    pb.phase_start("child", "unknown_parent", 10)
    assert pb._indentation_levels["child"] == 0
    pb.phase_finish("child")
