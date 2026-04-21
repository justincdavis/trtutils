# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for ProgressBar -- TRT build progress monitoring."""

from __future__ import annotations

import pytest

from trtutils._flags import FLAGS

if not FLAGS.BUILD_PROGRESS:
    pytest.skip(
        "ProgressBar requires trt.IProgressMonitor support",
        allow_module_level=True,
    )

from trtutils.builder._progress import ProgressBar


@pytest.mark.gpu
def test_progress_bar_lifecycle() -> None:
    """ProgressBar handles nested phases and step updates without error."""
    pb = ProgressBar()

    pb.phase_start("root", None, 100)
    pb.phase_start("child", "root", 50)
    assert pb._indentation_levels["root"] == 0
    assert pb._indentation_levels["child"] == 1

    assert pb.step_complete("child", 10) is True
    assert pb.step_complete("child", 5) is True  # regression ignored
    assert pb._last_steps["child"] == 10

    # unknown phase / parent are tolerated
    assert pb.step_complete("missing", 1) is True
    pb.phase_start("orphan", "unknown_parent", 5)
    assert pb._indentation_levels["orphan"] == 0

    pb.phase_finish("orphan")
    pb.phase_finish("child")
    pb.phase_finish("root")
    pb.phase_finish("missing")  # no-op
    assert pb._progress_bars == {}
