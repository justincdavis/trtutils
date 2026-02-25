"""Tests for ProgressBar -- TRT build progress monitoring."""

from __future__ import annotations

import pytest


@pytest.mark.cpu
class TestProgressBar:
    """Tests for ProgressBar class."""

    def test_init(self) -> None:
        """ProgressBar initializes without error."""
        from trtutils.builder._progress import ProgressBar

        pb = ProgressBar()
        assert pb._progress_bars == {}
        assert pb._interrupted is False

    def test_phase_start(self) -> None:
        """phase_start creates a progress bar entry."""
        from trtutils.builder._progress import ProgressBar

        pb = ProgressBar()
        pb.phase_start("test_phase", None, 100)
        assert "test_phase" in pb._progress_bars
        assert pb._last_steps["test_phase"] == 0
        pb.phase_finish("test_phase")

    def test_phase_start_with_parent(self) -> None:
        """phase_start with parent sets correct indentation."""
        from trtutils.builder._progress import ProgressBar

        pb = ProgressBar()
        pb.phase_start("parent", None, 50)
        pb.phase_start("child", "parent", 25)
        assert pb._indentation_levels["parent"] == 0
        assert pb._indentation_levels["child"] == 1
        pb.phase_finish("child")
        pb.phase_finish("parent")

    def test_step_complete(self) -> None:
        """step_complete updates progress."""
        from trtutils.builder._progress import ProgressBar

        pb = ProgressBar()
        pb.phase_start("phase", None, 10)
        result = pb.step_complete("phase", 5)
        assert result is True
        assert pb._last_steps["phase"] == 5
        pb.phase_finish("phase")

    def test_step_complete_returns_true(self) -> None:
        """step_complete returns True when not interrupted."""
        from trtutils.builder._progress import ProgressBar

        pb = ProgressBar()
        pb.phase_start("phase", None, 10)
        assert pb.step_complete("phase", 1) is True
        pb.phase_finish("phase")

    def test_step_complete_no_regression(self) -> None:
        """step_complete ignores steps <= last step."""
        from trtutils.builder._progress import ProgressBar

        pb = ProgressBar()
        pb.phase_start("phase", None, 10)
        pb.step_complete("phase", 5)
        pb.step_complete("phase", 3)  # Should be ignored
        assert pb._last_steps["phase"] == 5
        pb.phase_finish("phase")

    def test_phase_finish_cleanup(self) -> None:
        """phase_finish removes the phase from tracking."""
        from trtutils.builder._progress import ProgressBar

        pb = ProgressBar()
        pb.phase_start("phase", None, 10)
        pb.phase_finish("phase")
        assert "phase" not in pb._progress_bars
        assert "phase" not in pb._phase_parents
        assert "phase" not in pb._indentation_levels
        assert "phase" not in pb._last_steps

    def test_del_closes_bars(self) -> None:
        """__del__ closes all progress bars."""
        from trtutils.builder._progress import ProgressBar

        pb = ProgressBar()
        pb.phase_start("phase", None, 10)
        del pb  # Should not crash


@pytest.mark.cpu
class TestProgressBarEdgeCases:
    """Edge case tests for ProgressBar."""

    def test_step_zero_delta(self) -> None:
        """step_diff=0 produces no update."""
        from trtutils.builder._progress import ProgressBar

        pb = ProgressBar()
        pb.phase_start("phase", None, 10)
        pb.step_complete("phase", 5)
        # Calling with same step value → step_diff=0, no update
        pb.step_complete("phase", 5)
        assert pb._last_steps["phase"] == 5
        pb.phase_finish("phase")

    def test_unknown_phase_step(self) -> None:
        """step_complete on unknown phase silently does nothing."""
        from trtutils.builder._progress import ProgressBar

        pb = ProgressBar()
        # Should not raise — the `if phase_name in self._progress_bars` guard handles it
        result = pb.step_complete("nonexistent_phase", 1)
        assert result is True  # Not interrupted

    def test_phase_finish_unknown_phase(self) -> None:
        """phase_finish on unknown phase does not raise."""
        from trtutils.builder._progress import ProgressBar

        pb = ProgressBar()
        # Should not raise — all cleanup is guarded by `if phase_name in`
        pb.phase_finish("nonexistent")

    def test_nested_phases_indentation(self) -> None:
        """Nested phases get increasing indentation levels."""
        from trtutils.builder._progress import ProgressBar

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

    def test_parent_not_in_levels(self) -> None:
        """phase_start with unknown parent defaults to indentation 0."""
        from trtutils.builder._progress import ProgressBar

        pb = ProgressBar()
        pb.phase_start("child", "unknown_parent", 10)
        assert pb._indentation_levels["child"] == 0
        pb.phase_finish("child")
