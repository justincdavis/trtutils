# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for trtutils.trtexec._run -- run_trtexec."""

from __future__ import annotations

import pytest

# ============================================================================
# Jetson hardware tests
# ============================================================================


@pytest.mark.jetson
class TestRunTrtexecOnJetson:
    """Test run_trtexec on actual Jetson hardware."""

    def test_help_succeeds(self) -> None:
        """run_trtexec('--help') returns success on Jetson."""
        from trtutils.trtexec import run_trtexec

        success, stdout, stderr = run_trtexec("--help")

        assert success is True
        assert isinstance(stdout, str)
        assert isinstance(stderr, str)


# ============================================================================
# Mocked subprocess tests
# ============================================================================


@pytest.mark.cpu
class TestRunTrtexecMocked:
    """Test run_trtexec with mocked subprocess and find_trtexec."""

    def test_success_returns_true_and_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Successful trtexec run returns (True, stdout, stderr)."""
        from pathlib import Path
        from unittest.mock import MagicMock

        from trtutils.trtexec import _run as run_module

        # Mock find_trtexec to return a fake path
        monkeypatch.setattr(run_module, "find_trtexec", lambda: Path("/fake/trtexec"))

        # Mock subprocess.run to simulate success
        mock_process = MagicMock()
        mock_process.stdout = b"trtexec output here"
        mock_process.stderr = b""
        monkeypatch.setattr(run_module.subprocess, "run", MagicMock(return_value=mock_process))

        success, stdout, stderr = run_module.run_trtexec("--help")

        assert success is True
        assert stdout == "trtexec output here"
        assert stderr == ""

    def test_failure_returns_false_and_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Failed trtexec run returns (False, stdout, stderr)."""
        import subprocess
        from pathlib import Path
        from unittest.mock import MagicMock

        from trtutils.trtexec import _run as run_module

        # Mock find_trtexec to return a fake path
        monkeypatch.setattr(run_module, "find_trtexec", lambda: Path("/fake/trtexec"))

        # Mock subprocess.run to raise CalledProcessError
        error = subprocess.CalledProcessError(1, "trtexec")
        error.stdout = b"partial output"
        error.stderr = b"error message"
        mock_run = MagicMock(side_effect=error)
        monkeypatch.setattr(run_module.subprocess, "run", mock_run)

        success, stdout, stderr = run_module.run_trtexec("--bad-arg")

        assert success is False
        assert stdout == "partial output"
        assert stderr == "error message"

    def test_custom_trtexec_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """run_trtexec uses the provided trtexec_path instead of find_trtexec."""
        from pathlib import Path
        from unittest.mock import MagicMock

        from trtutils.trtexec import _run as run_module

        mock_process = MagicMock()
        mock_process.stdout = b"custom path output"
        mock_process.stderr = b""
        mock_subprocess_run = MagicMock(return_value=mock_process)
        monkeypatch.setattr(run_module.subprocess, "run", mock_subprocess_run)

        success, stdout, _stderr = run_module.run_trtexec(
            "--help", trtexec_path=Path("/custom/trtexec")
        )

        assert success is True
        assert stdout == "custom path output"
        # Verify the custom path was used in the command
        call_args = mock_subprocess_run.call_args
        cmd_list = call_args[0][0]
        assert cmd_list[0] == "/custom/trtexec"

    def test_none_stdout_stderr_handled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When subprocess returns None for stdout/stderr, empty strings are returned."""
        from pathlib import Path
        from unittest.mock import MagicMock

        from trtutils.trtexec import _run as run_module

        monkeypatch.setattr(run_module, "find_trtexec", lambda: Path("/fake/trtexec"))

        mock_process = MagicMock()
        mock_process.stdout = None
        mock_process.stderr = None
        monkeypatch.setattr(run_module.subprocess, "run", MagicMock(return_value=mock_process))

        success, stdout, stderr = run_module.run_trtexec("--help")

        assert success is True
        assert stdout == ""
        assert stderr == ""
