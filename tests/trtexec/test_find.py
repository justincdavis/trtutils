# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for trtutils.trtexec._find -- find_trtexec."""

from __future__ import annotations

import pytest

# ============================================================================
# Jetson hardware tests
# ============================================================================


@pytest.mark.jetson
class TestFindTrtexecOnJetson:
    """Test find_trtexec on actual Jetson hardware."""

    def test_returns_existing_path(self) -> None:
        """find_trtexec returns an existing Path on Jetson."""
        from trtutils.trtexec import find_trtexec

        path = find_trtexec()
        assert path.exists()
        assert path.name == "trtexec"


# ============================================================================
# Mocked filesystem tests
# ============================================================================


@pytest.mark.cpu
class TestFindTrtexecMocked:
    """Test find_trtexec with mocked filesystem paths."""

    def teardown_method(self) -> None:
        """Clear lru_cache after each test to avoid cross-test contamination."""
        from trtutils.trtexec._find import find_trtexec

        find_trtexec.cache_clear()

    def test_not_found_raises_file_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FileNotFoundError raised when no trtexec binary exists."""
        from unittest.mock import MagicMock

        from trtutils.trtexec._find import find_trtexec

        # Make all candidate directories report as non-existent
        mock_path_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.exists.return_value = False
        mock_path_cls.return_value = mock_instance

        monkeypatch.setattr("trtutils.trtexec._find.Path", mock_path_cls)

        with pytest.raises(FileNotFoundError, match="trtexec binary not found"):
            find_trtexec()

    def test_first_match_returned(self, tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
        """find_trtexec returns the first matching trtexec binary."""
        from pathlib import Path

        from trtutils.trtexec._find import find_trtexec

        # Create a fake trtexec in tmp_path
        fake_dir = Path(str(tmp_path)) / "tensorrt" / "bin"
        fake_dir.mkdir(parents=True)
        fake_trtexec = fake_dir / "trtexec"
        fake_trtexec.write_text("fake binary")

        # Patch the possible_dirs list inside the function by replacing Path
        # so that the first directory resolves to our fake dir
        original_path = Path

        def patched_path(p: str) -> Path:
            if p == "/usr/src/tensorrt/bin":
                return original_path(str(fake_dir))
            return original_path(p)

        monkeypatch.setattr("trtutils.trtexec._find.Path", patched_path)

        result = find_trtexec()
        assert result == fake_trtexec
        assert result.exists()

    def test_lru_cache_returns_same_result(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repeated calls return the cached result without re-scanning."""
        from pathlib import Path

        from trtutils.trtexec._find import find_trtexec

        fake_dir = Path(str(tmp_path)) / "tensorrt" / "bin"
        fake_dir.mkdir(parents=True)
        fake_trtexec = fake_dir / "trtexec"
        fake_trtexec.write_text("fake binary")

        original_path = Path

        call_count = 0

        def patched_path(p: str) -> Path:
            nonlocal call_count
            call_count += 1
            if p == "/usr/src/tensorrt/bin":
                return original_path(str(fake_dir))
            return original_path(p)

        monkeypatch.setattr("trtutils.trtexec._find.Path", patched_path)

        result1 = find_trtexec()
        first_call_count = call_count
        result2 = find_trtexec()

        assert result1 is result2
        # Second call should hit the cache and not increment call_count
        assert call_count == first_call_count
