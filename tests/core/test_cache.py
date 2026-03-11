# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/cache.py -- engine cache operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from trtutils.core import cache
from trtutils.core.cache import get_cache_dir


@pytest.mark.cpu
def test_get_cache_dir() -> None:
    """get_cache_dir() returns a stable Path with correct name and existing parent."""
    result = get_cache_dir()
    assert isinstance(result, Path)
    assert result.name == "_engine_cache"
    assert result == get_cache_dir()
    assert result.parent.exists()


@pytest.mark.cpu
def test_store_query_remove_engine(patched_cache_dir, tmp_path) -> None:
    """Full lifecycle: store → query (True) → remove → query (False)."""
    src = tmp_path / "test.engine"
    src.write_text("engine data")
    cache.store(src)
    exists, path = cache.query("test")
    assert exists is True
    assert path.exists()
    cache.remove("test")
    exists, path = cache.query("test")
    assert exists is False
    assert not path.exists()


@pytest.mark.cpu
def test_store_file_query_file_remove_file(patched_cache_dir, tmp_path) -> None:
    """Full lifecycle with store_file/query_file/remove_file."""
    src = tmp_path / "data.txt"
    src.write_text("some data")
    new_path = cache.store_file(src, cache_filename="data.cache")
    assert new_path.exists()
    exists, _ = cache.query_file("data", "cache")
    assert exists is True
    cache.remove_file("data", "cache")
    exists, _ = cache.query_file("data", "cache")
    assert exists is False


@pytest.mark.cpu
def test_store_file_uses_original_name(patched_cache_dir, tmp_path) -> None:
    """store_file without cache_filename uses the source filename."""
    src = tmp_path / "original.engine"
    src.write_text("data")
    result = cache.store_file(src)
    assert result.name == "original.engine"
    assert result.exists()


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("overwrite", "expected_content"),
    [
        pytest.param(True, "version2", id="overwrite"),
        pytest.param(False, "version1", id="no_overwrite"),
    ],
)
def test_store_overwrite(patched_cache_dir, tmp_path, overwrite, expected_content) -> None:
    """store_file overwrite parameter controls replacement behavior."""
    src1 = tmp_path / "v1.engine"
    src1.write_text("version1")
    src2 = tmp_path / "v2.engine"
    src2.write_text("version2")
    cache.store_file(src1, cache_filename="model.engine")
    cache.store_file(src2, cache_filename="model.engine", overwrite=overwrite)
    stored = patched_cache_dir / "model.engine"
    assert stored.read_text() == expected_content


@pytest.mark.cpu
def test_store_clear_old(patched_cache_dir, tmp_path) -> None:
    """Store with clear_old=True removes the source file."""
    src = tmp_path / "model.engine"
    src.write_text("data")
    cache.store(src, clear_old=True)
    assert not src.exists()
    exists, _ = cache.query("model")
    assert exists is True


@pytest.mark.cpu
def test_clear_recreates_empty_dir(patched_cache_dir, tmp_path) -> None:
    """clear() removes contents and recreates the directory."""
    src = tmp_path / "test.engine"
    src.write_text("data")
    cache.store(src)
    exists, _ = cache.query("test")
    assert exists is True
    cache.clear(no_warn=True)
    assert patched_cache_dir.exists()
    assert list(patched_cache_dir.iterdir()) == []
    exists, _ = cache.query("test")
    assert exists is False


@pytest.mark.cpu
def test_remove_nonexistent_raises(patched_cache_dir) -> None:
    """remove/remove_file raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        cache.remove("nonexistent")
    with pytest.raises(FileNotFoundError):
        cache.remove_file("nonexistent", "cache")


@pytest.mark.cpu
def test_timing_cache_lifecycle(patched_cache_dir, tmp_path) -> None:
    """Timing cache: store → query → overwrite → clear_old → remove → query (False)."""
    # query missing
    exists, path = cache.query_timing_cache()
    assert exists is False
    assert path.name == "global.cache"
    # store then query
    src = tmp_path / "timing.cache"
    src.write_bytes(b"timing data bytes")
    cache.store_timing_cache(src)
    exists, path = cache.query_timing_cache()
    assert exists is True
    assert path.read_bytes() == b"timing data bytes"
    # overwrite
    src2 = tmp_path / "v2.cache"
    src2.write_bytes(b"v2")
    cache.store_timing_cache(src2, overwrite=True)
    _, path = cache.query_timing_cache()
    assert path.read_bytes() == b"v2"
    # clear_old
    src3 = tmp_path / "v3.cache"
    src3.write_bytes(b"v3")
    cache.store_timing_cache(src3, clear_old=True)
    assert not src3.exists()


@pytest.mark.cpu
def test_save_timing_cache_to_global(patched_cache_dir) -> None:
    """save_timing_cache_to_global serializes and stores, respects overwrite."""

    class MockTimingCache:
        def __init__(self, data: bytes = b"serialized cache data") -> None:
            self._data = data

        def serialize(self) -> bytes:
            return self._data

    result = cache.save_timing_cache_to_global(MockTimingCache())
    assert result.exists()
    assert result.name == "global.cache"
    exists, path = cache.query_timing_cache()
    assert exists is True
    assert path.read_bytes() == b"serialized cache data"
    # overwrite then no-overwrite
    cache.save_timing_cache_to_global(MockTimingCache(b"second"), overwrite=True)
    cache.save_timing_cache_to_global(MockTimingCache(b"third"), overwrite=False)
    _, path = cache.query_timing_cache()
    assert path.read_bytes() == b"second"
