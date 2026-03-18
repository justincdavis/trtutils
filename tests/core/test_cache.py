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
@pytest.mark.parametrize(
    ("use_cache_filename", "extension"),
    [
        pytest.param(False, "engine", id="engine-original-name"),
        pytest.param(True, "engine", id="engine-custom-name"),
        pytest.param(False, "cache", id="cache-original-name"),
        pytest.param(True, "cache", id="cache-custom-name"),
    ],
)
def test_store_query_remove_roundtrip(
    patched_cache_dir, tmp_path, use_cache_filename, extension
) -> None:
    """store_file → query_file → remove_file round-trip for various extensions."""
    src = tmp_path / f"test.{extension}"
    src.write_text("data")
    cache_filename = f"custom.{extension}" if use_cache_filename else None
    stored = cache.store_file(src, cache_filename=cache_filename)
    assert stored.exists()
    assert stored.read_text() == "data"
    query_name = "custom" if use_cache_filename else "test"
    exists, path = cache.query_file(query_name, extension)
    assert exists is True
    assert path == stored
    cache.remove_file(query_name, extension)
    exists, _ = cache.query_file(query_name, extension)
    assert exists is False


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("overwrite", "expected"),
    [
        pytest.param(True, "v2", id="overwrite"),
        pytest.param(False, "v1", id="no-overwrite"),
    ],
)
def test_store_overwrite(patched_cache_dir, tmp_path, overwrite, expected) -> None:
    """overwrite=True replaces existing; overwrite=False keeps original."""
    src1 = tmp_path / "v1.engine"
    src1.write_text("v1")
    src2 = tmp_path / "v2.engine"
    src2.write_text("v2")
    cache.store_file(src1, cache_filename="model.engine")
    cache.store_file(src2, cache_filename="model.engine", overwrite=overwrite)
    assert (patched_cache_dir / "model.engine").read_text() == expected


@pytest.mark.cpu
@pytest.mark.parametrize(
    "delete_source",
    [
        pytest.param(True, id="delete-source"),
        pytest.param(False, id="keep-source"),
    ],
)
def test_store_delete_source(patched_cache_dir, tmp_path, delete_source) -> None:
    """delete_source=True removes the source file; False keeps it."""
    src = tmp_path / "model.engine"
    src.write_text("data")
    cache.store(src, delete_source=delete_source)
    assert src.exists() is (not delete_source)
    exists, _ = cache.query("model")
    assert exists is True


@pytest.mark.cpu
def test_clear_recreates_empty_dir(patched_cache_dir, tmp_path) -> None:
    """clear() removes contents and recreates the directory."""
    src = tmp_path / "test.engine"
    src.write_text("data")
    cache.store(src)
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
def test_timing_cache_roundtrip(patched_cache_dir, tmp_path) -> None:
    """Timing cache: query miss → store → query hit → overwrite → verify content."""
    exists, path = cache.query_timing_cache()
    assert exists is False
    assert path.name == "global.cache"
    src = tmp_path / "timing.cache"
    src.write_bytes(b"v1")
    cache.store_timing_cache(src)
    exists, path = cache.query_timing_cache()
    assert exists is True
    assert path.read_bytes() == b"v1"
    src2 = tmp_path / "v2.cache"
    src2.write_bytes(b"v2")
    cache.store_timing_cache(src2, overwrite=True)
    _, path = cache.query_timing_cache()
    assert path.read_bytes() == b"v2"
