# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/cache.py -- engine cache operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from trtutils.core import cache
from trtutils.core.cache import _delete_folder, _get_cache_file_path, get_cache_dir


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
    ("filename", "expected_suffix"),
    [
        pytest.param("mymodel", "mymodel.engine", id="no_ext"),
        pytest.param("mymodel.engine", "mymodel.engine", id="engine_ext"),
        pytest.param("mymodel.onnx", "mymodel.onnx", id="onnx_ext"),
        pytest.param("global.cache", "global.cache", id="cache_ext"),
        pytest.param("mymodel.txt", "mymodel.txt.engine", id="invalid_ext"),
        pytest.param("my.weird.model", "my.weird.model.engine", id="multi_dot"),
    ],
)
def test_cache_file_path_extension(filename: str, expected_suffix: str) -> None:
    """_get_cache_file_path handles extension logic correctly."""
    result = _get_cache_file_path(filename, "engine")
    assert result == get_cache_dir() / expected_suffix


def _setup_empty(d: Path) -> None:
    d.mkdir()


def _setup_files(d: Path) -> None:
    d.mkdir()
    (d / "a.txt").write_text("hello")
    (d / "b.txt").write_text("world")


def _setup_nested(d: Path) -> None:
    d.mkdir()
    sub = d / "sub"
    sub.mkdir()
    (sub / "deep.txt").write_text("content")


def _setup_mixed(d: Path) -> None:
    d.mkdir()
    (d / "file.txt").write_text("data")
    sub = d / "child"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested data")


@pytest.mark.cpu
@pytest.mark.parametrize(
    "setup_fn",
    [
        pytest.param(_setup_empty, id="empty"),
        pytest.param(_setup_files, id="with_files"),
        pytest.param(_setup_nested, id="with_subdirs"),
        pytest.param(_setup_mixed, id="mixed"),
    ],
)
def test_delete_folder(tmp_path, setup_fn) -> None:
    """_delete_folder removes directory trees."""
    d = tmp_path / "target"
    setup_fn(d)
    _delete_folder(d)
    assert not d.exists()


@pytest.mark.cpu
@pytest.mark.parametrize(
    "setup_fn",
    [
        pytest.param(lambda _d: None, id="empty"),
        pytest.param(lambda d: (d / "test.engine").write_text("data"), id="with_file"),
        pytest.param(
            lambda d: (d / "sub").mkdir() or (d / "sub" / "f.txt").write_text("t"),
            id="nested",
        ),
    ],
)
def test_clear_recreates_empty_dir(patched_cache_dir, setup_fn) -> None:
    """clear() removes contents and recreates the directory."""
    setup_fn(patched_cache_dir)
    cache.clear(no_warn=True)
    assert patched_cache_dir.exists()
    assert list(patched_cache_dir.iterdir()) == []


@pytest.mark.cpu
@pytest.mark.parametrize(
    "no_warn",
    [
        pytest.param(None, id="default"),
        pytest.param(False, id="explicit_false"),
    ],
)
def test_clear_warning_path(patched_cache_dir, no_warn) -> None:
    """clear() with default/explicit no_warn doesn't crash."""
    kwargs = {} if no_warn is None else {"no_warn": no_warn}
    cache.clear(**kwargs)
    assert patched_cache_dir.exists()


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("query_fn", "query_args", "expected_name"),
    [
        pytest.param(cache.query, ("nofile",), None, id="query"),
        pytest.param(cache.query_file, ("nofile", "cache"), "nofile.cache", id="query_file"),
    ],
)
def test_query_nonexistent(patched_cache_dir, query_fn, query_args, expected_name) -> None:
    """query/query_file returns (False, path) for missing files."""
    exists, path = query_fn(*query_args)
    assert exists is False
    assert isinstance(path, Path)
    if expected_name:
        assert path.name == expected_name


@pytest.mark.cpu
def test_store_then_query(patched_cache_dir, tmp_path) -> None:
    """store() then query() finds the stored file."""
    src = tmp_path / "test.engine"
    src.write_text("engine data")
    cache.store(src)
    exists, path = cache.query("test")
    assert exists is True
    assert path.exists()


@pytest.mark.cpu
def test_store_file_then_query_file(patched_cache_dir, tmp_path) -> None:
    """store_file() then query_file() finds the stored file."""
    src = tmp_path / "data.txt"
    src.write_text("some data")
    new_path = cache.store_file(src, cache_filename="data.cache")
    assert new_path.exists()
    exists, _path = cache.query_file("data", "cache")
    assert exists is True


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
    # basic clear_old
    src = tmp_path / "model.engine"
    src.write_text("data")
    cache.store(src, clear_old=True)
    assert not src.exists()
    # clear_old with no overwrite on existing
    (patched_cache_dir / "model2.engine").write_text("old")
    src2 = tmp_path / "model2.engine"
    src2.write_text("new")
    cache.store(src2, overwrite=False, clear_old=True)
    assert not src2.exists()
    assert (patched_cache_dir / "model2.engine").read_text() == "old"
    # clear_old with overwrite
    (patched_cache_dir / "model3.engine").write_text("old")
    src3 = tmp_path / "model3.engine"
    src3.write_text("new")
    cache.store(src3, overwrite=True, clear_old=True)
    assert not src3.exists()
    assert (patched_cache_dir / "model3.engine").read_text() == "new"


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
    ("setup_name", "remove_fn", "remove_args"),
    [
        pytest.param("model.engine", cache.remove, ("model",), id="remove"),
        pytest.param("data.cache", cache.remove_file, ("data", "cache"), id="remove_file"),
        pytest.param(
            "model.engine",
            cache.remove_file,
            ("model.engine", "engine"),
            id="remove_file_ext_in_name",
        ),
    ],
)
def test_remove_existing(patched_cache_dir, setup_name, remove_fn, remove_args) -> None:
    """remove/remove_file deletes cached files."""
    (patched_cache_dir / setup_name).write_text("data")
    remove_fn(*remove_args)
    assert not (patched_cache_dir / setup_name).exists()


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("remove_fn", "remove_args"),
    [
        pytest.param(cache.remove, ("nonexistent",), id="remove"),
        pytest.param(cache.remove_file, ("nonexistent", "cache"), id="remove_file"),
    ],
)
def test_remove_nonexistent_raises(patched_cache_dir, remove_fn, remove_args) -> None:
    """remove/remove_file raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        remove_fn(*remove_args)


@pytest.mark.cpu
def test_timing_cache_roundtrip(patched_cache_dir, tmp_path) -> None:
    """Timing cache store, query, overwrite, and clear_old lifecycle."""
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

    # basic store
    result = cache.save_timing_cache_to_global(MockTimingCache())
    assert result.exists()
    assert result.name == "global.cache"
    exists, path = cache.query_timing_cache()
    assert exists is True
    assert path.read_bytes() == b"serialized cache data"
    # overwrite behavior
    cache.save_timing_cache_to_global(MockTimingCache(b"first"), overwrite=True)
    cache.save_timing_cache_to_global(MockTimingCache(b"second"), overwrite=True)
    _, path = cache.query_timing_cache()
    assert path.read_bytes() == b"second"
    # no overwrite
    cache.save_timing_cache_to_global(MockTimingCache(b"third"), overwrite=False)
    _, path = cache.query_timing_cache()
    assert path.read_bytes() == b"second"
